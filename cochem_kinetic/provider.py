"""
CoChem-BASE Module Provider Entry Point (`cochem.modules: kinetic`)
-------------------------------------------------------------------
Implements the versioned `cochem.module-provider/1` / `cochem.module-handoff/1`
contract for Phase 2 CoChem-BASE integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cochem_kinetic.chem312_specs import (
    STATIONARY_STATES,
    get_builtin_spec,
    list_builtin_specs,
    read_xyz,
    validate_stationary_stoichiometry,
)
from cochem_kinetic.engine.rrkm_theory import RRKMSolver
from cochem_kinetic.rxn_profile import compute_rrho_thermochemistry
from kinetic_core.thermo import apply_implicit_solvation_correction, calculate_eyring_rate

SUPPORTED_OPERATIONS: list[str] = [
    "rxn_profile",
    "validate_stoichiometry",
    "eyring_rate",
    "rrkm_rate",
    "thermochemistry",
]

_OPERATION_ALIASES: dict[str, str] = {
    "rrkm": "rrkm_rate",
}


def kinetic_provider() -> dict[str, Any]:
    """Canonical `cochem.modules` entry point metadata for CoChem-KINETIC."""
    return {
        "scientific_api_version": "cochem.module-provider/1",
        "module_id": "kinetic",
        "distribution": "cochem-kinetic",
        "integration_contract": "cochem.module-handoff/1",
        "operations": list(SUPPORTED_OPERATIONS),
        "builtin_specs": list_builtin_specs(),
        "execute_handoff": "cochem_kinetic.provider:execute_handoff",
    }


def execute_handoff(
    handoff_path: str | Path,
    output_directory: str | Path,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Executes a validated CoChem-BASE handoff JSON envelope and writes structured
    outputs into `output_directory`.
    """
    handoff_file = Path(handoff_path)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(handoff_file.read_text(encoding="utf-8"))
    raw_operation = str(payload.get("operation", "")).strip()
    operation = _OPERATION_ALIASES.get(raw_operation, raw_operation)
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported CoChem-KINETIC operation '{raw_operation}'. Supported: {SUPPORTED_OPERATIONS}."
        )

    params = payload.get("parameters", {}) or {}
    result: dict[str, Any]

    if operation == "validate_stoichiometry":
        spec_name = params.get("spec")
        spec = get_builtin_spec(spec_name) if spec_name else None
        charge = int(params.get("charge", spec["charge"] if spec else 0))
        mult = int(params.get("multiplicity", spec["multiplicity"] if spec else 1))
        expected_formula = params.get("expected_formula", spec["formula"] if spec else None)
        input_dir = Path(params["input_dir"])
        loaded: dict[str, dict[str, Any]] = {}
        for st in STATIONARY_STATES:
            p = input_dir / f"{st}.xyz"
            if p.is_file():
                syms, coords, _ = read_xyz(p)
                loaded[st] = {"symbols": syms, "coords": coords}
        result = validate_stationary_stoichiometry(
            loaded,
            expected_charge=charge,
            expected_multiplicity=mult,
            expected_formula=expected_formula,
            require_same_atom_order=True,
        )

    elif operation == "rrkm_rate":
        solver = RRKMSolver(
            frequencies=params["reactant_frequencies_cm1"],
            ts_frequencies=params["ts_frequencies_cm1"],
            e0_cm1=float(params["e0_cm1"]),
            reaction_degeneracy=float(params.get("reaction_degeneracy", 1.0)),
        )
        k_e = solver.compute_kE(
            energy=float(params["energy_cm1"]),
            dE=float(params.get("dE_cm1", 1.0)),
        )
        result = {
            "operation": "rrkm_rate",
            "energy_cm1": float(params["energy_cm1"]),
            "e0_cm1": float(params["e0_cm1"]),
            "k_E_s_inv": k_e,
        }

    elif operation in {"eyring_rate", "thermochemistry"}:
        if operation == "thermochemistry" and "xyz_path" in params and "frequencies_cm1" in params:
            syms, coords, _ = read_xyz(params["xyz_path"])
            thermo = compute_rrho_thermochemistry(
                symbols=syms,
                coords=coords,
                freqs_cm1=params["frequencies_cm1"],
                temp_k=float(params.get("temp_k", 298.15)),
                multiplicity=int(params.get("multiplicity", 1)),
            )
            result = {"operation": "thermochemistry", "thermo": thermo}
        else:
            dg_gas = float(params["delta_g_gas_kcal"])
            dg_eff = apply_implicit_solvation_correction(
                delta_g_gas=dg_gas,
                solvent_name=str(params.get("solvent", "acetonitrile")),
                delta_g_solv_ts=float(params.get("delta_g_solv_ts_kcal", 0.0)),
                delta_g_solv_reactants=float(params.get("delta_g_solv_reactants_kcal", 0.0)),
                phase=str(params.get("phase", "pcm")),
                gas_phase_ftmw_lock=bool(params.get("gas_phase_ftmw_lock", False)),
            )
            temp_k = float(params.get("temp_k", 298.15))
            k_tst = calculate_eyring_rate(
                dg_eff,
                wigner_coeff=float(params.get("wigner_coeff", 1.0)),
                temp=temp_k,
                reaction_order=int(params.get("reaction_order", 1)),
                pressure_atm=float(params.get("pressure_atm", 1.0)),
            )
            result = {
                "operation": operation,
                "delta_g_eff_kcal": dg_eff,
                "k_tst": k_tst,
                "temp_k": temp_k,
            }

    elif operation == "rxn_profile":
        from cochem_kinetic.cli import main as cli_main

        cli_args = [
            "rxn-profile",
            "--input-dir", str(params["input_dir"]),
            "--output-dir", str(out_dir),
            "--solvent", str(params.get("solvent", "acetonitrile")),
            "--temp-k", str(params.get("temp_k", 298.15)),
            "--dft-engine", str(params.get("dft_engine", "cpu")),
            "--stage", str(params.get("stage", "all")),
        ]
        if params.get("spec"):
            cli_args.extend(["--spec", str(params["spec"])])
        if params.get("charge") is not None:
            cli_args.extend(["--charge", str(params["charge"])])
        if params.get("multiplicity") is not None:
            cli_args.extend(["--multiplicity", str(params["multiplicity"])])
        if params.get("aimnet_python"):
            cli_args.extend(["--aimnet-python", str(params["aimnet_python"])])
        if params.get("pyscf_python"):
            cli_args.extend(["--pyscf-python", str(params["pyscf_python"])])
        if params.get("gas_phase_ftmw_lock"):
            cli_args.append("--gas-phase-ftmw-lock")

        rc = cli_main(cli_args)
        result = {
            "operation": "rxn_profile",
            "exit_code": rc,
            "output_directory": str(out_dir),
            "results_json": str(out_dir / "results.json"),
            "manifest_sha256": str(out_dir / "manifest_sha256.json"),
        }

    else:
        raise ValueError(f"Unhandled operation '{operation}'.")

    receipt = {
        "schema_version": "cochem.module-handoff-result/1",
        "module_id": "kinetic",
        "operation": operation,
        "registry_keys": sorted(registry.keys()) if isinstance(registry, dict) else [],
        "result": result,
    }
    receipt_path = out_dir / "kinetic_handoff_result.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
