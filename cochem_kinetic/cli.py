"""
CoChem-KINETIC Command-Line Interface (`cochem-kinetic`)
--------------------------------------------------------
Registered via [project.scripts] in pyproject.toml.

Subcommands:
  - rxn-profile     : Run or stage a 5-stationary-point charge-aware reaction profile
                      (AIMNet2 CI-NEB + P-RFO/Sella TS + Hessian + IRC + PySCF DF-wB97X-D3/def2-TZVPP + PCM).
  - validate-stoich : Strictly validate atom, element, charge, spin, and clash balance across XYZs.
  - list-specs      : List built-in CHEM 312 Spring 2027 balanced reaction specifications.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cochem_kinetic.chem312_specs import (
    STATIONARY_STATES,
    get_builtin_spec,
    list_builtin_specs,
    read_xyz,
    resolve_solvent_eps,
    validate_stationary_stoichiometry,
    write_xyz,
)
from cochem_kinetic.rxn_profile import (
    analyze_stationary_point_aimnet2,
    assemble_and_export_profile,
    build_aimnet2_calculator,
    optimize_ts_aimnet2,
    relax_minimum_aimnet2,
    run_cineb_aimnet2,
    run_pyscf_single_point,
    verify_ts_irc_aimnet2,
)

logger = logging.getLogger("cochem_kinetic.cli")


def _load_input_geometries(input_dir: Path, required_states: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for st in STATIONARY_STATES:
        p = input_dir / f"{st}.xyz"
        if p.is_file():
            syms, coords, comment = read_xyz(p)
            loaded[st] = {"symbols": syms, "coords": coords, "comment": comment, "source": str(p)}
        elif st in required_states:
            raise FileNotFoundError(f"Required input XYZ '{p}' not found in {input_dir}.")
    return loaded


def _run_mlff_stage(args: argparse.Namespace, out_dir: Path, charge: int, mult: int, case_name: str, eq: str) -> None:
    input_dir = Path(args.input_dir)
    loaded = _load_input_geometries(input_dir, required_states=("R", "Int_tet", "P"))
    stoich = validate_stationary_stoichiometry(
        loaded,
        expected_charge=charge,
        expected_multiplicity=mult,
        expected_formula=args.expected_formula,
        require_same_atom_order=True,
    )

    calc = build_aimnet2_calculator(charge=charge, multiplicity=mult, model=args.aimnet_model)
    symbols = loaded["R"]["symbols"]
    work_xyz_dir = out_dir / "mlff_geometries"
    work_xyz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Relax minima R, Int_tet, P (unless --skip-min-opt)
    relaxed: dict[str, np.ndarray] = {}
    opt_logs: dict[str, Any] = {}
    for st in ("R", "Int_tet", "P"):
        if args.skip_min_opt:
            relaxed[st] = np.asarray(loaded[st]["coords"], dtype=np.float64)
            opt_logs[st] = {"skipped": True}
        else:
            pos, e_ev, fmax = relax_minimum_aimnet2(
                symbols, loaded[st]["coords"], charge=charge, multiplicity=mult, calc=calc, fmax=0.005, max_steps=400
            )
            relaxed[st] = pos
            opt_logs[st] = {"energy_ev": e_ev, "fmax_ev_ang": fmax}

    # 2. Obtain TS1 & TS2 guesses via 11-image CI-NEB (if not supplied or --force-neb) and refine
    neb_logs: dict[str, Any] = {}
    for ts_name, left_st, right_st in (("TS1", "R", "Int_tet"), ("TS2", "Int_tet", "P")):
        if (ts_name not in loaded) or args.force_neb:
            neb_res = run_cineb_aimnet2(
                symbols,
                relaxed[left_st],
                relaxed[right_st],
                charge=charge,
                multiplicity=mult,
                n_images=args.neb_images,
                calc=calc,
            )
            ts_guess = neb_res["peak_coords"]
            neb_logs[ts_name] = {
                "n_images": neb_res["n_images"],
                "peak_index": neb_res["peak_index"],
                "peak_energy_ev": neb_res["peak_energy_ev"],
                "energies_ev": neb_res["energies_ev"],
                "wall_seconds": neb_res["wall_seconds"],
            }
        else:
            ts_guess = np.asarray(loaded[ts_name]["coords"], dtype=np.float64)

        ts_opt = optimize_ts_aimnet2(
            symbols,
            ts_guess,
            charge=charge,
            multiplicity=mult,
            calc=calc,
            method=args.ts_method,
            fmax=0.002,
            max_steps=args.ts_max_steps,
        )
        relaxed[ts_name] = ts_opt["coords"]
        opt_logs[ts_name] = {k: v for k, v in ts_opt.items() if k != "coords"}

    # 3. Validate final stoichiometry across all 5 stationary states
    final_states_check = {st: {"symbols": symbols, "coords": relaxed[st]} for st in STATIONARY_STATES}
    stoich_final = validate_stationary_stoichiometry(
        final_states_check,
        expected_charge=charge,
        expected_multiplicity=mult,
        expected_formula=args.expected_formula,
        require_same_atom_order=True,
    )

    # 4. Hessians + RRHO/qRRHO thermo for all 5 states
    hessians: dict[str, Any] = {}
    for st in STATIONARY_STATES:
        write_xyz(work_xyz_dir / f"{st}.xyz", symbols, relaxed[st], f"{case_name} {st} AIMNet2")
        hessians[st] = analyze_stationary_point_aimnet2(
            symbols,
            relaxed[st],
            charge=charge,
            multiplicity=mult,
            temp_k=args.temp_k,
            calc=calc,
        )

    # 5. Bidirectional mass-weighted IRC verification for TS1 and TS2
    irc_reports: dict[str, Any] = {}
    for ts_name, left_st, right_st in (("TS1", "R", "Int_tet"), ("TS2", "Int_tet", "P")):
        imag_vec = hessians[ts_name].get("imag_mode_cart")
        if imag_vec is not None:
            irc = verify_ts_irc_aimnet2(
                symbols=symbols,
                ts_coords=relaxed[ts_name],
                imag_mode_cart=np.asarray(imag_vec, dtype=np.float64),
                ref_a_coords=relaxed[left_st],
                ref_b_coords=relaxed[right_st],
                charge=charge,
                multiplicity=mult,
                calc=calc,
                rmsd_tol=0.15,
            )
            irc_reports[ts_name] = {
                k: v for k, v in irc.items() if k not in {"path_coords", "endpoint_a_coords", "endpoint_b_coords"}
            }
            if "path_energies" in irc:
                irc_reports[ts_name]["path_energies"] = [float(x) for x in irc["path_energies"]]
        else:
            irc_reports[ts_name] = {"connected_strict": False, "error": f"No imaginary mode found for {ts_name}."}

    mlff_stage_payload = {
        "case_name": case_name,
        "reaction_equation": eq,
        "charge": charge,
        "multiplicity": mult,
        "temp_k": args.temp_k,
        "stoichiometry": stoich_final,
        "optimization": opt_logs,
        "neb": neb_logs,
        "hessians": hessians,
        "irc": irc_reports,
    }
    (out_dir / "mlff_stage.json").write_text(json.dumps(mlff_stage_payload, indent=2) + "\n", encoding="utf-8")


def _run_dft_stage(args: argparse.Namespace, out_dir: Path, charge: int, mult: int) -> None:
    geom_dir = out_dir / "mlff_geometries"
    if not geom_dir.is_dir():
        geom_dir = Path(args.input_dir)
    loaded = _load_input_geometries(geom_dir, required_states=STATIONARY_STATES)

    dft_results: dict[str, Any] = {}
    cpu_benchmarks: dict[str, Any] = {}

    for st in STATIONARY_STATES:
        syms = loaded[st]["symbols"]
        coords = loaded[st]["coords"]
        primary_engine = "gpu" if args.dft_engine in {"gpu", "both"} else "cpu"
        dft_results[st] = run_pyscf_single_point(
            symbols=syms,
            coords=coords,
            charge=charge,
            multiplicity=mult,
            basis=args.basis,
            xc=args.xc,
            disp=args.disp,
            solvent_name=args.solvent,
            eps=args.eps,
            solvation_model=args.solvation_model,
            engine=primary_engine,
            grid_level=args.grid_level,
            gas_phase_ftmw_lock=args.gas_phase_ftmw_lock,
        )
        if args.dft_engine == "both" and (args.benchmark_all_cpu or st == "R"):
            cpu_benchmarks[st] = run_pyscf_single_point(
                symbols=syms,
                coords=coords,
                charge=charge,
                multiplicity=mult,
                basis=args.basis,
                xc=args.xc,
                disp=args.disp,
                solvent_name=args.solvent,
                eps=args.eps,
                solvation_model=args.solvation_model,
                engine="cpu",
                grid_level=args.grid_level,
                gas_phase_ftmw_lock=args.gas_phase_ftmw_lock,
            )

    payload = {
        "dft_engine": args.dft_engine,
        "dft_results": dft_results,
        "cpu_benchmarks": cpu_benchmarks,
    }
    (out_dir / "dft_stage.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_assemble_stage(args: argparse.Namespace, out_dir: Path, charge: int, mult: int, case_name: str, eq: str) -> dict[str, Any]:
    mlff_data = json.loads((out_dir / "mlff_stage.json").read_text(encoding="utf-8"))
    dft_data = json.loads((out_dir / "dft_stage.json").read_text(encoding="utf-8"))
    geom_dir = out_dir / "mlff_geometries"
    if not geom_dir.is_dir():
        geom_dir = Path(args.input_dir)
    loaded = _load_input_geometries(geom_dir, required_states=STATIONARY_STATES)

    states_data: dict[str, Any] = {}
    for st in STATIONARY_STATES:
        states_data[st] = {
            "symbols": loaded[st]["symbols"],
            "coords": loaded[st]["coords"],
            "hessian": mlff_data["hessians"][st],
            "dft": dft_data["dft_results"][st],
            "dft_cpu_benchmark": dft_data.get("cpu_benchmarks", {}).get(st),
        }

    eff_solvent, eff_eps = resolve_solvent_eps(args.solvent, args.eps)
    metadata = {
        "case_name": case_name,
        "temp_k": args.temp_k,
        "basis": args.basis,
        "xc": args.xc,
        "disp": args.disp,
        "solvation_model": args.solvation_model,
        "solvent_name": eff_solvent,
        "eps": eff_eps,
        "dft_engine": args.dft_engine,
        "aimnet_model": args.aimnet_model,
    }
    return assemble_and_export_profile(
        output_dir=out_dir,
        case_name=case_name,
        reaction_equation=eq,
        states_data=states_data,
        stoich_report=mlff_data["stoichiometry"],
        irc_reports=mlff_data["irc"],
        metadata=metadata,
    )


def _cmd_list_specs(_args: argparse.Namespace) -> int:
    for name in list_builtin_specs():
        spec = get_builtin_spec(name)
        print(
            f"{name}: {spec['formula']} ({spec['n_atoms']} atoms, Q={spec['charge']}, M={spec['multiplicity']}) | "
            f"{spec['reaction_equation']}"
        )
    return 0


def _cmd_validate_stoich(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    spec = get_builtin_spec(args.spec) if args.spec else None
    charge = args.charge if args.charge is not None else (spec["charge"] if spec else 0)
    mult = args.multiplicity if args.multiplicity is not None else (spec["multiplicity"] if spec else 1)
    expected_formula = args.expected_formula or (spec["formula"] if spec else None)

    loaded = _load_input_geometries(input_dir, required_states=STATIONARY_STATES)
    report = validate_stationary_stoichiometry(
        loaded,
        expected_charge=charge,
        expected_multiplicity=mult,
        expected_formula=expected_formula,
        require_same_atom_order=not args.allow_reordered,
    )
    print(json.dumps(report, indent=2))
    return 0


def _cmd_rxn_profile(args: argparse.Namespace) -> int:
    spec = get_builtin_spec(args.spec) if args.spec else None
    charge = args.charge if args.charge is not None else (spec["charge"] if spec else 0)
    mult = args.multiplicity if args.multiplicity is not None else (spec["multiplicity"] if spec else 1)
    case_name = args.case_name or (spec["name"] if spec else Path(args.output_dir).name)
    eq = args.reaction_equation or (spec["reaction_equation"] if spec else "R -> TS1 -> Int_tet -> TS2 -> P")
    if args.expected_formula is None and spec is not None:
        args.expected_formula = spec["formula"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "validate":
        loaded = _load_input_geometries(Path(args.input_dir), required_states=("R", "Int_tet", "P"))
        report = validate_stationary_stoichiometry(
            loaded,
            expected_charge=charge,
            expected_multiplicity=mult,
            expected_formula=args.expected_formula,
            require_same_atom_order=True,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.stage == "mlff":
        _run_mlff_stage(args, out_dir, charge, mult, case_name, eq)
        return 0

    if args.stage == "dft":
        _run_dft_stage(args, out_dir, charge, mult)
        return 0

    if args.stage == "assemble":
        summary = _run_assemble_stage(args, out_dir, charge, mult, case_name, eq)
        print(json.dumps(summary, indent=2))
        return 0

    # stage == "all": allow separate Python silos if --aimnet-python or --pyscf-python supplied
    if args.aimnet_python or args.pyscf_python:
        aimnet_py = args.aimnet_python or sys.executable
        pyscf_py = args.pyscf_python or sys.executable
        base_flags = [
            "-m", "cochem_kinetic.cli", "rxn-profile",
            "--input-dir", str(args.input_dir),
            "--output-dir", str(out_dir),
            "--charge", str(charge),
            "--multiplicity", str(mult),
            "--solvent", str(args.solvent),
            "--temp-k", str(args.temp_k),
            "--basis", str(args.basis),
            "--xc", str(args.xc),
            "--disp", str(args.disp),
            "--solvation-model", str(args.solvation_model),
            "--dft-engine", str(args.dft_engine),
            "--ts-method", str(args.ts_method),
            "--ts-max-steps", str(args.ts_max_steps),
            "--neb-images", str(args.neb_images),
            "--aimnet-model", str(args.aimnet_model),
            "--case-name", case_name,
            "--reaction-equation", eq,
        ]
        if args.eps is not None:
            base_flags.extend(["--eps", str(args.eps)])
        if args.expected_formula:
            base_flags.extend(["--expected-formula", args.expected_formula])
        if args.skip_min_opt:
            base_flags.append("--skip-min-opt")
        if args.force_neb:
            base_flags.append("--force-neb")
        if args.benchmark_all_cpu:
            base_flags.append("--benchmark-all-cpu")
        if args.gas_phase_ftmw_lock:
            base_flags.append("--gas-phase-ftmw-lock")

        subprocess.run([aimnet_py, *base_flags, "--stage", "mlff"], check=True)
        subprocess.run([pyscf_py, *base_flags, "--stage", "dft"], check=True)
        summary = _run_assemble_stage(args, out_dir, charge, mult, case_name, eq)
        print(json.dumps(summary, indent=2))
        return 0

    _run_mlff_stage(args, out_dir, charge, mult, case_name, eq)
    _run_dft_stage(args, out_dir, charge, mult)
    summary = _run_assemble_stage(args, out_dir, charge, mult, case_name, eq)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cochem-kinetic",
        description="CoChem-KINETIC: Charge-Aware Reaction Profile, IRC & Thermochemistry CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-specs", help="List built-in CHEM 312 reaction specifications.")
    p_list.set_defaults(func=_cmd_list_specs)

    p_val = sub.add_parser("validate-stoich", help="Validate atom/charge/spin balance across 5 stationary XYZ files.")
    p_val.add_argument("--input-dir", required=True, help="Directory containing R.xyz, TS1.xyz, Int_tet.xyz, TS2.xyz, P.xyz")
    p_val.add_argument("--spec", choices=list_builtin_specs(), default=None)
    p_val.add_argument("--charge", type=int, default=None)
    p_val.add_argument("--multiplicity", type=int, default=None)
    p_val.add_argument("--expected-formula", type=str, default=None)
    p_val.add_argument("--allow-reordered", action="store_true")
    p_val.set_defaults(func=_cmd_validate_stoich)

    p_rxn = sub.add_parser("rxn-profile", help="Run 5-stationary-point AIMNet2 + PySCF/PCM reaction profile.")
    p_rxn.add_argument("--input-dir", required=True, help="Directory with R.xyz, Int_tet.xyz, P.xyz (and optional TS1/TS2.xyz)")
    p_rxn.add_argument("--output-dir", required=True, help="Output directory for XYZs, tables, logs, and SHA-256 manifest")
    p_rxn.add_argument("--spec", choices=list_builtin_specs(), default=None, help="Built-in CHEM 312 specification")
    p_rxn.add_argument("--case-name", type=str, default=None)
    p_rxn.add_argument("--reaction-equation", type=str, default=None)
    p_rxn.add_argument("--expected-formula", type=str, default=None)
    p_rxn.add_argument("--charge", type=int, default=None)
    p_rxn.add_argument("--multiplicity", type=int, default=None)
    p_rxn.add_argument("--solvent", type=str, default="acetonitrile")
    p_rxn.add_argument("--eps", type=float, default=None)
    p_rxn.add_argument("--temp-k", type=float, default=298.15)
    p_rxn.add_argument("--basis", type=str, default="def2-tzvpp")
    p_rxn.add_argument("--xc", type=str, default="wb97x")
    p_rxn.add_argument("--disp", type=str, default="d3zero")
    p_rxn.add_argument("--solvation-model", choices=["pcm", "ddcosmo", "gas"], default="pcm")
    p_rxn.add_argument("--dft-engine", choices=["cpu", "gpu", "both"], default="cpu")
    p_rxn.add_argument("--grid-level", type=int, default=3)
    p_rxn.add_argument("--aimnet-model", type=str, default="aimnet2")
    p_rxn.add_argument("--ts-method", choices=["auto", "prfo", "sella"], default="auto")
    p_rxn.add_argument("--ts-max-steps", type=int, default=60)
    p_rxn.add_argument("--neb-images", type=int, default=11)
    p_rxn.add_argument("--skip-min-opt", action="store_true")
    p_rxn.add_argument("--force-neb", action="store_true")
    p_rxn.add_argument("--benchmark-all-cpu", action="store_true")
    p_rxn.add_argument("--gas-phase-ftmw-lock", action="store_true")
    p_rxn.add_argument("--stage", choices=["all", "validate", "mlff", "dft", "assemble"], default="all")
    p_rxn.add_argument("--aimnet-python", type=str, default=None, help="Optional isolated Python executable for AIMNet2 stage")
    p_rxn.add_argument("--pyscf-python", type=str, default=None, help="Optional isolated Python executable for PySCF stage")
    p_rxn.set_defaults(func=_cmd_rxn_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
