"""
CoChem-KINETIC Command-Line Interface (`cochem-kinetic`)
--------------------------------------------------------
Registered via [project.scripts] in pyproject.toml.

Subcommands:
  - rxn-profile     : Run or stage a 5-stationary-point charge-aware reaction profile
                      (AIMNet2 CI-NEB + P-RFO/Sella TS + Hessian + IRC + PySCF DF-wB97X-D3(0)/def2-TZVPP + PCM).
  - validate-stoich : Strictly validate atom, element, charge, spin, and clash balance across XYZs.
  - list-specs      : List built-in CHEM 312 Spring 2027 balanced reaction specifications.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cochem_kinetic.chem312_specs import (
    ELEMENTARY_STATIONARY_STATES,
    STATIONARY_STATES,
    get_builtin_spec,
    list_builtin_specs,
    read_xyz,
    resolve_solvent_eps,
    validate_stationary_stoichiometry,
    write_multi_xyz,
    write_xyz,
)
from cochem_kinetic.rxn_profile import (
    ALL_CANDIDATE_STATES,
    analyze_stationary_point_aimnet2,
    assemble_and_export_profile,
    build_aimnet2_calculator,
    detect_input_mode,
    discover_midpoint_intermediate_aimnet2,
    load_reaction_inputs,
    optimize_ts_aimnet2,
    relax_minimum_aimnet2,
    resolve_state_xyz_path,
    run_cineb_aimnet2,
    run_pyscf_single_point,
    verify_ts_irc_aimnet2,
)

logger = logging.getLogger("cochem_kinetic.cli")


def _positive_float(value: str) -> float:
    """Argparse type enforcing a finite strictly positive floating-point value (> 0)."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive float, got {value!r}.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"Value must be finite and > 0, got {value!r}.")
    return parsed


def _validated_pressure_atm(args: argparse.Namespace, fallback: float = 1.0) -> float:
    pressure_atm = float(getattr(args, "pressure_atm", fallback))
    if not math.isfinite(pressure_atm) or pressure_atm <= 0.0:
        raise ValueError(f"--pressure-atm must be a finite value > 0, got {pressure_atm!r}.")
    return pressure_atm


def _load_input_geometries(
    input_dir: Path,
    required_states: tuple[str, ...],
    candidate_states: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    return load_reaction_inputs(
        input_dir,
        required_states=required_states,
        candidate_states=candidate_states if candidate_states is not None else STATIONARY_STATES,
    )


def _build_ts_diagnostics(
    ts_states: tuple[str, ...],
    opt_logs: dict[str, Any],
    hessians: dict[str, Any],
) -> dict[str, Any]:
    """Builds honest TS convergence diagnostics without fabricating N_imag == 1."""
    diag: dict[str, Any] = {}
    for ts_name in ts_states:
        opt_info = opt_logs.get(ts_name, {})
        thermo = hessians.get(ts_name, {}).get("thermo", {})
        n_imag = int(thermo.get("n_imag", -1)) if thermo else -1
        imag_freqs = thermo.get("imag_freqs_cm1", [])
        fmax_val = float(opt_info.get("fmax_ev_ang", hessians.get(ts_name, {}).get("fmax_ev_ang", float("nan"))))
        is_first_order = n_imag == 1
        grad_ok = bool(opt_info.get("converged", False))
        status = "CONVERGED_FIRST_ORDER_SADDLE" if (grad_ok and is_first_order) else "FAILED_SADDLE_CONVERGENCE"
        entry: dict[str, Any] = {
            "status": status,
            "method": opt_info.get("method", "unknown"),
            "gradient_converged": grad_ok,
            "fmax_ev_ang": fmax_val,
            "n_imag": n_imag,
            "imag_freqs_cm1": imag_freqs,
            "is_first_order_saddle": is_first_order,
        }
        if not (grad_ok and is_first_order):
            reasons: list[str] = []
            if not grad_ok:
                reasons.append(f"TS optimizer did not meet gradient fmax tolerance (fmax={fmax_val:.5f} eV/A)")
            if not is_first_order:
                reasons.append(f"Eckart-projected Hessian found N_imag={n_imag} (expected 1; imag={imag_freqs})")
            entry["diagnostic_message"] = "; ".join(reasons)
        diag[ts_name] = entry
    return diag


def _run_mlff_stage(
    args: argparse.Namespace,
    out_dir: Path,
    charge: int,
    mult: int,
    case_name: str,
    eq: str | None,
) -> None:
    pressure_atm = _validated_pressure_atm(args)
    input_dir = Path(args.input_dir)
    loaded = load_reaction_inputs(
        input_dir,
        required_states=("R", "P"),
        candidate_states=ALL_CANDIDATE_STATES,
    )
    validate_stationary_stoichiometry(
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

    detected_mode = detect_input_mode(loaded)
    requested_pathway = getattr(args, "pathway_mode", "auto")

    relaxed: dict[str, np.ndarray] = {}
    opt_logs: dict[str, Any] = {}

    # 1. Relax endpoints R and P (and explicit Int_tet if present and not forced to 3pt)
    endpoint_minima = ["R", "P"]
    if "Int_tet" in loaded and requested_pathway != "3pt":
        endpoint_minima.insert(1, "Int_tet")

    for st in endpoint_minima:
        if args.skip_min_opt:
            relaxed[st] = np.asarray(loaded[st]["coords"], dtype=np.float64)
            opt_logs[st] = {"skipped": True, "source_filename": loaded[st].get("source_filename")}
        else:
            pos, e_ev, fmax = relax_minimum_aimnet2(
                symbols, loaded[st]["coords"], charge=charge, multiplicity=mult, calc=calc, fmax=0.005, max_steps=400
            )
            relaxed[st] = pos
            opt_logs[st] = {
                "energy_ev": e_ev,
                "fmax_ev_ang": fmax,
                "converged": bool(fmax <= 0.02),
                "source_filename": loaded[st].get("source_filename"),
            }

    # 2. Pathway determination: explicit 5pt, 2pt midpoint intermediate discovery, or elementary 3pt
    intermediate_discovery: dict[str, Any] | None = None
    if requested_pathway == "3pt":
        pathway_mode = "elementary_3pt"
        input_mode = "2pt_elementary_forced" if detected_mode == "2pt_auto" else "3pt_forced"
        active_states: tuple[str, ...] = ELEMENTARY_STATIONARY_STATES
        ts_pairs: tuple[tuple[str, str, str], ...] = (("TS", "R", "P"),)
    elif "Int_tet" in relaxed:
        pathway_mode = "stepwise_5pt"
        input_mode = "5pt_explicit"
        active_states = STATIONARY_STATES
        ts_pairs = (("TS1", "R", "Int_tet"), ("TS2", "Int_tet", "P"))
    else:
        # 2-endpoint folder without Int_tet.xyz: test midpoint relaxation for a true local minimum
        disc = discover_midpoint_intermediate_aimnet2(
            symbols=symbols,
            r_coords=relaxed["R"],
            p_coords=relaxed["P"],
            charge=charge,
            multiplicity=mult,
            temp_k=args.temp_k,
            pressure_atm=pressure_atm,
            calc=calc,
            rmsd_threshold_ang=0.25,
            fmax_threshold_ev_ang=0.02,
        )
        intermediate_discovery = {
            k: v for k, v in disc.items() if k not in {"coords", "hessian"}
        }
        if disc["discovered"] and disc["coords"] is not None:
            pathway_mode = "stepwise_5pt"
            input_mode = "2pt_auto_discovered_intermediate"
            active_states = STATIONARY_STATES
            ts_pairs = (("TS1", "R", "Int_tet"), ("TS2", "Int_tet", "P"))
            relaxed["Int_tet"] = np.asarray(disc["coords"], dtype=np.float64)
            opt_logs["Int_tet"] = {
                "energy_ev": disc["energy_ev"],
                "fmax_ev_ang": disc["fmax_ev_ang"],
                "converged": True,
                "auto_discovered": True,
                "source_candidate": disc["source_candidate"],
                "rmsd_from_r_ang": disc["rmsd_from_r_ang"],
                "rmsd_from_p_ang": disc["rmsd_from_p_ang"],
            }
            logger.info(
                "2-endpoint auto-discovery found distinct intermediate minimum Int_tet "
                "(source=%s, RMSD_R=%.3f A, RMSD_P=%.3f A, N_imag=0) -> running 5-point pathway.",
                disc["source_candidate"],
                disc["rmsd_from_r_ang"],
                disc["rmsd_from_p_ang"],
            )
        elif requested_pathway == "5pt":
            raise RuntimeError(
                "pathway_mode='5pt' requested, but Int_tet.xyz is absent and midpoint relaxation "
                "did not discover a distinct local minimum (RMSD > 0.25 A, fmax < 0.02 eV/A, N_imag == 0)."
            )
        else:
            pathway_mode = "elementary_3pt"
            input_mode = "2pt_elementary"
            active_states = ELEMENTARY_STATIONARY_STATES
            ts_pairs = (("TS", "R", "P"),)
            logger.info(
                "2-endpoint auto-discovery found no distinct intermediate minimum -> running elementary 3-point pathway (R -> TS -> P)."
            )

    eff_eq = eq if eq else ("R -> TS -> P" if pathway_mode == "elementary_3pt" else "R -> TS1 -> Int_tet -> TS2 -> P")

    # 3. Obtain TS guess(es) via 11-image CI-NEB (if not supplied or --force-neb) and refine
    neb_logs: dict[str, Any] = {}
    neb_frames_combined: list[np.ndarray] = []
    neb_comments_combined: list[str] = []
    for ts_name, left_st, right_st in ts_pairs:
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
            seg_comments = [
                f"{case_name} | NEB {left_st}->{right_st} ({ts_name}) image {idx}/{neb_res['n_images'] - 1} | E_eV={e_val:.8f}"
                for idx, e_val in enumerate(neb_res["energies_ev"])
            ]
            if len(ts_pairs) > 1:
                write_multi_xyz(
                    out_dir / f"neb_{ts_name.lower()}_path.xyz",
                    symbols,
                    neb_res["all_coords"],
                    seg_comments,
                )
            neb_frames_combined.extend(neb_res["all_coords"])
            neb_comments_combined.extend(seg_comments)
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

    if neb_frames_combined:
        write_multi_xyz(out_dir / "neb_path.xyz", symbols, neb_frames_combined, neb_comments_combined)

    # 4. Validate final stoichiometry across all active stationary states
    final_states_check = {st: {"symbols": symbols, "coords": relaxed[st]} for st in active_states}
    stoich_final = validate_stationary_stoichiometry(
        final_states_check,
        expected_charge=charge,
        expected_multiplicity=mult,
        expected_formula=args.expected_formula,
        require_same_atom_order=True,
    )

    # 5. Hessians + RRHO/qRRHO thermo for all active states
    hessians: dict[str, Any] = {}
    for st in active_states:
        write_xyz(work_xyz_dir / f"{st}.xyz", symbols, relaxed[st], f"{case_name} {st} AIMNet2")
        hessians[st] = analyze_stationary_point_aimnet2(
            symbols,
            relaxed[st],
            charge=charge,
            multiplicity=mult,
            temp_k=args.temp_k,
            pressure_atm=pressure_atm,
            calc=calc,
        )

    ts_names = tuple(ts_name for ts_name, _, _ in ts_pairs)
    ts_diagnostics = _build_ts_diagnostics(ts_names, opt_logs, hessians)

    # Write top-level ts_polished.xyz for the primary/highest-energy TS right after MLFF stage
    primary_ts = "TS" if "TS" in relaxed else max(ts_names, key=lambda s: float(hessians[s]["aimnet2_energy_ev"]))
    p_thermo = hessians[primary_ts]["thermo"]
    p_imag = p_thermo.get("imag_freqs_cm1", [])
    p_imag_str = f"{p_imag[0]:.2f} cm^-1" if p_imag else "none"
    write_xyz(
        out_dir / "ts_polished.xyz",
        symbols,
        relaxed[primary_ts],
        f"{case_name} | {primary_ts} AIMNet2 ({pathway_mode}) | N_imag={p_thermo['n_imag']} ({p_imag_str})",
    )

    # 6. Bidirectional mass-weighted IRC verification for every TS
    irc_reports: dict[str, Any] = {}
    irc_frames_combined: list[np.ndarray] = []
    irc_comments_combined: list[str] = []
    for ts_name, left_st, right_st in ts_pairs:
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
            if "path_coords" in irc and len(irc["path_coords"]) > 0:
                irc_energies = irc.get("path_energies", [float("nan")] * len(irc["path_coords"]))
                seg_comments = [
                    f"{case_name} | IRC {ts_name} ({left_st}<->{right_st}) step {idx}/{len(irc['path_coords']) - 1} | E_eV={float(e_val):.8f}"
                    for idx, e_val in enumerate(irc_energies)
                ]
                if len(ts_pairs) > 1:
                    write_multi_xyz(
                        out_dir / f"irc_{ts_name.lower()}_path.xyz",
                        symbols,
                        irc["path_coords"],
                        seg_comments,
                    )
                irc_frames_combined.extend(list(irc["path_coords"]))
                irc_comments_combined.extend(seg_comments)
        else:
            irc_reports[ts_name] = {"connected_strict": False, "error": f"No imaginary mode found for {ts_name}."}

    if irc_frames_combined:
        write_multi_xyz(out_dir / "irc_path.xyz", symbols, irc_frames_combined, irc_comments_combined)

    mlff_stage_payload: dict[str, Any] = {
        "case_name": case_name,
        "reaction_equation": eff_eq,
        "pathway_mode": pathway_mode,
        "input_mode": input_mode,
        "stationary_points": list(active_states),
        "charge": charge,
        "multiplicity": mult,
        "temp_k": args.temp_k,
        "pressure_atm": pressure_atm,
        "stoichiometry": stoich_final,
        "intermediate_discovery": intermediate_discovery,
        "optimization": opt_logs,
        "ts_diagnostics": ts_diagnostics,
        "neb": neb_logs,
        "hessians": hessians,
        "irc": irc_reports,
    }
    (out_dir / "mlff_stage.json").write_text(json.dumps(mlff_stage_payload, indent=2) + "\n", encoding="utf-8")


def _resolve_active_states(out_dir: Path, geom_dir: Path) -> tuple[str, ...]:
    mlff_json = out_dir / "mlff_stage.json"
    if mlff_json.is_file():
        mlff_data = json.loads(mlff_json.read_text(encoding="utf-8"))
        pts = mlff_data.get("stationary_points")
        if pts:
            return tuple(str(s) for s in pts)
    if (geom_dir / "TS.xyz").is_file() and not (geom_dir / "Int_tet.xyz").is_file():
        return ELEMENTARY_STATIONARY_STATES
    return STATIONARY_STATES


def _run_dft_stage(args: argparse.Namespace, out_dir: Path, charge: int, mult: int) -> None:
    geom_dir = out_dir / "mlff_geometries"
    if not geom_dir.is_dir():
        geom_dir = Path(args.input_dir)
    active_states = _resolve_active_states(out_dir, geom_dir)
    loaded = load_reaction_inputs(geom_dir, required_states=active_states, candidate_states=active_states)

    dft_results: dict[str, Any] = {}
    cpu_benchmarks: dict[str, Any] = {}

    for st in active_states:
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
        "stationary_points": list(active_states),
        "dft_results": dft_results,
        "cpu_benchmarks": cpu_benchmarks,
    }
    (out_dir / "dft_stage.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_assemble_stage(
    args: argparse.Namespace,
    out_dir: Path,
    charge: int,
    mult: int,
    case_name: str,
    eq: str | None,
) -> dict[str, Any]:
    mlff_data = json.loads((out_dir / "mlff_stage.json").read_text(encoding="utf-8"))
    dft_data = json.loads((out_dir / "dft_stage.json").read_text(encoding="utf-8"))
    geom_dir = out_dir / "mlff_geometries"
    if not geom_dir.is_dir():
        geom_dir = Path(args.input_dir)
    active_states = tuple(str(s) for s in mlff_data.get("stationary_points", STATIONARY_STATES))
    loaded = load_reaction_inputs(geom_dir, required_states=active_states, candidate_states=active_states)

    states_data: dict[str, Any] = {}
    for st in active_states:
        states_data[st] = {
            "symbols": loaded[st]["symbols"],
            "coords": loaded[st]["coords"],
            "hessian": mlff_data["hessians"][st],
            "dft": dft_data["dft_results"][st],
            "dft_cpu_benchmark": dft_data.get("cpu_benchmarks", {}).get(st),
        }

    eff_solvent, eff_eps = resolve_solvent_eps(args.solvent, args.eps)
    eff_eq = eq or mlff_data.get("reaction_equation") or (
        "R -> TS -> P" if active_states == ELEMENTARY_STATIONARY_STATES else "R -> TS1 -> Int_tet -> TS2 -> P"
    )
    pressure_atm = _validated_pressure_atm(args, fallback=float(mlff_data.get("pressure_atm", 1.0)))
    metadata = {
        "case_name": case_name,
        "pathway_mode": mlff_data.get(
            "pathway_mode",
            "elementary_3pt" if active_states == ELEMENTARY_STATIONARY_STATES else "stepwise_5pt",
        ),
        "input_mode": mlff_data.get("input_mode", "5pt_explicit"),
        "stationary_points": list(active_states),
        "intermediate_discovery": mlff_data.get("intermediate_discovery"),
        "ts_diagnostics": mlff_data.get("ts_diagnostics"),
        "temp_k": args.temp_k,
        "pressure_atm": pressure_atm,
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
        reaction_equation=eff_eq,
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

    if args.states:
        required_states = tuple(args.states)
        candidate_states = required_states
    elif spec is not None and "required_states" in spec:
        required_states = tuple(spec["required_states"])
        candidate_states = tuple(spec.get("stationary_states", required_states))
    elif resolve_state_xyz_path(input_dir, "Int_tet") is None and resolve_state_xyz_path(input_dir, "R") is not None and resolve_state_xyz_path(input_dir, "P") is not None:
        required_states = ("R", "P")
        candidate_states = ALL_CANDIDATE_STATES
    else:
        required_states = STATIONARY_STATES
        candidate_states = STATIONARY_STATES

    loaded = _load_input_geometries(
        input_dir,
        required_states=required_states,
        candidate_states=candidate_states,
    )
    report = validate_stationary_stoichiometry(
        loaded,
        expected_charge=charge,
        expected_multiplicity=mult,
        expected_formula=expected_formula,
        require_same_atom_order=not args.allow_reordered,
    )
    print(json.dumps(report, indent=2))
    return 0


def _default_aimnet_python() -> str | None:
    """Return non-empty $COCHEM_AIMNET_PYTHON from the environment, or None."""
    raw = os.environ.get("COCHEM_AIMNET_PYTHON", "").strip()
    return raw or None


def _require_executable_python(executable: str, label: str) -> str:
    """Fail closed if a configured isolated Python interpreter does not exist."""
    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise FileNotFoundError(
            f"{label} Python executable not found: {executable!r}. "
            "Pass a valid interpreter path or update COCHEM_AIMNET_PYTHON."
        )
    return executable


def _cmd_rxn_profile(args: argparse.Namespace) -> int:
    pressure_atm = _validated_pressure_atm(args)
    spec = get_builtin_spec(args.spec) if args.spec else None
    charge = args.charge if args.charge is not None else (spec["charge"] if spec else 0)
    mult = args.multiplicity if args.multiplicity is not None else (spec["multiplicity"] if spec else 1)
    case_name = args.case_name or (spec["name"] if spec else Path(args.output_dir).name)
    eq: str | None = args.reaction_equation or (spec["reaction_equation"] if spec else None)
    if args.expected_formula is None and spec is not None:
        args.expected_formula = spec["formula"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "validate":
        loaded = load_reaction_inputs(
            Path(args.input_dir),
            required_states=("R", "P"),
            candidate_states=ALL_CANDIDATE_STATES,
        )
        report = validate_stationary_stoichiometry(
            loaded,
            expected_charge=charge,
            expected_multiplicity=mult,
            expected_formula=args.expected_formula,
            require_same_atom_order=True,
        )
        report["input_mode"] = detect_input_mode(loaded)
        report["temp_k"] = float(args.temp_k)
        report["pressure_atm"] = pressure_atm
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

    # stage == "all": allow separate Python silos if --aimnet-python, $COCHEM_AIMNET_PYTHON, or --pyscf-python supplied
    resolved_aimnet_py = (args.aimnet_python or "").strip() or _default_aimnet_python()
    resolved_pyscf_py = (args.pyscf_python or "").strip() or None
    if resolved_aimnet_py or resolved_pyscf_py:
        aimnet_py = (
            _require_executable_python(resolved_aimnet_py, "AIMNet2")
            if resolved_aimnet_py
            else sys.executable
        )
        pyscf_py = (
            _require_executable_python(resolved_pyscf_py, "PySCF")
            if resolved_pyscf_py
            else sys.executable
        )
        base_flags = [
            "-m", "cochem_kinetic.cli", "rxn-profile",
            "--input-dir", str(args.input_dir),
            "--output-dir", str(out_dir),
            "--charge", str(charge),
            "--multiplicity", str(mult),
            "--solvent", str(args.solvent),
            "--temp-k", str(args.temp_k),
            "--pressure-atm", str(pressure_atm),
            "--basis", str(args.basis),
            "--xc", str(args.xc),
            "--disp", str(args.disp),
            "--solvation-model", str(args.solvation_model),
            "--dft-engine", str(args.dft_engine),
            "--grid-level", str(args.grid_level),
            "--ts-method", str(args.ts_method),
            "--ts-max-steps", str(args.ts_max_steps),
            "--neb-images", str(args.neb_images),
            "--aimnet-model", str(args.aimnet_model),
            "--pathway-mode", str(args.pathway_mode),
            "--case-name", case_name,
        ]
        if eq is not None:
            base_flags.extend(["--reaction-equation", eq])
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

    p_val = sub.add_parser("validate-stoich", help="Validate atom/charge/spin balance across stationary or monomer XYZ files.")
    p_val.add_argument("--input-dir", required=True, help="Directory containing XYZ files (e.g. R.xyz/P.xyz, 5-state set, or monomer.xyz)")
    p_val.add_argument("--spec", choices=list_builtin_specs(), default=None)
    p_val.add_argument("--charge", type=int, default=None)
    p_val.add_argument("--multiplicity", type=int, default=None)
    p_val.add_argument("--expected-formula", type=str, default=None)
    p_val.add_argument("--states", nargs="+", default=None, help="Explicit state basenames to validate (e.g. --states R P or --states monomer)")
    p_val.add_argument("--allow-reordered", action="store_true")
    p_val.set_defaults(func=_cmd_validate_stoich)

    p_rxn = sub.add_parser(
        "rxn-profile",
        help="Run AIMNet2 + PySCF/PCM reaction profile (5-point stepwise or 2-endpoint auto 3pt/5pt).",
    )
    p_rxn.add_argument(
        "--input-dir",
        required=True,
        help="Directory with R.xyz + P.xyz (or reactant.xyz + product.xyz), plus optional Int_tet.xyz / TS*.xyz",
    )
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
    p_rxn.add_argument(
        "--pressure-atm",
        type=_positive_float,
        default=1.0,
        help="Ideal-gas reference pressure in atm for RRHO/qRRHO thermochemistry (must be > 0; default: 1.0).",
    )
    p_rxn.add_argument("--basis", "--dft-basis", dest="basis", type=str, default="def2-tzvpp")
    p_rxn.add_argument("--xc", type=str, default="wb97x")
    p_rxn.add_argument("--disp", type=str, default="d3zero")
    p_rxn.add_argument("--solvation-model", choices=["pcm", "ddcosmo", "gas"], default="pcm")
    p_rxn.add_argument("--dft-engine", choices=["cpu", "gpu", "both"], default="cpu")
    p_rxn.add_argument("--grid-level", type=int, default=3)
    p_rxn.add_argument("--aimnet-model", type=str, default="aimnet2")
    p_rxn.add_argument("--pathway-mode", choices=["auto", "5pt", "3pt"], default="auto")
    p_rxn.add_argument("--ts-method", choices=["auto", "prfo", "sella"], default="auto")
    p_rxn.add_argument("--ts-max-steps", type=int, default=60)
    p_rxn.add_argument("--neb-images", type=int, default=11)
    p_rxn.add_argument("--skip-min-opt", action="store_true")
    p_rxn.add_argument("--force-neb", action="store_true")
    p_rxn.add_argument("--benchmark-all-cpu", action="store_true")
    p_rxn.add_argument("--gas-phase-ftmw-lock", action="store_true")
    p_rxn.add_argument("--stage", choices=["all", "validate", "mlff", "dft", "assemble"], default="all")
    p_rxn.add_argument(
        "--aimnet-python",
        type=str,
        default=_default_aimnet_python(),
        help="Optional isolated Python executable for AIMNet2 stage (defaults to $COCHEM_AIMNET_PYTHON when set)",
    )
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
