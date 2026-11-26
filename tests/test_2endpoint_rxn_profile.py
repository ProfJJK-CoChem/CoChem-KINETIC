"""
CoChem-KINETIC 2-Endpoint Reaction-Profile & Auto-Intermediate Discovery Tests
------------------------------------------------------------------------------
Zero-mock unit and integration tests covering:
  1. 2-endpoint input resolution for both `reactant.xyz` + `product.xyz` and
     `R.xyz` + `P.xyz` (when `Int_tet.xyz` is absent), plus preservation of
     explicit 5-point (`R`, `Int_tet`, `P`) mode.
  2. CLI `validate-stoich`, `rxn-profile --stage validate`, `--dft-basis` alias,
     and `--pathway-mode` parser flags.
  3. Genuine analytical ASE potential-energy-surface (PES) tests (zero mocks):
     - Triple-well analytical PES -> `discover_midpoint_intermediate_aimnet2`
       discovers a distinct minimum (`RMSD > 0.25 A`, `fmax < 0.02 eV/A`, `N_imag == 0`).
     - Double-well analytical PES -> `discover_midpoint_intermediate_aimnet2`
       correctly finds NO intermediate (`discovered == False`), followed by real
       11-image CI-NEB -> Eckart P-RFO TS polish (`N_imag == 1`) -> bidirectional
       mass-weighted IRC (`connected_strict == True`) -> `assemble_and_export_profile`
       with `pathway_mode == "elementary_3pt"`.
  4. Honest TS convergence diagnostics (never fabricating `N_imag == 1` when
     Hessian yields `N_imag != 1`).
  5. Multi-frame XYZ export (`neb_path.xyz` / `irc_path.xyz`) and SHA-256 manifest
     verification including `ts_polished.xyz`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from cochem_kinetic.chem312_specs import (
    ELEMENTARY_STATIONARY_STATES,
    STATIONARY_STATES,
    read_xyz,
    write_multi_xyz,
    write_xyz,
)
from cochem_kinetic.cli import _build_ts_diagnostics, build_parser, main as cli_main
from cochem_kinetic.rxn_profile import (
    HARTREE_TO_KCAL,
    analyze_stationary_point_aimnet2,
    assemble_and_export_profile,
    detect_input_mode,
    discover_midpoint_intermediate_aimnet2,
    load_reaction_inputs,
    optimize_ts_prfo_aimnet2,
    relax_minimum_aimnet2,
    run_cineb_aimnet2,
    sha256_file,
    verify_ts_irc_aimnet2,
)


class AnalyticalBondPES(Calculator):
    """
    Real translationally & rotationally invariant analytical 2-atom PES for
    zero-mock testing of ASE relaxers, CI-NEB, Eckart P-RFO, Hessians, and IRC.

    Modes:
      - 'double_well': V(r) = k * ((r - 2.0)^2 - 0.64)^2
        Minima at r = 1.2 A and r = 2.8 A; single TS at r = 2.0 A.
      - 'triple_well': V(r) = k * (r - 1.2)^2 * (r - 2.4)^2 * (r - 3.6)^2
        Minima at r = 1.2 A (R), r = 2.4 A (Int_tet), r = 3.6 A (P);
        TS1 at r ~ 1.71 A and TS2 at r ~ 3.09 A.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, mode: str = "double_well", scale: float = 1.0) -> None:
        super().__init__()
        self.mode = mode
        self.scale = float(scale)

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties or ["energy", "forces"], system_changes)
        pos = np.asarray(self.atoms.get_positions(), dtype=np.float64)
        vec = pos[1] - pos[0]
        r = float(np.linalg.norm(vec))
        unit = vec / max(r, 1e-12)

        if self.mode == "double_well":
            u = r - 2.0
            a2 = 0.64  # minima at 2.0 +- 0.8 -> 1.2 and 2.8 A
            energy = self.scale * float((u * u - a2) ** 2)
            dv_dr = self.scale * 4.0 * u * (u * u - a2)
        elif self.mode == "triple_well":
            d1 = r - 1.2
            d2 = r - 2.4
            d3 = r - 3.6
            energy = self.scale * float((d1 * d2 * d3) ** 2)
            dv_dr = self.scale * 2.0 * (d1 * d2 * d3) * (d2 * d3 + d1 * d3 + d1 * d2)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        forces = np.zeros_like(pos, dtype=np.float64)
        forces[0] = dv_dr * unit
        forces[1] = -dv_dr * unit

        self.results["energy"] = energy
        self.results["forces"] = forces


def _stage_hcof_2endpoint(folder: Path, use_student_names: bool = True) -> None:
    """Writes the C1H1F2O- (Q=-1, M=1) identity acyl-substitution 2-endpoint XYZ pair."""
    symbols = ["C", "O", "H", "F", "F"]
    r_coords = np.array(
        [
            [0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.18500000, 0.00000000],
            [0.62179821, -0.66191422, -0.61176920],
            [-1.13000000, -0.73000000, 0.08000000],
            [2.40000000, -0.73000000, 1.25000000],
        ],
        dtype=np.float64,
    )
    p_coords = np.array(
        [
            [0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.18500000, 0.00000000],
            [-0.62179821, -0.66191422, -0.61176920],
            [-2.40000000, -0.73000000, 1.25000000],
            [1.13000000, -0.73000000, 0.08000000],
        ],
        dtype=np.float64,
    )
    r_name = "reactant.xyz" if use_student_names else "R.xyz"
    p_name = "product.xyz" if use_student_names else "P.xyz"
    write_xyz(folder / r_name, symbols, r_coords, "HCOF + F- reactant")
    write_xyz(folder / p_name, symbols, p_coords, "HCOF + F- product")


def test_2endpoint_input_resolution_aliases_and_mode_detection(tmp_path: Path) -> None:
    """Verify reactant.xyz+product.xyz, R.xyz+P.xyz, explicit 5pt, and missing-file diagnostics."""
    # Case A: student names reactant.xyz + product.xyz (no Int_tet.xyz)
    student_dir = tmp_path / "student_pair"
    _stage_hcof_2endpoint(student_dir, use_student_names=True)
    loaded_student = load_reaction_inputs(student_dir)
    assert set(loaded_student.keys()) == {"R", "P"}
    assert loaded_student["R"]["source_filename"] == "reactant.xyz"
    assert loaded_student["P"]["source_filename"] == "product.xyz"
    assert detect_input_mode(loaded_student) == "2pt_auto"

    # Case B: canonical names R.xyz + P.xyz (no Int_tet.xyz)
    rp_dir = tmp_path / "rp_pair"
    _stage_hcof_2endpoint(rp_dir, use_student_names=False)
    loaded_rp = load_reaction_inputs(rp_dir)
    assert set(loaded_rp.keys()) == {"R", "P"}
    assert loaded_rp["R"]["source_filename"] == "R.xyz"
    assert loaded_rp["P"]["source_filename"] == "P.xyz"
    assert detect_input_mode(loaded_rp) == "2pt_auto"

    # Case C: explicit Int_tet.xyz present -> 5pt_explicit preserved
    mid_coords = 0.5 * (loaded_rp["R"]["coords"] + loaded_rp["P"]["coords"])
    write_xyz(rp_dir / "Int_tet.xyz", loaded_rp["R"]["symbols"], mid_coords, "tetrahedral int")
    loaded_5pt = load_reaction_inputs(rp_dir)
    assert "Int_tet" in loaded_5pt
    assert detect_input_mode(loaded_5pt) == "5pt_explicit"

    # Case D: missing product raises FileNotFoundError naming aliases
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    write_xyz(incomplete_dir / "reactant.xyz", loaded_student["R"]["symbols"], loaded_student["R"]["coords"], "R only")
    with pytest.raises(FileNotFoundError, match="product.xyz"):
        load_reaction_inputs(incomplete_dir)


def test_cli_validate_and_parser_flags_for_2endpoint_folders(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI validate-stoich, rxn-profile --stage validate, --dft-basis, and --pathway-mode."""
    student_dir = tmp_path / "hcof_student"
    _stage_hcof_2endpoint(student_dir, use_student_names=True)

    # validate-stoich works directly on reactant.xyz + product.xyz without --states
    rc_val = cli_main(
        ["validate-stoich", "--input-dir", str(student_dir), "--charge", "-1", "--multiplicity", "1"]
    )
    assert rc_val == 0
    val_payload = json.loads(capsys.readouterr().out)
    assert val_payload["valid"] is True
    assert val_payload["formula"] == "CHF2O-"
    assert val_payload["states_checked"] == ["R", "P"]

    # rxn-profile --stage validate detects 2pt_auto on reactant.xyz + product.xyz
    out_dir = tmp_path / "val_out"
    rc_prof = cli_main(
        [
            "rxn-profile",
            "--input-dir", str(student_dir),
            "--output-dir", str(out_dir),
            "--charge", "-1",
            "--multiplicity", "1",
            "--dft-basis", "def2-SVP",
            "--stage", "validate",
        ]
    )
    assert rc_prof == 0
    prof_payload = json.loads(capsys.readouterr().out)
    assert prof_payload["valid"] is True
    assert prof_payload["input_mode"] == "2pt_auto"

    # Parser accepts --dft-basis alias and --pathway-mode
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "rxn-profile",
            "--input-dir", str(student_dir),
            "--output-dir", str(out_dir),
            "--dft-basis", "def2-SVP",
            "--pathway-mode", "3pt",
        ]
    )
    assert parsed.basis == "def2-SVP"
    assert parsed.pathway_mode == "3pt"


def test_midpoint_intermediate_discovery_triple_well_vs_double_well() -> None:
    """Zero-mock analytical PES check: triple-well discovers Int_tet, double-well does not."""
    symbols = ["C", "O"]

    # Triple-well: minima at 1.2, 2.4, 3.6 A -> midpoint (2.4 A) is a genuine local minimum (N_imag == 0)
    calc_triple = AnalyticalBondPES(mode="triple_well", scale=0.5)
    r_tri = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=np.float64)
    p_tri = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.6]], dtype=np.float64)
    disc_tri = discover_midpoint_intermediate_aimnet2(
        symbols=symbols,
        r_coords=r_tri,
        p_coords=p_tri,
        calc=calc_triple,
        rmsd_threshold_ang=0.25,
        fmax_threshold_ev_ang=0.02,
    )
    assert disc_tri["discovered"] is True
    assert disc_tri["coords"] is not None
    assert disc_tri["rmsd_from_r_ang"] > 0.25
    assert disc_tri["rmsd_from_p_ang"] > 0.25
    assert disc_tri["fmax_ev_ang"] < 0.02
    assert disc_tri["hessian"]["thermo"]["n_imag"] == 0
    int_dist = float(np.linalg.norm(disc_tri["coords"][1] - disc_tri["coords"][0]))
    assert abs(int_dist - 2.4) < 1e-3

    # Double-well: minima at 1.2 and 2.8 A; midpoint at 2.0 A is a saddle (N_imag == 1, NOT a minimum!)
    calc_double = AnalyticalBondPES(mode="double_well", scale=1.5)
    r_dbl = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=np.float64)
    p_dbl = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.8]], dtype=np.float64)
    disc_dbl = discover_midpoint_intermediate_aimnet2(
        symbols=symbols,
        r_coords=r_dbl,
        p_coords=p_dbl,
        calc=calc_double,
        rmsd_threshold_ang=0.25,
        fmax_threshold_ev_ang=0.02,
    )
    assert disc_dbl["discovered"] is False
    assert disc_dbl["coords"] is None
    assert len(disc_dbl["candidates"]) >= 1
    assert all(c["qualifies"] is False for c in disc_dbl["candidates"])


def test_elementary_3pt_end_to_end_analytical_pipeline_and_export(tmp_path: Path) -> None:
    """
    Zero-mock elementary 3-point (R -> TS -> P) integration test on a real analytical
    double-well PES: CI-NEB -> Eckart P-RFO -> Hessian (N_imag == 1) -> IRC -> Export.
    """
    symbols = ["C", "O"]
    calc = AnalyticalBondPES(mode="double_well", scale=1.5)

    r_init = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]], dtype=np.float64)
    p_init = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.85]], dtype=np.float64)

    r_opt, _, r_fmax = relax_minimum_aimnet2(symbols, r_init, calc=calc, fmax=0.002)
    p_opt, _, p_fmax = relax_minimum_aimnet2(symbols, p_init, calc=calc, fmax=0.002)
    assert r_fmax < 0.02 and p_fmax < 0.02

    neb = run_cineb_aimnet2(symbols, r_opt, p_opt, n_images=11, calc=calc, fmax=0.05, max_steps=120)
    assert neb["n_images"] == 11
    neb_path = write_multi_xyz(tmp_path / "neb_path.xyz", symbols, neb["all_coords"])
    assert neb_path.is_file()

    ts_opt = optimize_ts_prfo_aimnet2(symbols, neb["peak_coords"], calc=calc, fmax=0.002, max_steps=40)
    assert ts_opt["converged"] is True
    ts_dist = float(np.linalg.norm(ts_opt["coords"][1] - ts_opt["coords"][0]))
    assert abs(ts_dist - 2.0) < 1e-3

    hess_r = analyze_stationary_point_aimnet2(symbols, r_opt, calc=calc)
    hess_ts = analyze_stationary_point_aimnet2(symbols, ts_opt["coords"], calc=calc)
    hess_p = analyze_stationary_point_aimnet2(symbols, p_opt, calc=calc)
    assert hess_r["thermo"]["n_imag"] == 0
    assert hess_ts["thermo"]["n_imag"] == 1
    assert hess_p["thermo"]["n_imag"] == 0

    irc = verify_ts_irc_aimnet2(
        symbols=symbols,
        ts_coords=ts_opt["coords"],
        imag_mode_cart=np.asarray(hess_ts["imag_mode_cart"], dtype=np.float64),
        ref_a_coords=r_opt,
        ref_b_coords=p_opt,
        calc=calc,
        step_size_amu_ang=0.06,
        max_steps=120,
        rmsd_tol=0.15,
    )
    assert irc["connected_strict"] is True
    assert irc["max_com_drift_ang"] < 1e-8
    irc_path = write_multi_xyz(tmp_path / "irc_path.xyz", symbols, irc["path_coords"])
    assert irc_path.is_file()

    # Assemble 3-point profile using analytical energies converted to Hartree
    ev_to_ha = 1.0 / 27.211386245988
    states_data = {
        "R": {
            "symbols": symbols,
            "coords": r_opt,
            "hessian": hess_r,
            "dft": {"converged": True, "e_tot_ha": -113.0 + hess_r["aimnet2_energy_ev"] * ev_to_ha, "wall_seconds": 0.01, "engine": "cpu"},
        },
        "TS": {
            "symbols": symbols,
            "coords": ts_opt["coords"],
            "hessian": hess_ts,
            "dft": {"converged": True, "e_tot_ha": -113.0 + hess_ts["aimnet2_energy_ev"] * ev_to_ha, "wall_seconds": 0.01, "engine": "cpu"},
        },
        "P": {
            "symbols": symbols,
            "coords": p_opt,
            "hessian": hess_p,
            "dft": {"converged": True, "e_tot_ha": -113.0 + hess_p["aimnet2_energy_ev"] * ev_to_ha, "wall_seconds": 0.01, "engine": "cpu"},
        },
    }
    stoich_report = {"valid": True, "n_atoms": 2, "charge": 0, "multiplicity": 1, "formula": "CO"}
    ts_diag = _build_ts_diagnostics(("TS",), {"TS": ts_opt}, {"TS": hess_ts})
    assert ts_diag["TS"]["status"] == "CONVERGED_FIRST_ORDER_SADDLE"
    assert ts_diag["TS"]["is_first_order_saddle"] is True

    summary = assemble_and_export_profile(
        output_dir=tmp_path,
        case_name="analytical_double_well",
        reaction_equation="R -> TS -> P",
        states_data=states_data,
        stoich_report=stoich_report,
        irc_reports={"TS": irc},
        metadata={
            "pathway_mode": "elementary_3pt",
            "input_mode": "2pt_elementary",
            "stationary_points": list(ELEMENTARY_STATIONARY_STATES),
            "ts_diagnostics": ts_diag,
            "temp_k": 298.15,
            "basis": "def2-SVP",
            "xc": "wb97x",
            "disp": "d3zero",
            "solvation_model": "pcm",
            "solvent_name": "acetonitrile",
            "eps": 35.688,
        },
    )

    assert summary["pathway_mode"] == "elementary_3pt"
    assert summary["stationary_points"] == ["R", "TS", "P"]
    assert summary["ts_n_imaginary_modes"] == 1
    assert summary["ts_all_first_order"] is True

    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert results["pathway_mode"] == "elementary_3pt"
    assert results["stationary_points"] == ["R", "TS", "P"]
    assert results["ts_n_imaginary_modes"] == 1
    assert results["ts_all_first_order"] is True
    assert results["irc_verification"]["TS"]["connected_strict"] is True

    # Verify ts_polished.xyz, stationary_table.md, stationary_table.csv, and manifest_sha256.json
    assert (tmp_path / "ts_polished.xyz").is_file()
    ts_syms, ts_xyz, _ = read_xyz(tmp_path / "ts_polished.xyz")
    assert ts_syms == symbols
    assert np.allclose(ts_xyz, ts_opt["coords"], atol=1e-6)

    md_text = (tmp_path / "stationary_table.md").read_text(encoding="utf-8")
    assert "3-Stationary-Point Reaction Profile" in md_text
    assert "elementary_3pt" in md_text
    assert "DF-wB97X-D3(0)/def2-SVP" in md_text

    with open(tmp_path / "stationary_table.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["state"] for r in rows] == ["R", "TS", "P"]
    assert [int(r["n_imag"]) for r in rows] == [0, 1, 0]
    de_ts_expected = (float(rows[1]["e_dft_ha"]) - float(rows[0]["e_dft_ha"])) * HARTREE_TO_KCAL
    assert abs(de_ts_expected - float(rows[1]["delta_e_kcal"])) < 0.01

    manifest = json.loads((tmp_path / "manifest_sha256.json").read_text(encoding="utf-8"))
    for required_artifact in (
        "results.json",
        "stationary_table.md",
        "stationary_table.csv",
        "ts_polished.xyz",
        "neb_path.xyz",
        "irc_path.xyz",
        "geometries/R.xyz",
        "geometries/TS.xyz",
        "geometries/P.xyz",
    ):
        assert required_artifact in manifest["files_sha256"]
        assert sha256_file(tmp_path / required_artifact) == manifest["files_sha256"][required_artifact]


def test_honest_ts_failure_diagnostics_never_fabricate_n_imag() -> None:
    """If TS polish ends at a minimum (N_imag == 0), diagnostics must honestly report failure."""
    symbols = ["C", "O"]
    calc = AnalyticalBondPES(mode="double_well", scale=1.5)
    # Deliberately evaluate a minimum geometry as a failed TS guess
    r_min = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=np.float64)
    hess_min = analyze_stationary_point_aimnet2(symbols, r_min, calc=calc)
    assert hess_min["thermo"]["n_imag"] == 0

    diag = _build_ts_diagnostics(
        ("TS",),
        {"TS": {"converged": False, "fmax_ev_ang": 0.05, "method": "eckart_prfo"}},
        {"TS": hess_min},
    )
    assert diag["TS"]["status"] == "FAILED_SADDLE_CONVERGENCE"
    assert diag["TS"]["n_imag"] == 0
    assert diag["TS"]["is_first_order_saddle"] is False
    assert "N_imag=0" in diag["TS"]["diagnostic_message"]
