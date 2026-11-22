"""
CoChem-KINETIC T-KIN-01 & CHEM 410/411 Verification Test Suite
--------------------------------------------------------------
Zero-mock tests verifying:
  1. IRC M^{-1/2} mass-weighting COM invariance (< 1e-8 A drift on asymmetric m=[12,1] diatomic)
     and endpoint relaxation / Kabsch RMSD / bond-graph connectivity.
  2. Phase-aware solvation API (gas / pcm / ddcosmo + ERR_DOMAIN_VIOLATION when gas_phase_ftmw_lock=True)
     and arbitrary-T R*T/P ideal-gas standard-state molar volume in Eyring TST.
  3. Explicit RuntimeError in MACEPreOptimizer when no valid MLFF backend is available (zero EMT fallback).
  4. Exact Beyer-Swinehart microcanonical RRKM k(E) = sigma * N^ddagger(E - E_0) / (h * rho(E)).
  5. CHEM 312 balanced reaction specs & strict stoichiometry validator (Team 1 Q=0 22 atoms,
     Team 4 cyclization Q=-1 15 atoms, and Team 4 intermolecular step Q=-1 21 atoms).
  6. cochem.modules entry point (kinetic_provider) and CLI (cochem-kinetic) real execution.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from cochem_kinetic.chem312_specs import (
    get_builtin_spec,
    list_builtin_specs,
    validate_stationary_stoichiometry,
    write_xyz,
)
from cochem_kinetic.cli import main as cli_main
from cochem_kinetic.engine.rrkm_theory import H_CM1_S, RRKMSolver
from cochem_kinetic.provider import execute_handoff, kinetic_provider
from kinetic_core.irc_tracer import IRCTracerEngine, compute_center_of_mass
from kinetic_core.mace_pre_opt import MACEPreOptimizer
from kinetic_core.thermo import apply_implicit_solvation_correction, calculate_eyring_rate


def _double_well_diatomic(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Translationally invariant double-well potential V(r) = ((r - 1.5)^2 - 0.25)^2."""
    c = np.asarray(coords, dtype=np.float64)
    vec = c[1] - c[0]
    r = float(np.linalg.norm(vec))
    u = r - 1.5
    val = float((u * u - 0.25) ** 2)
    dv_dr = 4.0 * u * (u * u - 0.25)
    grad = np.zeros_like(c)
    if r > 1e-12:
        unit = vec / r
        grad[1] = dv_dr * unit
        grad[0] = -dv_dr * unit
    return val, grad


def _relax_double_well_endpoint(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Steepest-descent relaxation preserving COM on the double-well diatomic."""
    c = np.asarray(coords, dtype=np.float64).copy()
    masses = np.array([12.0, 1.0], dtype=np.float64)
    inv_m = (1.0 / masses)[:, np.newaxis]
    for _ in range(300):
        e, g = _double_well_diatomic(c)
        if float(np.linalg.norm(g)) < 1e-8:
            break
        c = c - 0.1 * g * inv_m
    e, _ = _double_well_diatomic(c)
    return c, e


def test_irc_mass_weighting_com_invariance_asymmetric_diatomic() -> None:
    """Verify max |COM drift| < 1e-8 A on asymmetric m=[12, 1] diatomic and endpoint verification."""
    ts_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=np.float64)
    masses = np.array([12.0, 1.0], dtype=np.float64)
    # Deliberately supply an unprojected Cartesian stretch vector to verify COM projection too
    imag_mode = np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.9]], dtype=np.float64)

    ref_a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ref_b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)

    tracer = IRCTracerEngine(step_size_amu_ang=0.05, grad_tol=1e-5)
    report = tracer.verify_endpoints(
        symbols=["C", "H"],
        ts_coords=ts_coords,
        masses=masses,
        imag_mode_vector=imag_mode,
        grad_fn=_double_well_diatomic,
        ref_a_coords=ref_a,
        ref_b_coords=ref_b,
        relax_fn=_relax_double_well_endpoint,
        max_steps=80,
        rmsd_tol=0.05,
    )

    assert report["max_com_drift_ang"] < 1e-8, f"COM drift too large: {report['max_com_drift_ang']}"
    ts_com = compute_center_of_mass(ts_coords, masses)
    for frame in report["path_coords"]:
        drift = float(np.linalg.norm(compute_center_of_mass(frame, masses) - ts_com))
        assert drift < 1e-8

    assert report["rmsd_a_ang"] < 1e-4
    assert report["rmsd_b_ang"] < 1e-4
    assert report["bond_graph_match_a"] is True
    assert report["bond_graph_match_b"] is True
    assert report["connected_strict"] is True


def test_thermo_phase_aware_solvation_and_arbitrary_t_standard_state() -> None:
    """Verify phase-aware solvation + ERR_DOMAIN_VIOLATION guard and V_m = R*T/P scaling."""
    # Gas phase always returns gas value, even with gas_phase_ftmw_lock=True
    assert np.isclose(
        apply_implicit_solvation_correction(15.0, "water", -4.0, -1.5, phase="gas", gas_phase_ftmw_lock=True),
        15.0,
    )
    # PCM and ddCOSMO succeed when lock is False
    assert np.isclose(
        apply_implicit_solvation_correction(15.0, "acetonitrile", -4.0, -1.5, phase="pcm", gas_phase_ftmw_lock=False),
        12.5,
    )
    assert np.isclose(
        apply_implicit_solvation_correction(15.0, "acetonitrile", -4.0, -1.5, phase="ddcosmo", gas_phase_ftmw_lock=False),
        12.5,
    )
    # Lock=True blocks continuum solvation with ERR_DOMAIN_VIOLATION
    for sol_phase in ("pcm", "ddcosmo"):
        with pytest.raises(RuntimeError, match="ERR_DOMAIN_VIOLATION"):
            apply_implicit_solvation_correction(15.0, "water", -4.0, -1.5, phase=sol_phase, gas_phase_ftmw_lock=True)

    # Arbitrary-T ideal-gas molar volume V_m = R*T/P (fixing hardcoded 24.46 L/mol)
    r_l_atm = 0.082057366080960
    for temp_k, p_atm in ((298.15, 1.0), (500.0, 1.0), (1000.0, 2.5)):
        k1 = calculate_eyring_rate(delta_g=12.0, temp=temp_k, reaction_order=1, pressure_atm=p_atm)
        k2 = calculate_eyring_rate(delta_g=12.0, temp=temp_k, reaction_order=2, pressure_atm=p_atm)
        expected_ratio = (r_l_atm * temp_k) / p_atm
        assert np.isclose(k2 / k1, expected_ratio, rtol=1e-12)


def test_mace_pre_opt_no_silent_emt_fallback() -> None:
    """Verify MACEPreOptimizer raises explicit RuntimeError when calculator is absent and rejects charged MACE."""
    opt_charged_mace = MACEPreOptimizer(backend="mace", charge=-1, multiplicity=1)
    assert opt_charged_mace.calculator is None
    with pytest.raises(RuntimeError, match="No valid MLFF/QM backend available"):
        opt_charged_mace.pre_relax_geometry(
            ["O", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]], dtype=np.float64),
        )

    opt_unknown = MACEPreOptimizer(backend="nonexistent_backend")
    assert opt_unknown.calculator is None
    with pytest.raises(RuntimeError):
        opt_unknown.pre_relax_geometry(
            ["O", "H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.75, 0.5], [0.0, -0.75, 0.5]], dtype=np.float64),
        )


def test_rrkm_beyer_swinehart_exact_counts_and_scaling() -> None:
    """Verify exact Beyer-Swinehart microcanonical RRKM k(E) = sigma * N^ddagger(E - E_0) / (h * rho(E))."""
    # Hand-verifiable exact case:
    # Reactant modes: [100, 200] cm^-1 -> at E = 400 cm^-1 (dE = 1):
    #   states at 400: (4,0), (2,1), (0,2) => count = 3 => rho(400) = 3.0 (cm^-1)^-1
    # TS mode: [100] cm^-1, E_0 = 200 cm^-1 -> excess energy = 200 cm^-1:
    #   states <= 200: n = 0, 1, 2 => N^ddagger(200) = 3
    # Exact k(400) = sigma * 3 / (h * 3) = sigma / H_CM1_S
    solver = RRKMSolver(
        frequencies=[100.0, 200.0],
        ts_frequencies=[100.0],
        e0_cm1=200.0,
        reaction_degeneracy=2.0,
    )
    assert solver.compute_kE(energy=199.0, dE=1.0) == 0.0
    k_exact = solver.compute_kE(energy=400.0, dE=1.0)
    expected = 2.0 / H_CM1_S
    assert np.isclose(k_exact, expected, rtol=1e-12)

    # Multi-mode physical molecule check: k(E) increases with E above E_0 and stays finite & physical
    r_freqs = [350.0, 520.0, 780.0, 1050.0, 1450.0, 2950.0]
    ts_freqs = [410.0, 620.0, 890.0, 1380.0, 2960.0]
    poly_solver = RRKMSolver(frequencies=r_freqs, ts_frequencies=ts_freqs, e0_cm1=3500.0)
    assert poly_solver.compute_kE(3000.0, dE=10.0) == 0.0
    k_low = poly_solver.compute_kE(4000.0, dE=10.0)
    k_mid = poly_solver.compute_kE(6000.0, dE=10.0)
    k_high = poly_solver.compute_kE(9000.0, dE=10.0)
    assert 0.0 < k_low < k_mid < k_high < 1e15


def test_chem312_builtin_specs_and_intermolecular_21_atom_validator() -> None:
    """Verify all 3 CHEM 312 specs including the 21-atom intermolecular step (C5H11O5-, Q=-1)."""
    specs = list_builtin_specs()
    assert "team1_propargyl_formate" in specs
    assert "team4_ec_cyclization" in specs
    assert "team4_diol_dmc_intermolecular" in specs

    s1 = get_builtin_spec("team1_propargyl_formate")
    assert (s1["n_atoms"], s1["charge"], s1["multiplicity"], s1["formula"]) == (22, 0, 1, "C5H10O6S")

    s4 = get_builtin_spec("team4_ec_cyclization")
    assert (s4["n_atoms"], s4["charge"], s4["multiplicity"], s4["formula"]) == (15, -1, 1, "C4H7O4-")

    s4_inter = get_builtin_spec("team4_diol_dmc_intermolecular")
    assert (s4_inter["n_atoms"], s4_inter["charge"], s4_inter["multiplicity"], s4_inter["formula"]) == (21, -1, 1, "C5H11O5-")

    # Build valid non-clashing 21-atom coordinates for C5H11O5- and verify validator passes
    symbols_21 = ["C"] * 5 + ["H"] * 11 + ["O"] * 5
    coords_a = np.array([[float(i) * 1.5, 0.0, 0.0] for i in range(21)], dtype=np.float64)
    coords_b = coords_a + np.array([0.0, 0.1, 0.0])
    report = validate_stationary_stoichiometry(
        {"R": {"symbols": symbols_21, "coords": coords_a}, "P": {"symbols": symbols_21, "coords": coords_b}},
        expected_charge=-1,
        expected_multiplicity=1,
        expected_formula="C5H11O5-",
    )
    assert report["valid"] is True
    assert report["n_atoms"] == 21
    assert report["formula"] == "C5H11O5-"
    assert report["per_state"]["R"]["n_electrons"] == 82  # 5*6 + 11*1 + 5*8 - (-1) = 82

    # Negative check: neutral charge Q=0 on C5H11O5 with M=1 violates electron parity (81 electrons, singlet)
    with pytest.raises(ValueError, match="parity"):
        validate_stationary_stoichiometry(
            {"R": {"symbols": symbols_21, "coords": coords_a}},
            expected_charge=0,
            expected_multiplicity=1,
        )

    # Negative check: mismatched atom count between R and P
    with pytest.raises(ValueError, match="Atom count mismatch"):
        validate_stationary_stoichiometry(
            {
                "R": {"symbols": symbols_21, "coords": coords_a},
                "P": {"symbols": symbols_21[:-1], "coords": coords_b[:-1]},
            },
            expected_charge=-1,
            expected_multiplicity=1,
        )


def test_cochem_modules_entry_point_and_provider_operations(tmp_path: Path) -> None:
    """Verify cochem.modules entry point metadata and real execution of provider operations."""
    meta = kinetic_provider()
    assert meta["module_id"] == "kinetic"
    for op in ("rxn_profile", "validate_stoichiometry", "eyring_rate", "rrkm_rate"):
        assert op in meta["operations"]

    # Check installed entry point discovery if package is installed
    eps = importlib.metadata.entry_points(group="cochem.modules")
    kinetic_eps = [ep for ep in eps if ep.name == "kinetic"]
    if kinetic_eps:
        loaded_fn = kinetic_eps[0].load()
        assert loaded_fn()["module_id"] == "kinetic"

    # Execute eyring_rate handoff
    eyring_req = tmp_path / "eyring_req.json"
    eyring_req.write_text(
        json.dumps(
            {
                "operation": "eyring_rate",
                "parameters": {
                    "delta_g_gas_kcal": 18.0,
                    "delta_g_solv_ts_kcal": -3.0,
                    "delta_g_solv_reactants_kcal": -1.0,
                    "phase": "pcm",
                    "solvent": "acetonitrile",
                    "temp_k": 298.15,
                    "reaction_order": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    eyring_res = execute_handoff(eyring_req, tmp_path / "eyring_out")
    assert eyring_res["operation"] == "eyring_rate"
    assert np.isclose(eyring_res["result"]["delta_g_eff_kcal"], 16.0)
    assert eyring_res["result"]["k_tst"] > 0.0

    # Execute rrkm_rate handoff
    rrkm_req = tmp_path / "rrkm_req.json"
    rrkm_req.write_text(
        json.dumps(
            {
                "operation": "rrkm_rate",
                "parameters": {
                    "reactant_frequencies_cm1": [100.0, 200.0],
                    "ts_frequencies_cm1": [100.0],
                    "e0_cm1": 200.0,
                    "energy_cm1": 400.0,
                    "dE_cm1": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    rrkm_res = execute_handoff(rrkm_req, tmp_path / "rrkm_out")
    assert rrkm_res["operation"] == "rrkm_rate"
    assert np.isclose(rrkm_res["result"]["k_E_s_inv"], 1.0 / H_CM1_S, rtol=1e-12)

    # Execute CLI list-specs and validate-stoich
    assert cli_main(["list-specs"]) == 0
    geom_dir = tmp_path / "geoms"
    syms = ["C"] * 4 + ["H"] * 7 + ["O"] * 4
    coords = np.array([[float(i) * 1.4, 0.0, 0.0] for i in range(15)], dtype=np.float64)
    for st in ("R", "TS1", "Int_tet", "TS2", "P"):
        write_xyz(geom_dir / f"{st}.xyz", syms, coords, f"test {st}")
    assert cli_main(["validate-stoich", "--input-dir", str(geom_dir), "--spec", "team4_ec_cyclization"]) == 0


def test_qualified_chem312_benchmark_artifacts_and_hartree_consistency() -> None:
    """Verify Case A, Case B, and 21-atom intermolecular benchmark tables, SHA-256 manifests, and <0.01 kcal/mol consistency."""
    import csv
    from cochem_kinetic.rxn_profile import HARTREE_TO_KCAL, sha256_file

    bench_root = Path(__file__).parent.parent / "benchmarks" / "chem312_t_kin_01"
    cases = [
        ("case_a_team1_formate", 22, 0, "C5H10O6S"),
        ("case_b_team4_ec_cyclization", 15, -1, "C4H7O4-"),
    ]
    for folder, n_atoms, charge, formula in cases:
        case_dir = bench_root / folder
        manifest = json.loads((case_dir / "manifest_sha256.json").read_text(encoding="utf-8"))
        for rel, expected_digest in manifest["files_sha256"].items():
            fpath = case_dir / rel
            assert fpath.is_file(), f"Missing manifest file {fpath}"
            assert sha256_file(fpath) == expected_digest, f"SHA-256 mismatch for {fpath}"

        with open(case_dir / "stationary_table.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["state"] for r in rows] == ["R", "TS1", "Int_tet", "TS2", "P"]
        assert [int(r["n_imag"]) for r in rows] == [0, 1, 0, 1, 0]
        assert all(r["status"] == "CONVERGED" for r in rows)
        assert all(int(r["n_atoms"]) == n_atoms and int(r["charge"]) == charge for r in rows)

        ref = rows[0]
        for r in rows:
            de_calc = (float(r["e_dft_ha"]) - float(ref["e_dft_ha"])) * HARTREE_TO_KCAL
            dh_calc = (float(r["h_tot_ha"]) - float(ref["h_tot_ha"])) * HARTREE_TO_KCAL
            dg_rrho_calc = (float(r["g_rrho_ha"]) - float(ref["g_rrho_ha"])) * HARTREE_TO_KCAL
            dg_qrrho_calc = (float(r["g_qrrho_ha"]) - float(ref["g_qrrho_ha"])) * HARTREE_TO_KCAL
            assert abs(de_calc - float(r["delta_e_kcal"])) < 0.01
            assert abs(dh_calc - float(r["delta_h_kcal"])) < 0.01
            assert abs(dg_rrho_calc - float(r["delta_g_rrho_kcal"])) < 0.01
            assert abs(dg_qrrho_calc - float(r["delta_g_qrrho_kcal"])) < 0.01

        results = json.loads((case_dir / "results.json").read_text(encoding="utf-8"))
        assert results["stoichiometry"]["formula"] == formula
        assert results["states"]["R"]["dft_cpu_benchmark"]["converged"] is True
        assert results["irc_verification"]["TS1"]["max_com_drift_ang"] < 1e-8
        assert results["irc_verification"]["TS2"]["max_com_drift_ang"] < 1e-8

    # Verify 21-atom intermolecular balanced input spec on disk
    inter_dir = bench_root / "case_b_intermolecular_21atom_spec" / "inputs"
    assert cli_main([
        "rxn-profile",
        "--input-dir", str(inter_dir),
        "--output-dir", str(bench_root / "case_b_intermolecular_21atom_spec"),
        "--spec", "team4_diol_dmc_intermolecular",
        "--stage", "validate",
    ]) == 0

