"""
Zero-mock regression tests for WAVE-3A CHEM 410 E2E remediation:
  - FINDING-410-02: Exact 1D unsymmetrical Eckart (1930) / Johnston & Heicklen (1962)
    tunneling transmission probability P(E) and thermally averaged kappa(T).
  - FINDING-410-07: Clean stdout JSON isolation (routing third-party stdout banners
    such as NVIDIA Warp initialization to stderr).
  - FINDING-410-09: `cochem-kinetic eyring` CLI subcommand and optional custom
    active-coordinate column header in stationary_table.csv (backward compatible).
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from cochem_kinetic.chem312_specs import write_xyz
from cochem_kinetic.cli import _run_stage_subprocess, build_parser
from cochem_kinetic.cli import main as cli_main
from cochem_kinetic.math.tunneling import (
    HC_KCAL_PER_CM1,
    R_KCAL_MOL_K,
    calculate_eckart_tunneling,
    calculate_wigner_correction,
    eckart_transmission_probability,
)
from cochem_kinetic.rxn_profile import (
    DEFAULT_COORDINATE_COLUMN,
    assemble_and_export_profile,
    compute_rrho_thermochemistry,
    redirect_stdout_to_stderr,
)
from kinetic_core.thermo import calculate_eyring_rate

# =============================================================================
# FINDING-410-02: Exact Eckart (1930) / Johnston & Heicklen (1962) Tunneling
# =============================================================================


@pytest.mark.parametrize(
    ("alpha", "u_star", "expected_kappa"),
    [
        # Exact Johnston & Heicklen (J. Phys. Chem. 1962, 66, 532) Table I
        # symmetric barrier (alpha_1 = alpha_2 = alpha) benchmark grid
        (0.5, 2.0, 1.150),
        (1.0, 2.0, 1.252),
        (2.0, 2.0, 1.321),
        (4.0, 2.0, 1.290),
        (8.0, 2.0, 1.241),
        (12.0, 2.0, 1.224),
        (2.0, 4.0, 1.906),
        (4.0, 4.0, 2.024),
        (8.0, 4.0, 2.049),
        (12.0, 4.0, 2.071),
        (2.0, 6.0, 2.895),
        (4.0, 6.0, 3.692),
        (8.0, 6.0, 4.540),
        (12.0, 6.0, 5.199),
        (2.0, 8.0, 4.550),
        (4.0, 8.0, 7.603),
        (8.0, 8.0, 13.780),
        (12.0, 8.0, 21.769),
    ],
)
def test_eckart_symmetric_johnston_heicklen_table1_benchmarks(
    alpha: float,
    u_star: float,
    expected_kappa: float,
) -> None:
    """Verify exact Johnston & Heicklen (1962) Table I symmetric benchmark values."""
    nu_cm1 = 1000.0
    hv_kcal = HC_KCAL_PER_CM1 * nu_cm1
    v_barrier = (alpha * hv_kcal) / (2.0 * math.pi)
    temp_k = hv_kcal / (R_KCAL_MOL_K * u_star)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = calculate_eckart_tunneling(
            imag_freq=nu_cm1,
            temp=temp_k,
            V_f=v_barrier,
            V_r=v_barrier,
        )

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert res["provenance_tag"] == "[M]"
    assert res["warning"] is None
    assert math.isclose(float(res["kappa"]), expected_kappa, rel_tol=2e-3)


def test_eckart_parabolic_limit_high_t_limit_and_reversibility() -> None:
    """Verify parabolic limit, high-T -> 1.0+, and microscopic reversibility."""
    nu_cm1 = 1000.0
    hv_kcal = HC_KCAL_PER_CM1 * nu_cm1
    u_star = 2.0
    temp_k = hv_kcal / (R_KCAL_MOL_K * u_star)
    # Large symmetric barrier approaches exact parabolic (u/2) / sin(u/2)
    v_large = (1000.0 * hv_kcal) / (2.0 * math.pi)
    res_par = calculate_eckart_tunneling(nu_cm1, temp_k, v_large, v_large)
    parabolic_exact = (u_star / 2.0) / math.sin(u_star / 2.0)
    assert math.isclose(float(res_par["kappa"]), parabolic_exact, rel_tol=1e-3)

    # High-T limit: kappa(T) -> 1.0+ monotonically as T increases
    kappas = [
        float(calculate_eckart_tunneling(896.91, t, 43.05, 65.6)["kappa"])
        for t in (298.15, 500.0, 1000.0, 5000.0, 20000.0)
    ]
    assert kappas == sorted(kappas, reverse=True)
    assert all(k > 1.0 for k in kappas)
    assert abs(kappas[-1] - 1.0) < 1e-3

    # Microscopic reversibility: kappa(V_f, V_r) == kappa(V_r, V_f)
    for v_f, v_r in ((43.0457, 65.6), (10.0, 25.0), (0.2, 0.5)):
        k_fwd = float(calculate_eckart_tunneling(896.91, 298.15, v_f, v_r)["kappa"])
        k_rev = float(calculate_eckart_tunneling(896.91, 298.15, v_r, v_f)["kappa"])
        assert math.isclose(k_fwd, k_rev, rel_tol=1e-12)


def test_eckart_ch3nc_isomerization_chem410_benchmark() -> None:
    """Verify CHEM 410 M3 CH3NC -> CH3CN Eckart tunneling at 298.15 K and 500 K."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_298 = calculate_eckart_tunneling(
            imag_freq=-896.91,
            temp=298.15,
            V_f=43.0457,
            V_r=65.6,
        )
        res_298_pos = calculate_eckart_tunneling(
            imag_freq=896.91,
            temp=298.15,
            V_f=43.05,
            V_r=65.6,
        )
        res_500 = calculate_eckart_tunneling(
            imag_freq=896.91,
            temp=500.0,
            V_f=43.0457,
            V_r=65.6,
        )

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert res_298["provenance_tag"] == "[M]"
    assert res_298["warning"] is None
    assert res_500["provenance_tag"] == "[M]"
    assert res_500["warning"] is None

    k_298 = float(res_298["kappa"])
    k_298_pos = float(res_298_pos["kappa"])
    k_500 = float(res_500["kappa"])
    assert math.isclose(k_298, 2.5596, rel_tol=1e-3)
    assert math.isclose(k_298_pos, 2.5596, rel_tol=1e-3)
    assert math.isclose(k_500, 1.3455, rel_tol=1e-3)
    assert k_298 < 10.0  # Regression guard against old 1e15 overflow bug


def test_eckart_transmission_probability_bounds_and_cos_branch() -> None:
    """Verify 0 <= P(E) <= 1 everywhere, barrier-top P(V_f), and cos branch."""
    # Large barrier regime where 2*pi*(a+b) >> 50 (retains cosh(2*pi*d) in log-space)
    hv = HC_KCAL_PER_CM1 * 896.91
    a1 = (2.0 * math.pi * 43.05) / hv
    a2 = (2.0 * math.pi * 65.6) / hv
    s_top = 2.0 * math.sqrt(a1 * a2)
    z_top = 2.0 * math.sqrt(a1 * a2 - (math.pi**2) / 4.0)
    expected_p_top = 1.0 / (1.0 + math.exp(z_top - s_top))
    p_top = eckart_transmission_probability(43.05, 896.91, 43.05, 65.6)
    assert math.isclose(p_top, expected_p_top, rel_tol=1e-10)
    assert 0.5 < p_top < 0.51

    for energy in (0.0, 10.0, 30.0, 43.05, 60.0, 200.0, 5000.0):
        p_val = eckart_transmission_probability(energy, 896.91, 43.05, 65.6)
        assert 0.0 <= p_val <= 1.0

    # Small barrier where 4*alpha_1*alpha_2 / pi^2 < 1 -> cos(...) branch
    p_cos_low = eckart_transmission_probability(0.1, 500.0, 0.2, 0.2)
    p_cos_top = eckart_transmission_probability(0.2, 500.0, 0.2, 0.2)
    p_cos_hi = eckart_transmission_probability(5.0, 500.0, 0.2, 0.2)
    assert 0.0 < p_cos_low < p_cos_top < p_cos_hi <= 1.0
    res_cos = calculate_eckart_tunneling(500.0, 298.15, 0.2, 0.2)
    assert res_cos["provenance_tag"] == "[M]"
    assert 1.0 < float(res_cos["kappa"]) < 2.0


# =============================================================================
# FINDING-410-07: Clean JSON Stdout Isolation (Warp / Third-Party Noise)
# =============================================================================


def test_redirect_stdout_to_stderr_isolates_third_party_noise(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Third-party prints inside redirect_stdout_to_stderr route to stderr."""
    with redirect_stdout_to_stderr():
        print("Warp 1.18.0 initialized:\n  CUDA Toolkit 12.8, Driver 12.8")
    print(json.dumps({"status": "ok", "kappa": 2.5596}))

    captured = capsys.readouterr()
    assert "Warp 1.18.0 initialized" not in captured.out
    assert "Warp 1.18.0 initialized" in captured.err
    parsed = json.loads(captured.out)
    assert parsed == {"status": "ok", "kappa": 2.5596}


def test_run_stage_subprocess_routes_child_stdout_banner_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Isolated stage subprocess stdout (e.g. Warp banner) is forwarded to stderr."""
    _run_stage_subprocess(
        [
            sys.executable,
            "-c",
            "print('Warp 1.18.0 initialized:'); print('  Devices: cpu')",
        ]
    )
    print(json.dumps({"stage": "assemble", "clean": True}))

    captured = capsys.readouterr()
    assert "Warp 1.18.0 initialized" not in captured.out
    assert "Warp 1.18.0 initialized" in captured.err
    assert json.loads(captured.out) == {"stage": "assemble", "clean": True}

    with pytest.raises(subprocess.CalledProcessError):
        _run_stage_subprocess([sys.executable, "-c", "import sys; sys.exit(3)"])


def test_rxn_profile_validate_and_eyring_cli_subprocess_json_clean(
    tmp_path: Path,
) -> None:
    """Real subprocess CLI invocations with --json emit 100% clean parseable JSON."""
    in_dir = tmp_path / "h2_in"
    out_dir = tmp_path / "h2_out"
    write_xyz(
        in_dir / "R.xyz",
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=np.float64),
        "H2 R",
    )
    write_xyz(
        in_dir / "P.xyz",
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.78]], dtype=np.float64),
        "H2 P",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_kinetic.cli",
            "rxn-profile",
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--charge",
            "0",
            "--multiplicity",
            "1",
            "--stage",
            "validate",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["valid"] is True
    assert payload["input_mode"] == "2pt_auto"


# =============================================================================
# FINDING-410-09: `cochem-kinetic eyring` CLI & Custom Coordinate Label
# =============================================================================


def test_cli_eyring_subcommand_dh_ds_with_wigner_and_eckart_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CHEM 410 M3 Eyring CLI with dH‡/dS‡, Wigner, Eckart, and linear fit."""
    dh_kcal = 43.0457
    ds_cal = 1.2101
    nu_imag = -896.91
    vf_kcal = 43.0457
    vr_kcal = 65.6

    rc = cli_main(
        [
            "eyring",
            "--dh-ddagger-kcal",
            str(dh_kcal),
            "--ds-ddagger-cal",
            str(ds_cal),
            "--nu-imag-cm1",
            str(nu_imag),
            "--vf-kcal",
            str(vf_kcal),
            "--vr-kcal",
            str(vr_kcal),
            "--temps",
            "298.15,400,500,600",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == "cochem.kinetic.eyring/1"
    assert out["mode"] == "dh_ds"
    assert out["rate_units"] == "s^-1"
    assert len(out["rows"]) == 4

    row_298 = out["rows"][0]
    row_500 = out["rows"][2]
    assert row_298["temp_k"] == pytest.approx(298.15)
    expected_dg_298 = dh_kcal - 298.15 * (ds_cal / 1000.0)
    expected_k_tst_298 = calculate_eyring_rate(
        delta_g=expected_dg_298, wigner_coeff=1.0, temp=298.15, reaction_order=1
    )
    expected_wig_298 = calculate_wigner_correction(abs(nu_imag), 298.15)
    assert math.isclose(row_298["dg_ddagger_kcal"], expected_dg_298, rel_tol=1e-12)
    assert math.isclose(row_298["k_tst"], expected_k_tst_298, rel_tol=1e-12)
    assert math.isclose(row_298["kappa_wigner"], expected_wig_298, rel_tol=1e-12)
    assert math.isclose(row_298["kappa_wigner"], 1.7805, rel_tol=1e-3)
    assert math.isclose(row_298["kappa_eckart"], 2.5596, rel_tol=1e-3)
    assert row_298["eckart_provenance_tag"] == "[M]"
    assert row_298["eckart_warning"] is None
    assert row_298["effective_method"] == "eckart"
    assert math.isclose(
        row_298["k_eckart"],
        row_298["k_tst"] * row_298["kappa_eckart"],
        rel_tol=1e-12,
    )

    assert math.isclose(row_500["kappa_eckart"], 1.3455, rel_tol=1e-3)

    # Exact Eyring linear regression recovery
    reg = out["eyring_regression"]
    assert reg["n_points"] == 4
    assert math.isclose(reg["r_squared"], 1.0, abs_tol=1e-12)
    assert math.isclose(reg["dh_ddagger_fit_kcal"], dh_kcal, rel_tol=1e-10)
    assert math.isclose(reg["ds_ddagger_fit_cal"], ds_cal, rel_tol=1e-9)


def test_cli_eyring_dg_direct_text_and_validation_guards(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --dg-ddagger-kcal mode, table rendering, order=2, and error guards."""
    rc = cli_main(
        [
            "eyring",
            "--dg-ddagger-kcal",
            "18.5",
            "--nu-imag-cm1",
            "500.0",
            "--temps",
            "298.15,350.0",
        ]
    )
    assert rc == 0
    txt = capsys.readouterr().out
    assert "k_TST (s^-1)" in txt
    assert "k_Wig_kappa" in txt
    assert "298.15" in txt
    assert "350.00" in txt

    # Bimolecular order 2 JSON check
    rc2 = cli_main(
        [
            "eyring",
            "--dg-ddagger-kcal",
            "15.0",
            "--temps",
            "298.15",
            "--reaction-order",
            "2",
            "--pressure-atm",
            "2.0",
            "--json",
        ]
    )
    assert rc2 == 0
    p2 = json.loads(capsys.readouterr().out)
    assert p2["rate_units"] == "L mol^-1 s^-1"
    expected_k2 = calculate_eyring_rate(
        delta_g=15.0, temp=298.15, reaction_order=2, pressure_atm=2.0
    )
    assert math.isclose(p2["rows"][0]["k_tst"], expected_k2, rel_tol=1e-12)

    # Fail-closed validation checks
    with pytest.raises(ValueError, match="either --dg-ddagger-kcal"):
        cli_main(["eyring", "--temps", "298.15"])
    with pytest.raises(ValueError, match="not both"):
        cli_main(
            [
                "eyring",
                "--dg-ddagger-kcal",
                "15.0",
                "--dh-ddagger-kcal",
                "15.0",
                "--ds-ddagger-cal",
                "0.0",
            ]
        )
    with pytest.raises(ValueError, match="Both --vf-kcal and --vr-kcal"):
        cli_main(
            [
                "eyring",
                "--dg-ddagger-kcal",
                "15.0",
                "--nu-imag-cm1",
                "500.0",
                "--vf-kcal",
                "15.0",
            ]
        )
    with pytest.raises(ValueError, match="--nu-imag-cm1 is required"):
        cli_main(
            [
                "eyring",
                "--dg-ddagger-kcal",
                "15.0",
                "--vf-kcal",
                "15.0",
                "--vr-kcal",
                "20.0",
            ]
        )


def test_stationary_table_custom_coordinate_label_and_backward_compat(
    tmp_path: Path,
) -> None:
    """Verify optional custom coordinate label in stationary_table.csv + compat."""
    symbols = ["C", "N"]
    coords_r = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.17]], dtype=np.float64)
    coords_ts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.954]], dtype=np.float64)
    coords_p = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]], dtype=np.float64)

    thermo_min = compute_rrho_thermochemistry(symbols, coords_r, np.array([2100.0]))
    thermo_ts = compute_rrho_thermochemistry(symbols, coords_ts, np.array([-896.91]))
    states_data = {
        "R": {
            "symbols": symbols,
            "coords": coords_r,
            "hessian": {"thermo": thermo_min, "aimnet2_energy_ha": -132.7},
            "dft": {
                "converged": True,
                "e_tot_ha": -132.70,
                "wall_seconds": 0.1,
                "engine": "cpu",
            },
        },
        "TS": {
            "symbols": symbols,
            "coords": coords_ts,
            "hessian": {
                "thermo": thermo_ts,
                "aimnet2_energy_ha": -132.63,
                "imag_mode_bond_activity": [
                    {
                        "atoms": [0, 1],
                        "label": "C0-N1",
                        "elements": ["C", "N"],
                        "distance_ang": 1.954,
                        "mode_stretch_component": 0.99,
                        "abs_stretch": 0.99,
                    }
                ],
            },
            "dft": {
                "converged": True,
                "e_tot_ha": -132.63,
                "wall_seconds": 0.1,
                "engine": "cpu",
            },
        },
        "P": {
            "symbols": symbols,
            "coords": coords_p,
            "hessian": {"thermo": thermo_min, "aimnet2_energy_ha": -132.73},
            "dft": {
                "converged": True,
                "e_tot_ha": -132.73,
                "wall_seconds": 0.1,
                "engine": "cpu",
            },
        },
    }
    stoich = {"formula": "CN", "n_atoms": 2, "charge": 0, "multiplicity": 1}

    # 1. Default preserves backward-compatible `forming_breaking_co` column
    out_default = tmp_path / "default_col"
    assemble_and_export_profile(
        output_dir=out_default,
        case_name="cn_default",
        reaction_equation="R -> TS -> P",
        states_data=states_data,
        stoich_report=stoich,
        irc_reports={},
        metadata={"stationary_points": ["R", "TS", "P"], "temp_k": 298.15},
    )
    with open(out_default / "stationary_table.csv", encoding="utf-8") as fh:
        rows_def = list(csv.DictReader(fh))
    assert DEFAULT_COORDINATE_COLUMN in rows_def[0]
    assert rows_def[1][DEFAULT_COORDINATE_COLUMN] == "C0-N1=1.954A"

    # 2. Custom coordinate label overrides CSV header, MD header, and results metadata
    out_custom = tmp_path / "custom_col"
    assemble_and_export_profile(
        output_dir=out_custom,
        case_name="cn_custom",
        reaction_equation="R -> TS -> P",
        states_data=states_data,
        stoich_report=stoich,
        irc_reports={},
        metadata={"stationary_points": ["R", "TS", "P"], "temp_k": 298.15},
        coordinate_label="active_cn_coord",
    )
    with open(out_custom / "stationary_table.csv", encoding="utf-8") as fh:
        rows_cust = list(csv.DictReader(fh))
    assert "active_cn_coord" in rows_cust[0]
    assert DEFAULT_COORDINATE_COLUMN not in rows_cust[0]
    assert rows_cust[1]["active_cn_coord"] == "C0-N1=1.954A"
    assert "| active_cn_coord |" in (out_custom / "stationary_table.md").read_text(
        encoding="utf-8"
    )
    res_json = json.loads((out_custom / "results.json").read_text(encoding="utf-8"))
    assert res_json["metadata"]["coordinate_label"] == "active_cn_coord"

    # 3. CLI parser accepts --coordinate-label and --json on rxn-profile
    parsed = build_parser().parse_args(
        [
            "rxn-profile",
            "--input-dir",
            "in_dir",
            "--output-dir",
            "out_dir",
            "--coordinate-label",
            "forming_breaking_cn",
            "--json",
        ]
    )
    assert parsed.coordinate_label == "forming_breaking_cn"
    assert parsed.json is True

    # 4. Invalid / reserved coordinate label fails closed
    for bad_label in ("", "   ", "state", "bad,header"):
        with pytest.raises(ValueError, match="coordinate_label"):
            assemble_and_export_profile(
                output_dir=tmp_path / "bad_col",
                case_name="bad",
                reaction_equation="R -> TS -> P",
                states_data=states_data,
                stoich_report=stoich,
                irc_reports={},
                metadata={"stationary_points": ["R", "TS", "P"]},
                coordinate_label=bad_label,
            )
