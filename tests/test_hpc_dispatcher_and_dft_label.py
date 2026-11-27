"""Zero-mock tests for KineticHPCDispatcher and exact DF-wB97X-D3(0) method labeling."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from cochem_kinetic.chem312_specs import write_xyz
from cochem_kinetic.cli import build_parser
from cochem_kinetic.engine.hpc_dispatcher import (
    DEFAULT_ENTRYPOINT_MODULE,
    KineticHPCDispatcher as PackageHPCDispatcher,
)
from cochem_kinetic.rxn_profile import (
    R_KCAL_MOL_K,
    assemble_and_export_profile,
    compute_rrho_thermochemistry,
    format_dft_method_label,
)
from kinetic_core.engine.hpc_dispatcher import (
    KineticHPCDispatcher as CoreHPCDispatcher,
)


def test_format_dft_method_label_distinguishes_wb97x_d3_zero() -> None:
    label = format_dft_method_label("wb97x", "d3zero")
    assert label == "DF-wB97X-D3(0)"
    assert label != "DF-wB97X-D3"
    assert format_dft_method_label("wb97x", "d3bj") == "DF-wB97X-D3(BJ)"
    assert format_dft_method_label("wb97x", "none") == "DF-wB97X"
    assert format_dft_method_label("wb97m_v", None) == "DF-wB97M-V"


def test_hpc_dispatcher_rejects_nonexistent_worker_module() -> None:
    with pytest.raises(RuntimeError, match="not importable"):
        PackageHPCDispatcher(entrypoint_module="cochem_kinetic.worker")
    with pytest.raises(RuntimeError, match="not importable"):
        CoreHPCDispatcher(entrypoint_module="kinetic_core.worker")


def test_hpc_dispatcher_builds_real_cli_command() -> None:
    disp = PackageHPCDispatcher(execution_mode="slurm")
    cmd = disp.build_command({"cli_args": ["list-specs"]})
    assert cmd[0] == "sbatch"
    assert DEFAULT_ENTRYPOINT_MODULE in cmd[-1]
    assert "cochem_kinetic.worker" not in cmd[-1]
    assert "list-specs" in cmd[-1]


def test_hpc_dispatcher_fails_closed_when_slurm_unavailable() -> None:
    if shutil.which("sbatch") is not None:
        pytest.skip("SLURM sbatch is installed on this host")

    pkg_disp = PackageHPCDispatcher(execution_mode="slurm")
    with pytest.raises(RuntimeError, match="sbatch executable not found"):
        asyncio.run(
            pkg_disp.dispatch_reaction_network([{"cli_args": ["list-specs"]}])
        )

    core_disp = CoreHPCDispatcher(execution_mode="slurm")
    with pytest.raises(RuntimeError, match="sbatch executable not found"):
        asyncio.run(
            core_disp.dispatch_reaction_network([{"cli_args": ["list-specs"]}])
        )


def test_hpc_dispatcher_local_mode_runs_real_cli_entrypoint() -> None:
    disp = PackageHPCDispatcher(
        execution_mode="local",
        python_executable=sys.executable,
    )

    async def _run() -> list[str]:
        payloads, workers = await disp.dispatch_reaction_network(
            [{"cli_args": ["list-specs"]}]
        )
        assert len(payloads) == 1
        return list(await asyncio.gather(*workers))

    outputs = asyncio.run(_run())
    assert len(outputs) == 1
    assert len(outputs[0]) > 0


def test_hpc_dispatcher_requires_cli_args_for_payloads() -> None:
    disp = PackageHPCDispatcher(execution_mode="local")
    with pytest.raises(ValueError, match="cli_args"):
        disp.build_command({"task": "transition_state_rate_calculation"})


def test_aimd_hoover_source_has_no_main_stub() -> None:
    import ast

    aimd_path = (
        Path(__file__).resolve().parents[1] / "kinetic_core" / "aimd_hoover.py"
    )
    tree = ast.parse(aimd_path.read_text(encoding="utf-8"))
    defined_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert all(not name.startswith("dummy") for name in defined_names)


def test_cli_pressure_atm_argument_validation() -> None:
    """M4 regression: CLI rxn-profile exposes --pressure-atm (default 1.0, must be > 0)."""
    parser = build_parser()
    args_default = parser.parse_args(
        ["rxn-profile", "--input-dir", "in_dir", "--output-dir", "out_dir"]
    )
    assert args_default.pressure_atm == pytest.approx(1.0)

    args_high_p = parser.parse_args(
        [
            "rxn-profile",
            "--input-dir",
            "in_dir",
            "--output-dir",
            "out_dir",
            "--pressure-atm",
            "10.0",
        ]
    )
    assert args_high_p.pressure_atm == pytest.approx(10.0)

    for bad_val in ("0", "0.0", "-1.5", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "rxn-profile",
                    "--input-dir",
                    "in_dir",
                    "--output-dir",
                    "out_dir",
                    "--pressure-atm",
                    bad_val,
                ]
            )


def test_dispatch_tp_grid_passes_distinct_pressure_atm_to_real_cli(tmp_path: Path) -> None:
    """M4 regression: dispatch_tp_grid in both dispatchers passes distinct --pressure-atm flags."""
    in_dir = tmp_path / "rxn_in"
    in_dir.mkdir()
    out_dir = tmp_path / "rxn_out"
    coords_r = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=np.float64)
    coords_p = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.78]], dtype=np.float64)
    write_xyz(in_dir / "R.xyz", ["H", "H"], coords_r, "H2 reactant")
    write_xyz(in_dir / "P.xyz", ["H", "H"], coords_p, "H2 product")

    base_args = [
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
    ]
    t_grid = [298.15, 400.0]
    p_grid = [1.0, 10.0]

    # 1. Package dispatcher (cochem_kinetic.engine.hpc_dispatcher)
    pkg_disp = PackageHPCDispatcher(
        execution_mode="local",
        python_executable=sys.executable,
    )

    async def _run_pkg() -> tuple[list[dict], list[str]]:
        payloads, workers = await pkg_disp.dispatch_tp_grid(
            t_grid, p_grid, base_cli_args=base_args
        )
        outputs = list(await asyncio.gather(*workers))
        return payloads, outputs

    pkg_payloads, pkg_outputs = asyncio.run(_run_pkg())
    assert len(pkg_payloads) == 4
    pkg_tp_pairs: set[tuple[float, float]] = set()
    for payload, stdout_text in zip(pkg_payloads, pkg_outputs):
        cli_args = payload["cli_args"]
        assert "--temp-k" in cli_args
        assert "--pressure-atm" in cli_args
        p_idx = cli_args.index("--pressure-atm")
        t_idx = cli_args.index("--temp-k")
        assert float(cli_args[p_idx + 1]) == pytest.approx(payload["P"])
        assert float(cli_args[t_idx + 1]) == pytest.approx(payload["T"])
        parsed_out = json.loads(stdout_text)
        assert parsed_out["pressure_atm"] == pytest.approx(payload["P"])
        assert parsed_out["temp_k"] == pytest.approx(payload["T"])
        pkg_tp_pairs.add((parsed_out["temp_k"], parsed_out["pressure_atm"]))
    assert pkg_tp_pairs == {(298.15, 1.0), (298.15, 10.0), (400.0, 1.0), (400.0, 10.0)}

    # 2. Core dispatcher (kinetic_core.engine.hpc_dispatcher)
    core_disp = CoreHPCDispatcher(
        execution_mode="local",
        python_executable=sys.executable,
    )

    async def _run_core() -> list[dict]:
        payloads, workers = await core_disp.dispatch_tp_grid(
            t_grid, p_grid, base_cli_args=base_args
        )
        await core_disp.swarm_queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        return payloads

    core_payloads = asyncio.run(_run_core())
    assert len(core_payloads) == 4
    core_tp_pairs: set[tuple[float, float]] = set()
    for payload in core_payloads:
        cli_args = payload["cli_args"]
        assert "--temp-k" in cli_args
        assert "--pressure-atm" in cli_args
        p_val = float(cli_args[cli_args.index("--pressure-atm") + 1])
        t_val = float(cli_args[cli_args.index("--temp-k") + 1])
        assert p_val == pytest.approx(payload["pressure"])
        assert t_val == pytest.approx(payload["temperature"])
        core_tp_pairs.add((t_val, p_val))
    assert core_tp_pairs == {(298.15, 1.0), (298.15, 10.0), (400.0, 1.0), (400.0, 10.0)}

    # Non-positive pressure must fail closed in both dispatchers
    with pytest.raises(ValueError, match="Pressure must be finite and > 0"):
        asyncio.run(pkg_disp.dispatch_tp_grid([298.15], [0.0], base_cli_args=base_args))
    with pytest.raises(ValueError, match="Pressure must be finite and > 0"):
        asyncio.run(core_disp.dispatch_tp_grid([298.15], [-2.0], base_cli_args=base_args))


def test_rrho_thermochemistry_pressure_shift_matches_ideal_gas_rt_ln_p(tmp_path: Path) -> None:
    """M4 regression: RRHO/qRRHO Gibbs free energy shifts by exact +R*T*ln(10) from 1.0 atm to 10.0 atm."""
    symbols = ["O", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ],
        dtype=np.float64,
    )
    freqs_cm1 = np.array([1595.0, 3657.0, 3756.0], dtype=np.float64)

    for temp_k in (298.15, 500.0):
        thermo_1 = compute_rrho_thermochemistry(
            symbols=symbols,
            coords=coords,
            freqs_cm1=freqs_cm1,
            temp_k=temp_k,
            pressure_atm=1.0,
        )
        thermo_10 = compute_rrho_thermochemistry(
            symbols=symbols,
            coords=coords,
            freqs_cm1=freqs_cm1,
            temp_k=temp_k,
            pressure_atm=10.0,
        )
        expected_dg_kcal = R_KCAL_MOL_K * temp_k * math.log(10.0)
        expected_ds_cal = -R_KCAL_MOL_K * 1000.0 * math.log(10.0)

        assert thermo_1["pressure_atm"] == pytest.approx(1.0)
        assert thermo_10["pressure_atm"] == pytest.approx(10.0)
        assert thermo_10["h_corr_kcal"] == pytest.approx(thermo_1["h_corr_kcal"], abs=1e-12)
        assert (thermo_10["s_rrho_cal_mol_k"] - thermo_1["s_rrho_cal_mol_k"]) == pytest.approx(
            expected_ds_cal, rel=1e-12
        )
        assert (thermo_10["s_qrrho_cal_mol_k"] - thermo_1["s_qrrho_cal_mol_k"]) == pytest.approx(
            expected_ds_cal, rel=1e-12
        )
        assert (thermo_10["g_corr_rrho_kcal"] - thermo_1["g_corr_rrho_kcal"]) == pytest.approx(
            expected_dg_kcal, rel=1e-12
        )
        assert (thermo_10["g_corr_qrrho_kcal"] - thermo_1["g_corr_qrrho_kcal"]) == pytest.approx(
            expected_dg_kcal, rel=1e-12
        )

    with pytest.raises(ValueError, match="pressure_atm"):
        compute_rrho_thermochemistry(symbols, coords, freqs_cm1, temp_k=298.15, pressure_atm=0.0)
    with pytest.raises(ValueError, match="pressure_atm"):
        compute_rrho_thermochemistry(symbols, coords, freqs_cm1, temp_k=298.15, pressure_atm=-5.0)

    # Verify pressure_atm is persisted in exported results.json and stationary_table.md
    thermo_10 = compute_rrho_thermochemistry(
        symbols=symbols, coords=coords, freqs_cm1=freqs_cm1, temp_k=298.15, pressure_atm=10.0
    )
    states_data = {
        "R": {
            "symbols": symbols,
            "coords": coords,
            "hessian": {"thermo": thermo_10, "aimnet2_energy_ha": -76.4},
            "dft": {"converged": True, "e_tot_ha": -76.42, "wall_seconds": 0.1, "engine": "cpu"},
        },
        "P": {
            "symbols": symbols,
            "coords": coords,
            "hessian": {"thermo": thermo_10, "aimnet2_energy_ha": -76.4},
            "dft": {"converged": True, "e_tot_ha": -76.42, "wall_seconds": 0.1, "engine": "cpu"},
        },
    }
    exported = assemble_and_export_profile(
        output_dir=tmp_path / "profile_10atm",
        case_name="h2o_10atm",
        reaction_equation="R -> P",
        states_data=states_data,
        stoich_report={"formula": "H2O", "n_atoms": 3, "charge": 0, "multiplicity": 1},
        irc_reports={},
        metadata={"temp_k": 298.15, "pressure_atm": 10.0, "stationary_points": ["R", "P"]},
    )
    results_json = json.loads(Path(exported["results_path"]).read_text(encoding="utf-8"))
    assert results_json["pressure_atm"] == pytest.approx(10.0)
    assert results_json["metadata"]["pressure_atm"] == pytest.approx(10.0)
    assert results_json["states"]["R"]["hessian"]["thermo"]["pressure_atm"] == pytest.approx(10.0)
    assert "10.0 atm" in Path(exported["md_path"]).read_text(encoding="utf-8")
