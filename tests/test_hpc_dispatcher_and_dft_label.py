"""Zero-mock tests for KineticHPCDispatcher and exact DF-wB97X-D3(0) method labeling."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from cochem_kinetic.engine.hpc_dispatcher import (
    DEFAULT_ENTRYPOINT_MODULE,
    KineticHPCDispatcher as PackageHPCDispatcher,
)
from cochem_kinetic.rxn_profile import format_dft_method_label
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
