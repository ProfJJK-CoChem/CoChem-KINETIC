from __future__ import annotations

import asyncio
import importlib.util
import shlex
import shutil
import sys
from typing import Any, Sequence


DEFAULT_ENTRYPOINT_MODULE = "cochem_kinetic.cli"


class KineticHPCDispatcher:
    """Dispatches CoChem-KINETIC CLI jobs via SLURM (`sbatch`), local subprocess, or WSL."""

    def __init__(
        self,
        execution_mode: str = "slurm",
        entrypoint_module: str = DEFAULT_ENTRYPOINT_MODULE,
        python_executable: str | None = None,
        wsl_python: str = "python3",
    ) -> None:
        mode = execution_mode.strip().lower()
        if mode not in {"slurm", "local", "wsl"}:
            raise ValueError(
                f"Unsupported execution_mode '{execution_mode}'. Expected 'slurm', 'local', or 'wsl'."
            )
        if importlib.util.find_spec(entrypoint_module) is None:
            raise RuntimeError(
                f"Entrypoint module '{entrypoint_module}' is not importable. "
                f"Use a real CoChem-KINETIC module such as '{DEFAULT_ENTRYPOINT_MODULE}'."
            )
        self.execution_mode = mode
        self.entrypoint_module = entrypoint_module
        self.python_executable = python_executable or sys.executable
        self.wsl_python = wsl_python
        self.swarm_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def _validate_backend_available(self) -> None:
        if self.execution_mode == "slurm":
            if shutil.which("sbatch") is None:
                raise RuntimeError(
                    "sbatch executable not found on PATH; SLURM HPC scheduler is unavailable on "
                    "this workstation. Pass execution_mode='local' or execution_mode='wsl' for "
                    "local execution."
                )
        elif self.execution_mode == "wsl" and sys.platform == "win32":
            if shutil.which("wsl") is None:
                raise RuntimeError(
                    "wsl executable not found on PATH; cannot dispatch WSL execution from Windows."
                )

    def _extract_cli_args(self, payload: dict[str, Any]) -> list[str]:
        raw_args = payload.get("cli_args")
        if raw_args is None and isinstance(payload.get("reaction_data"), dict):
            raw_args = payload["reaction_data"].get("cli_args")
        if raw_args is None:
            raise ValueError(
                "Dispatch payload must supply non-empty 'cli_args' (e.g. ['rxn-profile', ...]) "
                f"for real entrypoint '{self.entrypoint_module}'."
            )
        if isinstance(raw_args, str) or not isinstance(raw_args, Sequence) or len(raw_args) == 0:
            raise ValueError("'cli_args' must be a non-empty sequence of CLI argument strings.")
        return [str(arg) for arg in raw_args]

    def build_command(self, payload: dict[str, Any]) -> list[str]:
        """Construct the exact command list targeting the real `cochem_kinetic.cli` entrypoint."""
        cli_args = self._extract_cli_args(payload)
        if self.execution_mode == "slurm":
            wrap_cmd = shlex.join([self.python_executable, "-m", self.entrypoint_module, *cli_args])
            return ["sbatch", "--parsable", "--wrap", wrap_cmd]
        if self.execution_mode == "wsl" and sys.platform == "win32":
            wsl_cmd = shlex.join([self.wsl_python, "-m", self.entrypoint_module, *cli_args])
            return ["wsl", "bash", "-lc", wsl_cmd]
        return [self.python_executable, "-m", self.entrypoint_module, *cli_args]

    async def _evaluate_payload(self, payload: dict[str, Any]) -> str:
        """Execute a validated payload via SLURM `sbatch`, local Python, or WSL."""
        self._validate_backend_available()
        cmd = self.build_command(payload)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Executable '{cmd[0]}' not found during {self.execution_mode} dispatch."
            ) from exc

        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"{self.execution_mode} dispatch failed with exit code {process.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return stdout.decode(errors="replace").strip()

    async def dispatch_reaction_network(
        self, reactions: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[asyncio.Task[str]]]:
        """Dispatch a batch of reaction CLI payloads via the configured execution backend."""
        self._validate_backend_available()
        payloads: list[dict[str, Any]] = []
        workers: list[asyncio.Task[str]] = []

        for i, reaction in enumerate(reactions):
            payload: dict[str, Any] = {
                "id": i,
                "target": "/goal",
                "reaction_data": reaction,
                "task": "transition_state_rate_calculation",
            }
            if isinstance(reaction, dict) and "cli_args" in reaction:
                payload["cli_args"] = list(reaction["cli_args"])
            self._extract_cli_args(payload)
            payloads.append(payload)

        for payload in payloads:
            await self.swarm_queue.put(payload)

        for payload in payloads:
            workers.append(asyncio.create_task(self._evaluate_payload(payload)))

        return payloads, workers

    async def dispatch_tp_grid(
        self,
        T_grid: list[float],
        P_grid: list[float],
        base_cli_args: Sequence[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[asyncio.Task[str]]]:
        """Dispatch a 2D Temperature/Pressure grid using real `cochem_kinetic.cli` invocations."""
        self._validate_backend_available()
        if not base_cli_args:
            raise ValueError(
                "base_cli_args must be provided so each T/P grid point invokes a real "
                f"'{self.entrypoint_module}' subcommand."
            )

        payloads: list[dict[str, Any]] = []
        workers: list[asyncio.Task[str]] = []
        idx = 0
        for T in T_grid:
            for P in P_grid:
                payload = {
                    "id": idx,
                    "target": "/goal",
                    "task": "master_equation_grid_point",
                    "T": float(T),
                    "P": float(P),
                    "cli_args": [*map(str, base_cli_args), "--temp-k", str(float(T))],
                }
                payloads.append(payload)
                idx += 1

        for payload in payloads:
            await self.swarm_queue.put(payload)

        for payload in payloads:
            workers.append(asyncio.create_task(self._evaluate_payload(payload)))

        return payloads, workers
