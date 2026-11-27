from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import shlex
import shutil
import sys
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENTRYPOINT_MODULE = "cochem_kinetic.cli"


class KineticHPCDispatcher:
    """Asynchronous SLURM/local/WSL dispatcher backed by the real `cochem_kinetic.cli` entrypoint."""

    def __init__(
        self,
        batch_size: int = 1000,
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
        self.batch_size: int = batch_size
        self.execution_mode: str = mode
        self.entrypoint_module: str = entrypoint_module
        self.python_executable: str = python_executable or sys.executable
        self.wsl_python: str = wsl_python
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
                "Dispatch payload must supply non-empty 'cli_args' for real entrypoint "
                f"'{self.entrypoint_module}'."
            )
        if isinstance(raw_args, str) or not isinstance(raw_args, Sequence) or len(raw_args) == 0:
            raise ValueError("'cli_args' must be a non-empty sequence of CLI argument strings.")
        return [str(arg) for arg in raw_args]

    def build_command(self, payload: dict[str, Any]) -> list[str]:
        cli_args = self._extract_cli_args(payload)
        if self.execution_mode == "slurm":
            wrap_cmd = shlex.join([self.python_executable, "-m", self.entrypoint_module, *cli_args])
            return ["sbatch", "--parsable", "--wrap", wrap_cmd]
        if self.execution_mode == "wsl" and sys.platform == "win32":
            wsl_cmd = shlex.join([self.wsl_python, "-m", self.entrypoint_module, *cli_args])
            return ["wsl", "bash", "-lc", wsl_cmd]
        return [self.python_executable, "-m", self.entrypoint_module, *cli_args]

    async def _worker(self) -> None:
        while True:
            payload = await self.swarm_queue.get()
            try:
                self._validate_backend_available()
                cmd = self.build_command(payload)
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"{self.execution_mode} dispatch failed with exit code {process.returncode}: "
                        f"{stderr.decode(errors='replace').strip()}"
                    )
            finally:
                self.swarm_queue.task_done()

    async def dispatch_tp_grid(
        self,
        temperatures: Any,
        pressures: Any,
        base_cli_args: Sequence[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[asyncio.Task[None]]]:
        """Slice T/P coordinates into discrete CLI payloads and fail closed if backend is unavailable."""
        self._validate_backend_available()
        if not base_cli_args:
            raise ValueError(
                "base_cli_args must be provided so each T/P grid point invokes a real "
                f"'{self.entrypoint_module}' subcommand."
            )

        T, P = np.meshgrid(temperatures, pressures)
        t_flat = T.flatten()
        p_flat = P.flatten()

        payloads: list[dict[str, Any]] = []
        for t, p in zip(t_flat, p_flat):
            t_val = float(t)
            p_val = float(p)
            if not math.isfinite(t_val) or t_val <= 0.0:
                raise ValueError(f"Temperature must be finite and > 0 K, got {t!r}.")
            if not math.isfinite(p_val) or p_val <= 0.0:
                raise ValueError(f"Pressure must be finite and > 0 atm, got {p!r}.")
            payloads.append(
                {
                    "endpoint": "/goal",
                    "temperature": t_val,
                    "pressure": p_val,
                    "cli_args": [
                        *map(str, base_cli_args),
                        "--temp-k",
                        str(t_val),
                        "--pressure-atm",
                        str(p_val),
                    ],
                }
            )

        num_workers = min(len(payloads), self.batch_size)
        workers: list[asyncio.Task[None]] = [asyncio.create_task(self._worker()) for _ in range(num_workers)]

        for payload in payloads:
            self.swarm_queue.put_nowait(payload)

        return payloads, workers

    async def dispatch_reaction_network(
        self, reactions: list[Any]
    ) -> tuple[list[dict[str, Any]], list[asyncio.Task[None]]]:
        """Slice elementary reactions into discrete CLI payloads and fail closed if backend is unavailable."""
        self._validate_backend_available()
        payloads: list[dict[str, Any]] = []
        for i, r in enumerate(reactions):
            payload: dict[str, Any] = {"endpoint": "/goal", "reaction_id": i, "reaction_data": r}
            if isinstance(r, dict) and "cli_args" in r:
                payload["cli_args"] = list(r["cli_args"])
            self._extract_cli_args(payload)
            payloads.append(payload)

        num_workers = min(len(payloads), self.batch_size)
        workers: list[asyncio.Task[None]] = [asyncio.create_task(self._worker()) for _ in range(num_workers)]

        for payload in payloads:
            self.swarm_queue.put_nowait(payload)

        return payloads, workers
