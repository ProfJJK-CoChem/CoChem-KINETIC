#!/usr/bin/env python3
import hashlib
"""
CoChem-KINETIC - Stage 1: Payload Ingestion Dispatcher & HDF5 State Writer
----------------------------------------------------------------------------
Ingests SMILES strings, reactant/product structures, and BASE state payloads.
Enforces v4 10-tier wall-clock budgets (T1-10s..T4-1mo), State-Chaining Rules D1-D5,
and PESStore HDF5 state writing per §8B / §8C.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False

try:
    from kinetic_core.cochem_pes_store import PESStore
except ImportError:
    from cochem_pes_store import PESStore


# v4 Wall-Clock Budgets (§8B)
V4_WALLCLOCK_BUDGETS: dict[str, float] = {
    "T1-10s": 10.0,
    "T1-1min": 60.0,
    "T1-30min": 1800.0,
    "T2-1h": 3600.0,
    "T2-3h": 10800.0,
    "T2-12h": 43200.0,
    "T3-1d": 86400.0,
    "T3-3d": 259200.0,
    "T4-1w": 604800.0,
    "T4-1mo": 2592000.0,
}

# Tier hierarchy ranking for D5 safe overwrite checks
TIER_HIERARCHY: dict[str, int] = {
    "T1-10s": 1,
    "T1-1min": 2,
    "T1-30min": 3,
    "T2-1h": 4,
    "T2-3h": 5,
    "T2-12h": 6,
    "T3-1d": 7,
    "T3-3d": 8,
    "T4-1w": 9,
    "T4-1mo": 10,
}


class KineticDispatcher:
    """CLI and API Dispatcher for CoChem-KINETIC workflows with v4 State-Chaining & PESStore support."""

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self.workspace_dir = Path(workspace_dir or os.environ.get("COCHEM_ARTIFACT_DIR", "."))
        self.logger = logging.getLogger("CoChem_KINETIC_Dispatcher")

    def get_tier_budget(self, tier_name: str) -> float:
        """Returns the wall-clock budget in seconds for a given v4 tier (§8B)."""
        if tier_name not in V4_WALLCLOCK_BUDGETS:
            raise ValueError(f"Unknown v4 wall-clock tier '{tier_name}'. Valid tiers: {list(V4_WALLCLOCK_BUDGETS.keys())}")
        return V4_WALLCLOCK_BUDGETS[tier_name]

    # --- State-Chaining Rules D1-D5 (§8B) ---

    def check_d1_stationary_point(self, gradient: np.ndarray, tol: float = 1e-4) -> bool:
        """Rule D1: Stationary point check (gradient norm < tol)."""
        if gradient is None or len(gradient) == 0:
            return False
        grad_norm = float(np.max(np.linalg.norm(gradient, axis=-1)))
        passed = grad_norm < tol
        self.logger.debug(f"D1 Stationary Point Check: max grad norm = {grad_norm:.6e} (tol = {tol:.6e}) -> Passed: {passed}")
        return passed

    def check_d2_hessian_mode_match(
        self,
        freqs_prev: np.ndarray,
        freqs_curr: np.ndarray,
        is_ts: bool = False,
        tol: float = 50.0,
    ) -> bool:
        """Rule D2: Hessian mode match (imaginary frequency count & mode frequencies match within tol cm⁻¹)."""
        freqs_prev = np.asarray(freqs_prev)
        freqs_curr = np.asarray(freqs_curr)

        if len(freqs_prev) == 0 or len(freqs_curr) == 0:
            return False

        imag_prev = int(np.sum(freqs_prev < 0.0))
        imag_curr = int(np.sum(freqs_curr < 0.0))

        if is_ts:
            if imag_prev != 1 or imag_curr != 1:
                self.logger.warning(f"D2 Failed: Transition state requires exactly 1 imaginary frequency (prev={imag_prev}, curr={imag_curr}).")
                return False
        else:
            if imag_prev != 0 or imag_curr != 0:
                self.logger.warning(f"D2 Failed: Minimum requires 0 imaginary frequencies (prev={imag_prev}, curr={imag_curr}).")
                return False

        # Check frequency difference for lowest mode
        freq_diff = abs(float(freqs_prev[0]) - float(freqs_curr[0]))
        passed = freq_diff <= tol
        self.logger.debug(f"D2 Hessian Mode Match: diff = {freq_diff:.2f} cm⁻¹ (tol = {tol:.2f}) -> Passed: {passed}")
        return passed

    def check_d3_scf_basin(self, density_matrix_prev: np.ndarray, density_matrix_curr: np.ndarray, tol: float = 1e-3) -> bool:
        """Rule D3: SCF density matrix basin check (Frobenius norm distance < tol)."""
        d_prev = np.asarray(density_matrix_prev)
        d_curr = np.asarray(density_matrix_curr)
        if d_prev.shape != d_curr.shape:
            return False
        dist = float(np.linalg.norm(d_prev - d_curr))
        passed = dist < tol
        self.logger.debug(f"D3 SCF Basin Check: density diff norm = {dist:.6e} (tol = {tol:.6e}) -> Passed: {passed}")
        return passed

    def check_d4_cp_ghost_match(self, ghost_atoms_prev: list[int] | None, ghost_atoms_curr: list[int] | None) -> bool:
        """Rule D4: Counterpoise ghost atom match (exact index match)."""
        passed = sorted(ghost_atoms_prev or []) == sorted(ghost_atoms_curr or [])
        self.logger.debug(f"D4 Ghost Atom Match: prev={ghost_atoms_prev}, curr={ghost_atoms_curr} -> Passed: {passed}")
        return passed

    def check_d5_safe_overwrite(self, meta_prev: dict[str, Any], meta_curr: dict[str, Any]) -> bool:
        """Rule D5: Safe overwrite rule (higher or equal tier can update state; unverified lower tier cannot)."""
        tier_prev = meta_prev.get("wallclock_tier", "T1-10s")
        tier_curr = meta_curr.get("wallclock_tier", "T1-10s")
        rank_prev = TIER_HIERARCHY.get(tier_prev, 0)
        rank_curr = TIER_HIERARCHY.get(tier_curr, 0)

        if rank_curr >= rank_prev:
            self.logger.debug(f"D5 Safe Overwrite Allowed: curr tier '{tier_curr}' (rank {rank_curr}) >= prev tier '{tier_prev}' (rank {rank_prev}).")
            return True
        else:
            self.logger.warning(f"D5 Safe Overwrite Rejected: curr tier '{tier_curr}' (rank {rank_curr}) < prev tier '{tier_prev}' (rank {rank_prev}).")
            return False

    def parse_payload_json(self, json_path: Path) -> dict[str, Any]:
        """Parses input payload JSON file specifying reactants, products, and TS candidates."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        self.logger.info(f"Ingested kinetic payload from {json_path.name}")
        return data

    def write_state_hdf5(
        self,
        h5_path: Path,
        thermo_data: dict[str, float],
        irc_coords: np.ndarray | None = None,
        irc_energies: np.ndarray | None = None,
        tier_name: str = "T1-30min",
        pes_store: PESStore | None = None,
        chaining_meta: dict[str, Any] | None = None,
    ):
        """
        Serializes rate parameters, Eyring thermo data, and IRC trajectory into h5_path.
        Integrates with PESStore and embeds v4 tier budgets and D1-D5 metadata (§8B / §8C).
        """
        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        budget_seconds = self.get_tier_budget(tier_name)

        if not H5PY_AVAILABLE:
            self.logger.warning("h5py not installed. Dumping state telemetry to JSON fallback.")
            json_fallback = h5_path.with_suffix(".json")
            with open(json_fallback, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "thermo": thermo_data,
                        "tier_name": tier_name,
                        "budget_seconds": budget_seconds,
                        "irc_energies": irc_energies.tolist() if irc_energies is not None else [],
                    },
                    f,
                    indent=4,
                )
            return

        with h5py.File(h5_path, "a") as f:
            # Group /kinetic/thermo
            thermo_group = f.require_group("kinetic/thermo")
            for k, v in thermo_data.items():
                if k in thermo_group:
                    del thermo_group[k]
                thermo_group.create_dataset(k, data=float(v))
            thermo_group.attrs["unit"] = "kcal/mol"
            thermo_group.attrs["temperature_k"] = thermo_data.get("temperature_k", 298.15)
            thermo_group.attrs["wallclock_tier"] = tier_name
            thermo_group.attrs["budget_seconds"] = budget_seconds

            if chaining_meta:
                for k, v in chaining_meta.items():
                    thermo_group.attrs[k] = str(v)

            # Group /kinetic/irc_path
            if irc_coords is not None or irc_energies is not None:
                irc_group = f.require_group("kinetic/irc_path")
                if irc_coords is not None:
                    if "coordinates" in irc_group:
                        del irc_group["coordinates"]
                    irc_group.create_dataset("coordinates", data=irc_coords)
                if irc_energies is not None:
                    if "energies" in irc_group:
                        del irc_group["energies"]
                    irc_group.create_dataset("energies", data=irc_energies)

        # Delegate PES grid/fit/uncertainty writing to PESStore if present
        if pes_store is not None or h5_path.name.endswith(".h5"):
            try:
                store = pes_store or PESStore(h5_path)
                if irc_coords is not None and irc_energies is not None:
                    store.append_batch(irc_coords, irc_energies, group="grid", metadata={"wallclock_tier": tier_name})
            except (OSError, ValueError, TypeError, RuntimeError) as e:
                self.logger.warning(f"Could not append state data to PESStore: {e}")

        self.logger.info(f"Successfully serialized kinetic state to {h5_path.name} [Tier: {tier_name}, Budget: {budget_seconds}s].")




def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()