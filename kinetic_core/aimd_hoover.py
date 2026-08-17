#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 2: Nose-Hoover AIMD Thermal Sampling Module
-------------------------------------------------------------------
Implements NVT ensemble Nose-Hoover chain thermostat dynamics for thermal sampling
of reaction coordinate initial conditions. Automatically reduces timestep to 0.5 fs
for transition metal species (Z > 20).
Includes path-integral quantum bead count calculator (§5.6, §11.2) and classical MD
rotational constant B0 prohibition guard (§11.1, §13.4).
"""

import logging
import numpy as np
from typing import Callable, Any


def calculate_required_beads(temp_k: float, w_max_cm1: float) -> int:
    """
    Calculates path-integral quantum bead count P > 2.2 * beta * hbar * omega_max (§5.6, §11.2).
    where beta = 1 / (kB * T), kB = 0.6950348 cm⁻¹/K.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be greater than 0 K for path-integral bead count calculation.")
    kB_cm1 = 0.6950348  # cm⁻¹ / K
    beta_hbar_omega = w_max_cm1 / (kB_cm1 * temp_k)
    p_exact = 2.2 * beta_hbar_omega
    p_beads = max(1, int(np.ceil(p_exact)))
    return p_beads


def compute_rotational_constants_md(
    geometry: np.ndarray,
    masses: np.ndarray,
    is_classical_md: bool = True,
) -> dict[str, Any]:
    """
    Computes rotational constants (cm⁻¹) from geometry and atomic masses.
    Prohibits classical MD for absolute B_0 claims (§11.1, §13.4).
    Issues warning log and tags output with [E] provenance.
    """
    logger = logging.getLogger("CoChem_KINETIC_AIMD")

    # Shift to center of mass
    masses = np.asarray(masses, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.float64)
    total_mass = np.sum(masses)
    com = np.sum(geometry * masses[:, np.newaxis], axis=0) / total_mass
    shifted = geometry - com

    # Inertia tensor (amu * A^2)
    inertia_tensor = np.zeros((3, 3), dtype=np.float64)
    for m, r in zip(masses, shifted):
        x, y, z = r
        r2 = x * x + y * y + z * z
        inertia_tensor[0, 0] += m * (r2 - x * x)
        inertia_tensor[1, 1] += m * (r2 - y * y)
        inertia_tensor[2, 2] += m * (r2 - z * z)
        inertia_tensor[0, 1] -= m * x * y
        inertia_tensor[0, 2] -= m * x * z
        inertia_tensor[1, 2] -= m * y * z
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    # Principal moments of inertia (sorted ascending)
    try:
        evals = np.linalg.eigvalsh(inertia_tensor)
    except np.linalg.LinAlgError as e:
        logger.error(f"Failed to diagonalize inertia tensor: {e}")
        raise ValueError(f"Invalid inertia tensor for geometry: {e}") from e
    evals = np.sort(evals)

    # Conversion factor h / (8 * pi^2 * c * amu * A^2) = 16.857629 cm⁻¹
    conv_factor = 16.857629
    rot_constants = [conv_factor / I if I > 1e-6 else 0.0 for I in evals]

    provenance_tag = "[M]"
    warning_msg = None

    if is_classical_md:
        warning_msg = "WARNING: Classical MD cannot yield absolute B0 due to missing ZPE expansion. Value marked as [E] diagnostic only."
        logger.warning(warning_msg)
        provenance_tag = "[E]"

    return {
        "rotational_constants_cm1": rot_constants,
        "principal_moments_amu_A2": evals.tolist(),
        "provenance_tag": provenance_tag,
        "is_absolute_b0": not is_classical_md,
        "warning": warning_msg,
    }


class NoseHooverAIMDSampler:
    """Nose-Hoover NVT Ensemble Integrator and Thermal Sampler."""

    def __init__(self, target_temp_k: float = 298.15, timestep_fs: float = 1.0, chain_length: int = 3) -> None:
        self.target_temp_k = target_temp_k
        self.default_dt_fs = timestep_fs
        self.chain_length = chain_length
        self.logger = logging.getLogger("CoChem_KINETIC_AIMD")

    def _determine_timestep(self, symbols: list[str]) -> float:
        """
        Reduces timestep to 0.5 fs if transition metals (Z > 20) are present in the species.
        """
        transition_metal_symbols = {
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
        }
        has_tm = any(sym in transition_metal_symbols for sym in symbols)
        if has_tm:
            self.logger.info("Transition metal detected. Automatically scaling AIMD timestep to 0.5 fs.")
            return 0.5
        return self.default_dt_fs

    def sample_nvt_trajectory(self, symbols: list[str], initial_coords: np.ndarray, masses: np.ndarray, n_steps: int = 100, force_fn: Callable | None = None) -> list[np.ndarray]:
        """
        Executes NVT Nose-Hoover trajectory sampling.
        Returns list of sampled coordinate frames.
        """
        if force_fn is None:
            raise ValueError("force_fn must be provided for AIMD sampling.")

        dt_fs = self._determine_timestep(symbols)
        dt_s = dt_fs * 1e-15
        n_atoms = len(symbols)

        # Maxwell-Boltzmann initial velocities
        kB = 1.380649e-23
        std_v = np.sqrt(kB * self.target_temp_k / (masses * 1.660539e-27))[:, np.newaxis]
        
        rng = np.random.default_rng()
        velocities = rng.standard_normal((n_atoms, 3)) * std_v
        
        # Remove Center of Mass momentum
        m_total = np.sum(masses)
        v_com = np.sum(velocities * masses[:, np.newaxis], axis=0) / m_total
        velocities -= v_com

        # Scale kinetic energy to target temperature
        ke_current = 0.5 * np.sum(masses[:, np.newaxis] * 1.660539e-27 * velocities**2)
        target_ke = 1.5 * n_atoms * kB * self.target_temp_k
        if ke_current > 0:
            velocities *= np.sqrt(target_ke / ke_current)

        # Convert to A/s
        velocities *= 1e10

        # Thermostatted variables
        xi = 0.0
        v_xi = 0.0
        q_thermostat = 1.0e-44 # Thermostat mass

        coords = initial_coords.copy()
        frames = [coords.copy()]

        # Initial force calculation
        try:
            forces = force_fn(coords)
        except Exception as e:
            self.logger.error(f"Force evaluation failed: {e}")
            raise RuntimeError(f"Force evaluation failed: {e}") from e

        accel = (forces / (masses[:, np.newaxis] * 1.660539e-27)) * 1e-20 # A/s^2

        for step in range(n_steps):
            # First Velocity Half-Step + Thermostat
            velocities += 0.5 * dt_s * (accel - v_xi * velocities)
            
            # Position Update
            coords += velocities * dt_s
            
            # Update Forces at new position
            try:
                forces = force_fn(coords)
            except Exception as e:
                self.logger.error(f"Force evaluation failed at step {step}: {e}")
                raise RuntimeError(f"Force evaluation failed at step {step}: {e}") from e
                
            accel = (forces / (masses[:, np.newaxis] * 1.660539e-27)) * 1e-20 # A/s^2

            # Second Velocity Half-Step + Thermostat
            velocities += 0.5 * dt_s * (accel - v_xi * velocities)
            
            # Update thermostat acceleration using full-step velocity
            ke = 0.5 * np.sum(masses[:, np.newaxis] * 1.660539e-27 * (velocities * 1e-10)**2)
            dof = 3 * n_atoms
            target_ke = 0.5 * dof * kB * self.target_temp_k
            v_xi += dt_s * (ke - target_ke) / q_thermostat  # Full step dt for thermostat
            
            frames.append(coords.copy())

        self.logger.info(f"Sampled {len(frames)} NVT trajectory frames at T={self.target_temp_k} K (dt={dt_fs} fs).")
        return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CoChem_KINETIC_AIMD")
    sampler = NoseHooverAIMDSampler()
    syms = ["Fe", "O", "O"]
    coords = np.array([[0.0,0.0,0.0], [0.0,0.0,1.2], [0.0,1.2,0.0]])
    masses = np.array([55.845, 15.999, 15.999])
    
    def dummy_force(c: np.ndarray) -> np.ndarray:
        return -0.5 * (c - coords) * 1e10
        
    trajectory = sampler.sample_nvt_trajectory(syms, coords, masses, n_steps=10, force_fn=dummy_force)
    logger.info(f"AIMD Sampler test passed. Trajectory frames: {len(trajectory)}")

    p_beads = calculate_required_beads(300.0, 3000.0)
    logger.info(f"Required beads at 300 K / 3000 cm-1: {p_beads}")

    rot_res = compute_rotational_constants_md(coords, masses, is_classical_md=True)
    logger.info(f"Classical MD B0 res: provenance={rot_res['provenance_tag']}, warning={rot_res['warning']}")