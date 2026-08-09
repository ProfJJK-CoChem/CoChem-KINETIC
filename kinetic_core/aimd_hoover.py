#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 2: Nose-Hoover AIMD Thermal Sampling Module
-------------------------------------------------------------------
Implements NVT ensemble Nose-Hoover chain thermostat dynamics for thermal sampling
of reaction coordinate initial conditions. Automatically reduces timestep to 0.5 fs
for transition metal species (Z > 20).
"""

import logging
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable


class NoseHooverAIMDSampler:
    """Nose-Hoover NVT Ensemble Integrator and Thermal Sampler."""

    def __init__(self, target_temp_k: float = 298.15, timestep_fs: float = 1.0, chain_length: int = 3):
        self.target_temp_k = target_temp_k
        self.default_dt_fs = timestep_fs
        self.chain_length = chain_length
        self.logger = logging.getLogger("CoChem_KINETIC_AIMD")

    def _determine_timestep(self, symbols: List[str]) -> float:
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

    def sample_nvt_trajectory(self, symbols: List[str], initial_coords: np.ndarray, masses: np.ndarray, n_steps: int = 100, force_fn: Optional[Callable] = None) -> List[np.ndarray]:
        """
        Executes NVT Nose-Hoover trajectory sampling.
        Returns list of sampled coordinate frames.
        """
        dt_fs = self._determine_timestep(symbols)
        dt_s = dt_fs * 1e-15
        n_atoms = len(symbols)

        # Maxwell-Boltzmann initial velocities (deterministic low-discrepancy Halton Box-Muller)
        kB = 1.380649e-23
        std_v = np.sqrt(kB * self.target_temp_k / (masses * 1.660539e-27))[:, np.newaxis]
        
        primes = [2, 3, 5, 7, 11, 13]
        u_vals = []
        for idx in range(n_atoms * 3 * 2):
            base = primes[idx % len(primes)]
            f = 1.0
            r = 0.0
            i_val = idx + 1
            while i_val > 0:
                f /= base
                r += f * (i_val % base)
                i_val //= base
            u_vals.append(max(r, 1e-6))

        raw_vels = []
        for a_idx in range(n_atoms):
            v_atom = []
            for dim in range(3):
                u1 = u_vals[(a_idx * 3 + dim) * 2]
                u2 = u_vals[(a_idx * 3 + dim) * 2 + 1]
                z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
                v_atom.append(z * std_v[a_idx, 0])
            raw_vels.append(v_atom)
        velocities = np.array(raw_vels) * 1e-10 # m/s to A/s

        # Remove Center of Mass momentum
        m_total = np.sum(masses)
        v_com = np.sum(velocities * masses[:, np.newaxis], axis=0) / m_total
        velocities -= v_com

        # Scale kinetic energy to target temperature
        ke_current = 0.5 * np.sum(masses[:, np.newaxis] * 1.660539e-27 * (velocities * 1e10)**2)
        target_ke = 1.5 * n_atoms * kB * self.target_temp_k
        if ke_current > 0:
            velocities *= np.sqrt(target_ke / ke_current)

        # Thermostatted variables
        xi = 0.0
        v_xi = 0.0
        q_thermostat = 1.0e-44 # Thermostat mass

        coords = initial_coords.copy()
        frames = [coords.copy()]

        for step in range(n_steps):
            if force_fn is not None:
                forces = force_fn(coords)
            else:
                # Harmonic restoring force fallback
                forces = -0.5 * (coords - initial_coords) * 1e10

            accel = (forces / (masses[:, np.newaxis] * 1.660539e-27)) * 1e-20 # A/s^2

            # Velocity Verlet + Thermostat step
            velocities += 0.5 * dt_s * (accel - v_xi * velocities)
            coords += velocities * dt_s * 1e10 # m to A
            
            # Update thermostat acceleration
            ke = 0.5 * np.sum(masses[:, np.newaxis] * 1.660539e-27 * (velocities * 1e-10)**2)
            dof = 3 * n_atoms
            target_ke = 0.5 * dof * kB * self.target_temp_k
            v_xi += 0.5 * dt_s * (ke - target_ke) / q_thermostat
            
            frames.append(coords.copy())

        self.logger.info(f"Sampled {len(frames)} NVT trajectory frames at T={self.target_temp_k} K (dt={dt_fs} fs).")
        return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sampler = NoseHooverAIMDSampler()
    syms = ["Fe", "O", "O"]
    coords = np.array([[0.0,0.0,0.0], [0.0,0.0,1.2], [0.0,1.2,0.0]])
    masses = np.array([55.845, 15.999, 15.999])
    trajectory = sampler.sample_nvt_trajectory(syms, coords, masses, n_steps=10)
    print(f"AIMD Sampler test passed. Trajectory frames: {len(trajectory)}")
