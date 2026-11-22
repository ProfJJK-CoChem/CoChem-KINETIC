#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 2: Charge-Aware MLFF Pre-Optimizer Module
----------------------------------------------------------------
Interfaces ASE with charge-aware AIMNet2 (recommended for Q = -1, 0, +1 and
H/C/N/O/F/S systems) or MACE-OFF24m (neutral closed-shell only) with Float32
noise guards per §8A.2 / §16.1.

Strict Scientific Integrity Policy:
- Silent metallic EMT() fallback is PROHIBITED.
- Pairwise Lennard-Jones and covalent-radii energy fabrication are PROHIBITED.
- Neutral-only MACE-OFF is PROHIBITED for charged (charge != 0) or open-shell
  (multiplicity != 1) species.
- Missing MLFF/QM backends raise an explicit RuntimeError.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import ase
    from ase import Atoms
    from ase.optimize import BFGS

    ASE_AVAILABLE = True
except ImportError:
    ase = None
    Atoms = None
    BFGS = None
    ASE_AVAILABLE = False


class MACEPreOptimizer:
    """Pre-relaxes structures using AIMNet2 or MACE-OFF24m with Float32 noise guards."""

    FLOAT32_NOISE_FLOOR: float = 1e-5
    MIN_FMAX_FLOOR: float = 1e-3

    def __init__(
        self,
        model_name: str = "medium",
        backend: str = "auto",
        charge: int = 0,
        multiplicity: int = 1,
        calculator: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend.lower()
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.logger = logging.getLogger("CoChem_KINETIC_MLFF")
        self.backend_name: str | None = None
        if calculator is not None:
            self.calculator = calculator
            self.backend_name = type(calculator).__name__
        else:
            self.calculator = self._init_calculator()

    def _init_calculator(self) -> Any | None:
        """
        Attempts to initialize a genuine MLFF calculator (AIMNet2 or MACE-OFF).
        Never falls back silently to metallic EMT or empirical covalent radii.
        """
        if not ASE_AVAILABLE:
            self.logger.warning("ASE is not installed; no MLFF ASE calculator initialized.")
            return None

        # 1. Charge-aware AIMNet2 (supports Q = -1, 0, +1 and H,C,N,O,F,S)
        if self.backend in {"auto", "aimnet", "aimnet2"}:
            try:
                from aimnet.calculators import AIMNet2ASE

                aimnet_model = "aimnet2" if self.model_name in {"medium", "small", "large", "aimnet2"} else self.model_name
                calc = AIMNet2ASE(aimnet_model, charge=self.charge, mult=self.multiplicity)
                self.backend_name = f"aimnet2:{aimnet_model}"
                self.logger.info(
                    f"Initialized charge-aware AIMNet2 ({aimnet_model}, Q={self.charge}, M={self.multiplicity})."
                )
                return calc
            except (ImportError, RuntimeError, ValueError) as exc:
                if self.backend in {"aimnet", "aimnet2"}:
                    self.logger.warning(f"Requested AIMNet2 backend unavailable: {exc}")
                    return None
                self.logger.info(f"AIMNet2 unavailable ({exc}); checking MACE-OFF.")

        # 2. Neutral-only MACE-OFF (strictly blocked for charged or open-shell species)
        if self.backend in {"auto", "mace", "mace_off"}:
            if self.charge != 0 or self.multiplicity != 1:
                self.logger.warning(
                    f"Refusing neutral-only MACE-OFF for charged/open-shell system "
                    f"(Q={self.charge}, M={self.multiplicity}). Use AIMNet2 instead."
                )
                return None
            try:
                from mace.calculators import mace_off

                calc = mace_off(model=self.model_name, device="cpu")
                self.backend_name = f"mace_off:{self.model_name}"
                self.logger.info(f"Initialized MACE-OFF24 ({self.model_name}) calculator.")
                return calc
            except (ImportError, RuntimeError, ValueError) as exc:
                self.logger.warning(f"MACE-OFF initialization unavailable ({exc}).")
                return None

        return None

    @classmethod
    def guard_fmax(cls, fmax: float) -> float:
        """Guards requested fmax against single-precision Float32 noise floor (§8A.2)."""
        return float(max(fmax, cls.MIN_FMAX_FLOOR))

    def apply_float32_noise_guard(
        self,
        prev_grad: np.ndarray | None,
        curr_grad: np.ndarray,
        prev_energy: float | None,
        curr_energy: float,
    ) -> bool:
        """
        Enforces Float32 noise guard per §8A.2 / §16.1.
        Returns True if optimization should stop due to gradient sign-flip or energy noise floor.
        """
        if prev_energy is not None and abs(curr_energy - prev_energy) < self.FLOAT32_NOISE_FLOOR:
            self.logger.info(
                f"Float32 noise guard activated: energy change ({abs(curr_energy - prev_energy):.2e}) "
                f"below precision floor ({self.FLOAT32_NOISE_FLOOR:.1e})."
            )
            return True

        if prev_grad is not None and curr_grad is not None:
            norm_curr = float(np.linalg.norm(curr_grad))
            norm_prev = float(np.linalg.norm(prev_grad))
            if norm_curr < 1e-3 and norm_prev < 1e-3:
                dot_prod = float(np.sum(prev_grad * curr_grad))
                if dot_prod < -0.2 * norm_curr * norm_prev:
                    self.logger.info(
                        "Float32 noise guard activated: gradient sign-flip detected in low-force regime. "
                        "Optimization halted safely to prevent noise hunting."
                    )
                    return True

        return False

    def pre_relax_geometry(
        self,
        symbols: list[str],
        coords: np.ndarray,
        max_steps: int = 50,
        fmax: float = 0.05,
        charge: int | None = None,
        multiplicity: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Pre-relaxes Cartesian coordinates using a genuine MLFF/QM ASE calculator.
        Raises RuntimeError if no valid MLFF/QM backend is installed (never falls back
        to metallic EMT or fabricated covalent-radii energies).

        Returns:
            (relaxed_coords_angstrom, final_energy_ev)
        """
        eff_charge = self.charge if charge is None else int(charge)
        eff_mult = self.multiplicity if multiplicity is None else int(multiplicity)
        fmax_guarded = self.guard_fmax(fmax)

        if not ASE_AVAILABLE:
            raise RuntimeError(
                "ASE is required for geometry pre-relaxation. No valid MLFF/QM backend is available; "
                "silent EMT and geometric fallbacks are prohibited."
            )

        if self.calculator is None:
            raise RuntimeError(
                f"No valid MLFF/QM backend available (backend='{self.backend}', Q={eff_charge}, M={eff_mult}). "
                "Install charge-aware 'aimnet' (AIMNet2) or neutral 'mace-torch'; silent EMT and "
                "covalent-radii energy fallbacks are strictly prohibited."
            )

        if self.backend_name and self.backend_name.startswith("mace_off") and (eff_charge != 0 or eff_mult != 1):
            raise RuntimeError(
                f"Neutral-only MACE-OFF cannot relax charged/open-shell species (Q={eff_charge}, M={eff_mult}). "
                "Use charge-aware AIMNet2."
            )

        atoms = Atoms(symbols=symbols, positions=np.asarray(coords, dtype=np.float64))
        atoms.info["charge"] = eff_charge
        atoms.info["spin"] = eff_mult
        if hasattr(self.calculator, "set_charge"):
            self.calculator.set_charge(eff_charge)
        if hasattr(self.calculator, "set_mult"):
            self.calculator.set_mult(eff_mult)
        atoms.calc = self.calculator

        opt = BFGS(atoms, logfile=None)

        prev_grad: np.ndarray | None = None
        prev_energy: float | None = None

        for step in range(max_steps):
            forces = atoms.get_forces()
            grads = -forces
            curr_energy = float(atoms.get_potential_energy())
            fmax_curr = float(np.max(np.linalg.norm(forces, axis=1)))

            if fmax_curr <= fmax_guarded:
                self.logger.info(
                    f"Pre-optimization converged to target fmax ({fmax_curr:.4f} <= {fmax_guarded:.4f}) at step {step}."
                )
                break

            if self.apply_float32_noise_guard(prev_grad, grads, prev_energy, curr_energy):
                break

            opt.step()
            prev_grad = grads.copy()
            prev_energy = curr_energy

        relaxed_coords = atoms.get_positions()
        energy_ev = float(atoms.get_potential_energy())
        self.logger.info(f"MLFF Pre-Relaxation complete ({self.backend_name}). Energy: {energy_ev:.6f} eV")
        return relaxed_coords, energy_ev
