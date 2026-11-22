"""
Microcanonical RRKM Theory Solver (Beyer-Swinehart Exact Direct Count)
----------------------------------------------------------------------
Implements microcanonical unimolecular rate constants:

    k(E) = sigma * N^ddagger(E - E_0) / (h * rho(E))

where:
- E is total active internal vibrational energy (cm^-1),
- E_0 is the critical threshold energy / barrier height (cm^-1),
- N^ddagger(E - E_0) is the transition-state sum of vibrational states up to
  excess energy E - E_0 (dimensionless),
- rho(E) is the reactant vibrational density of states at energy E (per cm^-1),
- h in wavenumber-seconds is 1 / c = 3.33564095e-11 s * cm^-1 (so 1 / (h * rho)
  = c / rho(E) has units of s^-1),
- sigma is the reaction-path degeneracy (dimensionless, default 1.0).
"""

from __future__ import annotations

import warnings

import numpy as np

SPEED_OF_LIGHT_CM_S: float = 2.99792458e10
H_CM1_S: float = 1.0 / SPEED_OF_LIGHT_CM_S  # 3.3356409519815204e-11 cm^-1 * s


class RRKMSolver:
    """Exact Beyer-Swinehart microcanonical RRKM rate calculator."""

    def __init__(
        self,
        frequencies: list[float] | None = None,
        ts_frequencies: list[float] | None = None,
        e0_cm1: float = 0.0,
        reaction_degeneracy: float = 1.0,
    ) -> None:
        self.frequencies: list[float] = [float(f) for f in frequencies] if frequencies is not None else []
        self.ts_frequencies: list[float] = [float(f) for f in ts_frequencies] if ts_frequencies is not None else []
        self.e0_cm1: float = float(e0_cm1)
        self.reaction_degeneracy: float = float(reaction_degeneracy)

        self.dE: float = 1.0
        self.grain_array: np.ndarray = np.array([], dtype=np.float64)
        self.rho_E: np.ndarray = np.array([], dtype=np.float64)
        self.sum_of_states_ts: np.ndarray = np.array([], dtype=np.float64)

    @staticmethod
    def _validate_frequencies(freqs: list[float], label: str) -> list[float]:
        if not freqs:
            raise ValueError(f"{label} vibrational frequencies must be non-empty.")
        clean = [float(f) for f in freqs]
        if any(f <= 0.0 or not np.isfinite(f) for f in clean):
            raise ValueError(f"All {label} vibrational frequencies must be finite and strictly positive (cm^-1).")
        return clean

    @staticmethod
    def _beyer_swinehart_state_counts(
        frequencies: list[float],
        max_energy: float,
        dE: float,
    ) -> np.ndarray:
        """
        Beyer-Swinehart direct count of vibrational states per grain of width dE.
        Returns integer-valued float64 array counts[i] for grain energy i * dE.
        """
        if dE <= 0.0:
            raise ValueError("Energy grain size dE must be strictly positive.")
        if max_energy < 0.0:
            return np.zeros(1, dtype=np.float64)

        n_grains = int(np.floor(max_energy / dE)) + 1
        counts = np.zeros(n_grains, dtype=np.float64)
        counts[0] = 1.0

        for freq in frequencies:
            freq_grains = int(round(freq / dE))
            if freq_grains <= 0:
                freq_grains = 1
            for i in range(freq_grains, n_grains):
                counts[i] += counts[i - freq_grains]

        return counts

    def _beyer_swinehart_exact_counting(self, max_energy: float, dE: float) -> np.ndarray:
        """
        Calculates the reactant vibrational density of states rho(E) in (cm^-1)^-1
        using the exact Beyer-Swinehart direct-count algorithm.
        """
        freqs = self._validate_frequencies(self.frequencies, "reactant")
        counts = self._beyer_swinehart_state_counts(freqs, max_energy, dE)
        return counts / float(dE)

    def _beyer_swinehart_sum_of_states(
        self,
        ts_frequencies: list[float],
        max_excess_energy: float,
        dE: float,
    ) -> np.ndarray:
        """
        Calculates the transition-state sum of vibrational states N^ddagger(E_excess)
        using the exact Beyer-Swinehart direct-count algorithm (cumulative states <= E_excess).
        """
        freqs = self._validate_frequencies(ts_frequencies, "transition-state")
        if dE <= 0.0:
            raise ValueError("Energy grain size dE must be strictly positive.")
        if max_excess_energy < 0.0:
            return np.zeros(1, dtype=np.float64)

        n_grains = int(np.floor(max_excess_energy / dE)) + 1
        sum_states = np.ones(n_grains, dtype=np.float64)
        for freq in freqs:
            freq_grains = int(round(freq / dE))
            if freq_grains <= 0:
                freq_grains = 1
            for i in range(freq_grains, n_grains):
                sum_states[i] += sum_states[i - freq_grains]
        return sum_states

    def compute_kE(
        self,
        energy: float,
        dE: float = 1.0,
        e0_cm1: float | None = None,
        ts_frequencies: list[float] | None = None,
        reaction_degeneracy: float | None = None,
    ) -> float:
        """
        Computes microcanonical RRKM rate constant k(E) in s^-1:
            k(E) = sigma * N^ddagger(E - E_0) / (h * rho(E))

        Args:
            energy: Total internal vibrational energy E (cm^-1).
            dE: Energy grain size (cm^-1); clamped to Nyquist bound min(10.0, min_freq / 2).
            e0_cm1: Critical threshold energy E_0 (cm^-1). Defaults to self.e0_cm1.
            ts_frequencies: Real harmonic frequencies of the TS (cm^-1, excluding imaginary mode).
                Defaults to self.ts_frequencies.
            reaction_degeneracy: Symmetry / reaction-path degeneracy sigma.

        Returns:
            Microcanonical unimolecular rate constant k(E) in s^-1 (0.0 if E < E_0).
        """
        reactant_freqs = self._validate_frequencies(self.frequencies, "reactant")
        eff_ts_freqs = self._validate_frequencies(
            self.ts_frequencies if ts_frequencies is None else ts_frequencies,
            "transition-state",
        )
        eff_e0 = self.e0_cm1 if e0_cm1 is None else float(e0_cm1)
        eff_sigma = self.reaction_degeneracy if reaction_degeneracy is None else float(reaction_degeneracy)

        if dE <= 0.0:
            raise ValueError("Energy grain size dE must be strictly positive.")
        if eff_e0 < 0.0:
            raise ValueError("Threshold energy e0_cm1 cannot be negative.")
        if eff_sigma <= 0.0:
            raise ValueError("Reaction path degeneracy must be strictly positive.")

        if energy < eff_e0:
            return 0.0

        min_freq = min(min(reactant_freqs), min(eff_ts_freqs))
        nyquist_limit = min_freq / 2.0
        strict_bound = min(10.0, nyquist_limit)

        if dE > strict_bound:
            warnings.warn(
                f"Requested dE ({dE} cm^-1) violates Nyquist integration limits for density of states. "
                f"Overriding to bounded limit {strict_bound} cm^-1 to prevent aliasing."
            )
            self.dE = float(strict_bound)
        else:
            self.dE = float(dE)

        counts_r = self._beyer_swinehart_state_counts(reactant_freqs, energy, self.dE)
        self.grain_array = np.arange(len(counts_r), dtype=np.float64) * self.dE
        self.rho_E = counts_r / self.dE

        excess_energy = float(energy - eff_e0)
        self.sum_of_states_ts = self._beyer_swinehart_sum_of_states(eff_ts_freqs, excess_energy, self.dE)

        n_ddagger = float(self.sum_of_states_ts[-1])
        rho_at_e = float(self.rho_E[-1])

        # For sparse low-energy discrete grids where the exact single grain may be empty,
        # average density of states over a narrow local bin window around E (+/- 5*dE)
        if rho_at_e <= 0.0:
            half_win = max(1, int(round(min(50.0, max(self.dE, 0.05 * energy)) / self.dE)))
            lo = max(0, len(self.rho_E) - 1 - half_win)
            rho_at_e = float(np.mean(self.rho_E[lo:]))

        if rho_at_e <= 0.0:
            return 0.0

        rate_s_inv = (eff_sigma * n_ddagger) / (H_CM1_S * rho_at_e)
        return float(rate_s_inv)
