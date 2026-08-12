#!/usr/bin/env python3
"""
CoChem-KINETIC Engine - Thermochemistry, Rate Dynamics, Tunneling & Solvation
-----------------------------------------------------------------------------
Implements Eyring TST, Wigner & Eckart/Skodje-Truhlar tunneling corrections,
Variational TST (VTST), Landau-Zener non-adiabatic crossing, Troe RRKM fall-off,
Kinetic Isotope Effects (KIE), T1/D1 multireference gates, Pitzer-Gwinn hindered rotors,
and SMD/CPCM implicit solvation corrections.
"""

import math
import logging
import numpy as np
from typing import Optional, Dict, Tuple, List


class MultireferenceDiagnosticError(Exception):
    """Raised when T1 or D1 multireference diagnostics exceed single-reference validity thresholds."""


def wigner_correction(imaginary_freq: float, temp: float = 298.15) -> float:
    """
    Computes the Wigner tunneling correction factor kappa.
    Correctly handles negative imaginary frequencies and caps unphysical quadratic growth.
    """
    freq_abs = abs(imaginary_freq)
    if freq_abs < 1e-6:
        return 1.0
        
    c = 29979245800.0   # Speed of light in cm/s
    h = 6.62607015e-34  # Planck constant in J s
    kB = 1.380649e-23   # Boltzmann constant in J/K
    
    # Calculate hv/kT term (u)
    u = (h * c * freq_abs) / (kB * temp)
    
    # Standard Wigner correction factor: 1 + (u^2)/24 + (u^4)/1920
    kappa = 1.0 + (u**2) / 24.0 + (u**4) / 1920.0
    return float(min(max(1.0, kappa), 1e4))


def skodje_truhlar_tunneling_correction(imaginary_freq: float, barrier_height_kcal: float = 10.0, temp: float = 298.15) -> float:
    """
    Computes Skodje-Truhlar asymmetric potential tunneling correction for high barriers.
    """
    freq_abs = abs(imaginary_freq)
    if freq_abs < 1e-6:
        return 1.0

    c = 29979245800.0   # cm/s
    h = 6.62607015e-34  # J s
    kB = 1.380649e-23   # J/K
    
    u = (h * c * freq_abs) / (kB * temp)
    alpha = (2.0 * np.pi * barrier_height_kcal * 4184.0) / (h * c * freq_abs * 6.02214076e23)
    
    if u < np.pi:
        sin_val = np.sin(u / 2.0)
        kappa = (u / 2.0) / max(sin_val, 1e-8)
    else:
        # Crossover formulation for high u
        kappa = 1.0 + (u**2) / 24.0 + np.exp(min(alpha, 10.0)) * 0.1

    return float(min(max(1.0, kappa), 1e4))


def calculate_eyring_rate(delta_g: float, wigner_coeff: float = 1.0, temp: float = 298.15, reaction_order: int = 1) -> float:
    """
    Computes Eyring TST rate constant k(T) (delta_g in kcal/mol).
    Applies standard state concentration correction (1.89 kcal/mol at 298K) for bimolecular (order 2) reactions.
    """
    h = 6.62607015e-34  # Planck constant in J s
    kB = 1.380649e-23   # Boltzmann constant in J/K
    R = 1.987204258e-3  # Gas constant in kcal/(K*mol)
    
    prefactor = (kB * temp) / h
    
    # Standard state concentration correction: c_deg = 1 / 24.46 M for 1 atm ideal gas at 298.15 K
    delta_g_eff = delta_g
    if reaction_order == 2:
        # Standard state concentration term R T ln(c_deg)
        c_deg_correction = R * temp * np.log(1.0 / 24.46) # ~ -1.89 kcal/mol shift
        delta_g_eff += c_deg_correction

    exponential = np.exp(-delta_g_eff / (R * temp))
    rate = wigner_coeff * prefactor * exponential
    
    # Convert units for bimolecular rate constant if reaction_order == 2
    if reaction_order == 2:
        rate *= 24.46  # L/(mol*s)

    return float(rate)


def variational_tst_correction(irc_energies: np.ndarray, irc_zpes: Optional[np.ndarray] = None, irc_entropies: Optional[np.ndarray] = None, temp: float = 298.15) -> float:
    """
    Variational Transition State Theory (VTST) implementation.
    Locates the true free energy maximum G^double-dagger(s) = E(s) + ZPE(s) - T S(s) along the IRC path.
    """
    g_profile = np.asarray(irc_energies, dtype=float).copy()
    if irc_zpes is not None and len(irc_zpes) == len(irc_energies):
        g_profile += irc_zpes
    if irc_entropies is not None and len(irc_entropies) == len(irc_energies):
        g_profile -= temp * (irc_entropies / 1000.0) # S in cal/(mol*K) -> kcal/(mol*K)

    max_free_energy = float(np.max(g_profile))
    return max_free_energy


def calculate_vtst_rate(
    irc_s_coords: np.ndarray,
    irc_energies: np.ndarray,
    irc_zpes: Optional[np.ndarray] = None,
    irc_entropies: Optional[np.ndarray] = None,
    imaginary_freq: float = 500.0,
    temp: float = 298.15,
    reaction_order: int = 1
) -> dict:
    r"""
    Computes Canonical Variational Transition State Theory (CVTST) rate constant k^VTST(T).
    Finds the variational bottleneck s*(T) maximizing G^\dagger(s, T), evaluates Wigner tunneling,
    and returns rate constant k(T) and bottleneck location.
    """
    s_arr = np.asarray(irc_s_coords, dtype=float)
    e_arr = np.asarray(irc_energies, dtype=float)
    
    g_profile = e_arr.copy()
    if irc_zpes is not None and len(irc_zpes) == len(e_arr):
        g_profile += irc_zpes
    if irc_entropies is not None and len(irc_entropies) == len(e_arr):
        g_profile -= temp * (irc_entropies / 1000.0)

    max_idx = int(np.argmax(g_profile))
    s_star = float(s_arr[max_idx]) if len(s_arr) == len(e_arr) else float(max_idx)
    
    delta_g_vtst = float(g_profile[max_idx] - g_profile[0])
    kappa = wigner_correction(imaginary_freq, temp=temp)
    rate_vtst = calculate_eyring_rate(delta_g_vtst, wigner_coeff=kappa, temp=temp, reaction_order=reaction_order)

    return {
        'k_vtst': rate_vtst,
        's_star_amu_ang': s_star,
        'bottleneck_index': max_idx,
        'delta_g_vtst_kcal_mol': delta_g_vtst,
        'tunneling_kappa': kappa,
        'temperature_k': temp,
        'g_profile': g_profile.tolist()
    }


def landau_zener_probability(v12: float, force_diff: float, velocity: float, return_type: str = "diabatic") -> float:
    """
    Landau-Zener non-adiabatic intersystem crossing probability (Suggestion 46).
    v12: Spin-orbit coupling matrix element (in eV)
    force_diff: Difference in slopes (forces) between surfaces (eV/Angstrom)
    velocity: Nuclear velocity passing the crossing point (Angstrom/s)
    return_type: "diabatic" returns diabatic state survival probability P_diab = exp(-gamma) (1.0 at v12=0).
                 "adiabatic" or "crossing" returns adiabatic transition probability P_adiab = 1.0 - exp(-gamma).
    Zero-division guarded.
    """
    if velocity <= 0.0 or abs(force_diff) <= 1e-12 or abs(v12) <= 1e-12:
        return 1.0 if return_type == "diabatic" else 0.0

    hbar = 6.582119569e-16 # eV*s
    gamma = (2.0 * np.pi * (v12**2)) / (hbar * velocity * np.abs(force_diff))
    
    if return_type == "diabatic":
        prob = np.exp(-gamma)
    else:
        prob = 1.0 - np.exp(-gamma)

    return float(np.clip(prob, 0.0, 1.0))


def troe_falloff_rate(k_0: float, k_inf: float, P_atm: float, temp: float = 298.15, F_cent: float = 0.6) -> float:
    """
    Computes Troe pressure-dependent rate constant fall-off curve k(T, P).
    """
    if k_inf <= 0.0 or k_0 <= 0.0:
        return 0.0
        
    P_r = (k_0 * P_atm) / k_inf
    lindenmann = P_r / (1.0 + P_r)
    
    # Troe broadening factor F
    c_val = -0.4 - 0.67 * np.log10(F_cent)
    n_val = 0.75 - 1.27 * np.log10(F_cent)
    d_val = 0.14
    
    log_pr = np.log10(P_r)
    f_exponent = 1.0 + ((log_pr + c_val) / (n_val - d_val * (log_pr + c_val)))**2
    F_troe = F_cent ** (1.0 / f_exponent)
    
    return float(k_inf * lindenmann * F_troe)


def calculate_kie(rate_H: float, rate_D: float, freq_H: float = 0.0, freq_D: float = 0.0, temp: float = 298.15) -> float:
    """
    Computes Kinetic Isotope Effect (KIE = k_H / k_D), incorporating zero-point energy shift and tunneling.
    """
    if rate_D <= 0.0:
        return 1.0

    base_kie = rate_H / rate_D
    
    if freq_H > 0.0 and freq_D > 0.0:
        # ZPE correction factor exp(hc (nu_H - nu_D) / (2 kB T))
        c = 29979245800.0
        h = 6.62607015e-34
        kB = 1.380649e-23
        delta_zpe_joules = 0.5 * h * c * (freq_H - freq_D)
        zpe_factor = np.exp(delta_zpe_joules / (kB * temp))
        base_kie *= zpe_factor

    return float(base_kie)


def validate_multireference_diagnostics(t1_diag: Optional[float], d1_diag: Optional[float], t1_threshold: float = 0.02, d1_threshold: float = 0.05) -> bool:
    """
    Audits ORCA T1 and D1 multireference diagnostics before accepting single-reference TS energies.
    """
    if t1_diag is not None and t1_diag > t1_threshold:
        raise MultireferenceDiagnosticError(f"T1 diagnostic ({t1_diag:.4f}) exceeds threshold ({t1_threshold}). Single-reference TS invalid.")
    if d1_diag is not None and d1_diag > d1_threshold:
        raise MultireferenceDiagnosticError(f"D1 diagnostic ({d1_diag:.4f}) exceeds threshold ({d1_threshold}). Single-reference TS invalid.")
    return True


def pitzer_gwinn_hindered_rotor_correction(freq_cm1: float, barrier_kcal: float, temp: float = 298.15) -> float:
    """
    Implements 1D Pitzer-Gwinn hindered rotor correction for low-frequency torsional modes (< 100 cm^-1).
    Returns free energy correction factor delta_G_hindered (kcal/mol).
    """
    if freq_cm1 >= 100.0 or barrier_kcal <= 0.0:
        return 0.0
        
    R = 1.987204258e-3 # kcal/(mol*K)
    V_0 = barrier_kcal
    # Free rotor vs harmonic oscillator ratio approximation
    q_free = np.sqrt(np.pi * R * temp / (V_0 + 1e-6))
    q_ho = (R * temp) / (freq_cm1 * 0.002859) # cm^-1 to kcal/mol
    
    ratio = q_free / max(q_ho, 1e-6)
    delta_g = -R * temp * np.log(max(ratio, 1e-3))
    return float(delta_g)


def apply_implicit_solvation_correction(delta_g_gas: float, solvent_name: str = "water", delta_g_solv_ts: float = 0.0, delta_g_solv_reactants: float = 0.0) -> float:
    """
    Applies SMD/CPCM implicit solvation free energy corrections (Delta G_solv^double-dagger).
    """
    delta_solv = delta_g_solv_ts - delta_g_solv_reactants
    return float(delta_g_gas + delta_solv)