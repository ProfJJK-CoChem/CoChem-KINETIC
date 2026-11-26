"""
CoChem-KINETIC Reaction-Profile Engine (`cochem-kinetic rxn-profile`)
---------------------------------------------------------------------
End-to-end charge-aware reaction-profile engine supporting both:
  - 5-stationary-point stepwise pathways (R -> TS1 -> Int_tet -> TS2 -> P), with
    either an explicit Int_tet.xyz or an auto-discovered midpoint intermediate.
  - 3-stationary-point elementary pathways (R -> TS -> P) from 2-endpoint student
    folders containing only reactant.xyz + product.xyz (or R.xyz + P.xyz).

Stages:
  1. Strict stoichiometry, atom-ordering, charge, spin, and clash validation.
  2. Charge-aware AIMNet2 (`aimnet`, Q = -1, 0, +1; H,C,N,O,F,S) geometry optimization,
     midpoint intermediate discovery (for 2-endpoint inputs), 11-image ASE Climbing-Image
     NEB (IDPP interpolation), and saddle-point polish (Sella internal-coordinate
     optimizer if installed, or Eckart-projected Partitioned Rational Function
     Optimization [P-RFO] eigenvector-following).
  3. Central-difference mass-weighted Eckart-projected Hessian analysis verifying
     N_imag == 0 on minima and N_imag == 1 on transition states, plus RRHO and
     Grimme qRRHO thermochemistry.
  4. Bidirectional mass-weighted IRC connectivity verification (IRCTracerEngine)
     confirming < 0.15 A heavy-atom RMSD and identical covalent bond graphs.
  5. PySCF / GPU4PySCF DF-wB97X-D3(0)/def2-TZVPP (or def2-SVP) + PCM/ddCOSMO single-point
     energies, wall-clock timing telemetry, CSV/Markdown tables, and SHA-256 manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from cochem_kinetic.chem312_specs import (
    ELEMENTARY_STATIONARY_STATES,
    STATIONARY_STATES,
    formula_from_symbols,
    masses_from_symbols,
    read_xyz,
    resolve_solvent_eps,
    validate_stationary_stoichiometry,
    write_multi_xyz,
    write_xyz,
)
from cochem_kinetic.math.thermodynamics import grimes_qrrho_entropy
from kinetic_core.irc_tracer import IRCTracerEngine, heavy_atom_rmsd, kabsch_rmsd
from kinetic_core.thermo import apply_implicit_solvation_correction

logger = logging.getLogger("CoChem_KINETIC_RxnProfile")

# Exact CODATA 2018 physical constants
HARTREE_TO_KCAL: float = 627.5094740631
HARTREE_TO_EV: float = 27.211386245988
EV_TO_KCAL: float = 23.06054783061903
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV
KCAL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCAL
ANG_TO_M: float = 1.0e-10
AMU_TO_KG: float = 1.66053906660e-27
EV_TO_J: float = 1.602176634e-19
PLANCK_H: float = 6.62607015e-34
KB_J_K: float = 1.380649e-23
AVOGADRO: float = 6.02214076e23
SPEED_OF_LIGHT_CM_S: float = 2.99792458e10
R_KCAL_MOL_K: float = 1.98720425864083e-3
R_J_MOL_K: float = 8.31446261815324
CM1_TO_KCAL: float = (PLANCK_H * SPEED_OF_LIGHT_CM_S * AVOGADRO) / 4184.0
CM1_TO_HARTREE: float = CM1_TO_KCAL / HARTREE_TO_KCAL

# Eigenvalue conversion from eV / (amu * A^2) to (cm^-1)^2:
# omega^2 (s^-2) = lambda * (eV->J) / (amu->kg * A^2->m^2)
# nu_cm1 = sqrt(omega^2) / (2 * pi * c_cm_s)
EIG_EV_AMU_A2_TO_CM1_SQ: float = (
    EV_TO_J / (AMU_TO_KG * (ANG_TO_M**2))
) / ((2.0 * math.pi * SPEED_OF_LIGHT_CM_S) ** 2)


def sha256_file(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


STATE_FILENAME_ALIASES: dict[str, tuple[str, ...]] = {
    "R": ("R.xyz", "reactant.xyz", "Reactant.xyz"),
    "P": ("P.xyz", "product.xyz", "Product.xyz"),
    "Int_tet": ("Int_tet.xyz", "intermediate.xyz", "int_tet.xyz"),
    "TS1": ("TS1.xyz",),
    "TS2": ("TS2.xyz",),
    "TS": ("TS.xyz", "ts.xyz"),
}

ALL_CANDIDATE_STATES: tuple[str, ...] = ("R", "TS1", "Int_tet", "TS2", "TS", "P")


def resolve_state_xyz_path(input_dir: str | Path, state: str) -> Path | None:
    """Resolves a canonical state name (e.g. 'R', 'P', 'Int_tet') to an existing XYZ file."""
    base = Path(input_dir)
    candidates = STATE_FILENAME_ALIASES.get(state, (f"{state}.xyz",))
    for fname in candidates:
        p = base / fname
        if p.is_file():
            return p
    return None


def load_reaction_inputs(
    input_dir: str | Path,
    required_states: tuple[str, ...] = ("R", "P"),
    candidate_states: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Loads reaction XYZ geometries from `input_dir`, accepting both canonical names
    (`R.xyz`, `P.xyz`, `Int_tet.xyz`, `TS1.xyz`, `TS2.xyz`, `TS.xyz`) and student
    2-endpoint aliases (`reactant.xyz`, `product.xyz`).
    """
    base = Path(input_dir)
    loaded: dict[str, dict[str, Any]] = {}
    search_order: list[str] = list(
        candidate_states if candidate_states is not None else ALL_CANDIDATE_STATES
    )
    for st in required_states:
        if st not in search_order:
            search_order.append(st)

    for st in search_order:
        p = resolve_state_xyz_path(base, st)
        if p is not None:
            syms, coords, comment = read_xyz(p)
            loaded[st] = {
                "symbols": syms,
                "coords": coords,
                "comment": comment,
                "source": str(p),
                "source_filename": p.name,
            }
        elif st in required_states:
            aliases = STATE_FILENAME_ALIASES.get(st, (f"{st}.xyz",))
            alias_str = " or ".join(repr(a) for a in aliases)
            raise FileNotFoundError(
                f"Required input XYZ for state '{st}' ({alias_str}) not found in {base}."
            )
    return loaded


def detect_input_mode(loaded: dict[str, Any]) -> str:
    """
    Classifies loaded input geometries:
      - '5pt_explicit' when 'Int_tet' is present alongside endpoints.
      - '2pt_auto' when 'R' and 'P' are present without 'Int_tet'.
      - 'custom' otherwise (e.g. single monomer).
    """
    if "Int_tet" in loaded:
        return "5pt_explicit"
    if "R" in loaded and "P" in loaded:
        return "2pt_auto"
    return "custom"


def kabsch_align(coords: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
    """Rigidly rotates and translates `coords` onto `ref_coords` via proper Kabsch alignment."""
    a0 = np.asarray(ref_coords, dtype=np.float64)
    b0 = np.asarray(coords, dtype=np.float64)
    a_mean = np.mean(a0, axis=0)
    b_mean = np.mean(b0, axis=0)
    cov = (b0 - b_mean).T @ (a0 - a_mean)
    u_mat, _, vt_mat = np.linalg.svd(cov)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0.0:
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T
    return (b0 - b_mean) @ rot.T + a_mean


def _min_interatomic_distance(coords: np.ndarray) -> float:
    """Returns minimum pairwise interatomic distance in Angstrom."""
    c = np.asarray(coords, dtype=np.float64)
    n = len(c)
    if n < 2:
        return float("inf")
    min_d = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(c[i] - c[j]))
            if d < min_d:
                min_d = d
    return min_d


# =============================================================================
# 1. Eckart Projection, Harmonic Frequencies & RRHO / Grimme qRRHO Thermo
# =============================================================================


def build_eckart_projector(coords: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Constructs orthogonal projector P (3N x 3N) in mass-weighted coordinates that
    removes rigid-body translation (3 modes) and rotation (3 modes, or 2 if linear).

    Returns:
        (projector_matrix, n_tr_modes)
    """
    c = np.asarray(coords, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    n_atoms = len(m)
    sqrt_m = np.sqrt(m)
    com = np.sum(c * m[:, np.newaxis], axis=0) / np.sum(m)
    r = c - com[np.newaxis, :]

    basis_vecs: list[np.ndarray] = []
    # 3 translational vectors in mass-weighted coordinates
    for axis in range(3):
        v = np.zeros((n_atoms, 3), dtype=np.float64)
        v[:, axis] = sqrt_m
        basis_vecs.append(v.reshape(-1))

    # 3 rotational vectors in mass-weighted coordinates: sqrt(m_i) * (e_axis x r_i)
    eye3 = np.eye(3, dtype=np.float64)
    for axis in range(3):
        cross = np.cross(eye3[axis][np.newaxis, :], r)
        v = cross * sqrt_m[:, np.newaxis]
        vec = v.reshape(-1)
        if np.linalg.norm(vec) > 1e-8:
            basis_vecs.append(vec)

    # Orthonormalize with Gram-Schmidt / SVD
    mat = np.column_stack(basis_vecs)
    u_mat, s_vals, _ = np.linalg.svd(mat, full_matrices=False)
    keep = s_vals > 1e-6 * np.max(s_vals)
    q_tr = u_mat[:, keep]
    n_tr = int(q_tr.shape[1])
    projector = np.eye(3 * n_atoms, dtype=np.float64) - q_tr @ q_tr.T
    return projector, n_tr


def diagonalize_hessian_eckart(
    hessian_ev_a2: np.ndarray,
    coords: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects translational and rotational degrees of freedom from a Cartesian
    Hessian (eV / A^2) in mass-weighted coordinates and diagonalizes it.

    Returns:
        (freqs_cm1, cart_normal_modes, eigenvalues_ev_amu_a2)
        where freqs_cm1 has length (3N - n_tr), negative values denote imaginary
        frequencies, and cart_normal_modes has shape (n_vib, N, 3) (normalized).
    """
    n_atoms = len(masses)
    m = np.asarray(masses, dtype=np.float64)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(m), 3)

    h_sym = 0.5 * (hessian_ev_a2 + hessian_ev_a2.T)
    h_mw = (inv_sqrt_m[:, np.newaxis] * h_sym) * inv_sqrt_m[np.newaxis, :]

    projector, n_tr = build_eckart_projector(coords, m)
    h_proj = projector @ h_mw @ projector
    h_proj = 0.5 * (h_proj + h_proj.T)

    evals, evecs = np.linalg.eigh(h_proj)

    # Identify and drop the n_tr projected zero modes (those lying in the TR subspace)
    tr_proj = np.eye(3 * n_atoms) - projector
    tr_overlaps = np.array([float(evecs[:, k].T @ tr_proj @ evecs[:, k]) for k in range(3 * n_atoms)])
    tr_indices = set(np.argsort(tr_overlaps)[-n_tr:].tolist())
    vib_indices = [k for k in range(3 * n_atoms) if k not in tr_indices]

    vib_evals = evals[vib_indices]
    vib_evecs_mw = evecs[:, vib_indices]

    # Sort ascending by eigenvalue (most negative / imaginary first)
    order = np.argsort(vib_evals)
    vib_evals = vib_evals[order]
    vib_evecs_mw = vib_evecs_mw[:, order]

    freqs_cm1 = np.sign(vib_evals) * np.sqrt(np.abs(vib_evals) * EIG_EV_AMU_A2_TO_CM1_SQ)

    cart_modes = np.zeros((len(vib_evals), n_atoms, 3), dtype=np.float64)
    for k in range(len(vib_evals)):
        v_cart = (vib_evecs_mw[:, k] * inv_sqrt_m).reshape(n_atoms, 3)
        norm = float(np.linalg.norm(v_cart))
        if norm > 1e-12:
            v_cart /= norm
        cart_modes[k] = v_cart

    return freqs_cm1, cart_modes, vib_evals


def analyze_mode_bond_activity(
    symbols: list[str],
    coords: np.ndarray,
    mode_cart: np.ndarray,
    max_pair_dist_ang: float = 4.2,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """
    Computes the signed bond-length change derivative (d r_ij / dq) for atom pairs
    along a normalized Cartesian normal mode vector (e.g., TS imaginary mode).
    """
    c = np.asarray(coords, dtype=np.float64)
    v = np.asarray(mode_cart, dtype=np.float64)
    n_atoms = len(symbols)
    pairs: list[dict[str, Any]] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            vec = c[j] - c[i]
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6 or dist > max_pair_dist_ang:
                continue
            unit = vec / dist
            dr = float(np.dot(v[j] - v[i], unit))
            pairs.append(
                {
                    "atoms": [i, j],
                    "label": f"{symbols[i]}{i}-{symbols[j]}{j}",
                    "elements": sorted([symbols[i], symbols[j]]),
                    "distance_ang": round(dist, 4),
                    "mode_stretch_component": round(dr, 5),
                    "abs_stretch": abs(dr),
                }
            )
    pairs.sort(key=lambda item: item["abs_stretch"], reverse=True)
    return pairs[:top_k]


def compute_rrho_thermochemistry(
    symbols: list[str],
    coords: np.ndarray,
    freqs_cm1: np.ndarray,
    temp_k: float = 298.15,
    pressure_atm: float = 1.0,
    multiplicity: int = 1,
    symmetry_number: int = 1,
    qrrho_cutoff_cm1: float = 100.0,
    imag_threshold_cm1: float = -15.0,
) -> dict[str, Any]:
    """
    Computes full ideal-gas RRHO and Grimme (2012) qRRHO thermochemical corrections
    (ZPE, H_corr, S_RRHO, S_qRRHO, G_corr_RRHO, G_corr_qRRHO) from molecular geometry
    and Eckart-projected harmonic frequencies.

    Only real positive vibrational frequencies (> 0, excluding imaginary modes) enter
    the vibrational partition function.
    """
    if not math.isfinite(temp_k) or temp_k <= 0.0:
        raise ValueError(f"temp_k must be a finite strictly positive float, got {temp_k!r}.")
    if not math.isfinite(pressure_atm) or pressure_atm <= 0.0:
        raise ValueError(f"pressure_atm must be a finite strictly positive float, got {pressure_atm!r}.")

    masses = masses_from_symbols(symbols)
    total_mass_kg = float(np.sum(masses)) * AMU_TO_KG
    c = np.asarray(coords, dtype=np.float64)
    com = np.sum(c * masses[:, np.newaxis], axis=0) / np.sum(masses)
    r = c - com[np.newaxis, :]

    # Principal moments of inertia (amu * A^2 -> kg * m^2)
    inertia = np.zeros((3, 3), dtype=np.float64)
    for m_i, r_i in zip(masses, r):
        x, y, z = r_i
        r2 = x * x + y * y + z * z
        inertia[0, 0] += m_i * (r2 - x * x)
        inertia[1, 1] += m_i * (r2 - y * y)
        inertia[2, 2] += m_i * (r2 - z * z)
        inertia[0, 1] -= m_i * x * y
        inertia[0, 2] -= m_i * x * z
        inertia[1, 2] -= m_i * y * z
    inertia[1, 0] = inertia[0, 1]
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
    moments_amu_a2 = np.sort(np.linalg.eigvalsh(inertia))
    moments_kg_m2 = moments_amu_a2 * AMU_TO_KG * (ANG_TO_M**2)

    is_linear = bool(moments_amu_a2[0] < 1e-4)

    # Translational energy & Sackur-Tetrode entropy (at pressure_atm)
    p_pa = pressure_atm * 101325.0
    e_trans_kcal = 1.5 * R_KCAL_MOL_K * temp_k
    q_trans = ((2.0 * math.pi * total_mass_kg * KB_J_K * temp_k) / (PLANCK_H**2)) ** 1.5 * (
        (KB_J_K * temp_k) / p_pa
    )
    s_trans_cal = R_KCAL_MOL_K * 1000.0 * (math.log(max(q_trans, 1e-300)) + 2.5)

    # Rotational energy & entropy
    if is_linear:
        e_rot_kcal = R_KCAL_MOL_K * temp_k
        i_lin = float(moments_kg_m2[-1])
        q_rot = (8.0 * math.pi**2 * i_lin * KB_J_K * temp_k) / (symmetry_number * PLANCK_H**2)
        s_rot_cal = R_KCAL_MOL_K * 1000.0 * (math.log(max(q_rot, 1e-300)) + 1.0)
        b_av_kg_m2 = i_lin
    else:
        e_rot_kcal = 1.5 * R_KCAL_MOL_K * temp_k
        prod_i = float(moments_kg_m2[0] * moments_kg_m2[1] * moments_kg_m2[2])
        q_rot = (
            math.sqrt(math.pi * prod_i)
            / symmetry_number
            * ((8.0 * math.pi**2 * KB_J_K * temp_k) / (PLANCK_H**2)) ** 1.5
        )
        s_rot_cal = R_KCAL_MOL_K * 1000.0 * (math.log(max(q_rot, 1e-300)) + 1.5)
        b_av_kg_m2 = float(np.mean(moments_kg_m2))

    # Electronic entropy
    s_elec_cal = R_KCAL_MOL_K * 1000.0 * math.log(max(int(multiplicity), 1))

    # Vibrational ZPE, thermal vibrational energy, RRHO entropy, and Grimme qRRHO entropy
    freqs_list = [float(f) for f in freqs_cm1]
    real_freqs = [f for f in freqs_list if f > 0.0]
    imag_freqs = [f for f in freqs_list if f < imag_threshold_cm1]
    soft_neg_freqs = [f for f in freqs_list if imag_threshold_cm1 <= f <= 0.0]

    zpe_kcal = 0.0
    e_vib_thermal_kcal = 0.0
    s_vib_rrho_cal = 0.0

    for nu in real_freqs:
        zpe_kcal += 0.5 * nu * CM1_TO_KCAL
        x = (PLANCK_H * SPEED_OF_LIGHT_CM_S * nu) / (KB_J_K * temp_k)
        if x < 500.0:
            ex = math.exp(x)
            emx = math.exp(-x)
            e_vib_thermal_kcal += (nu * CM1_TO_KCAL) / (ex - 1.0)
            s_vib_rrho_cal += (R_KCAL_MOL_K * 1000.0) * (x / (ex - 1.0) - math.log(1.0 - emx))

    # Grimme qRRHO vibrational entropy (uses molecular average moment of inertia B_av or standard 1e-44 kg m^2)
    s_vib_qrrho_cal = grimes_qrrho_entropy(
        real_freqs,
        temp=temp_k,
        cutoff_freq=qrrho_cutoff_cm1,
        B_av=1.0e-44,
    )

    e_vib_total_kcal = zpe_kcal + e_vib_thermal_kcal
    h_corr_kcal = e_trans_kcal + e_rot_kcal + e_vib_total_kcal + R_KCAL_MOL_K * temp_k

    s_total_rrho_cal = s_trans_cal + s_rot_cal + s_elec_cal + s_vib_rrho_cal
    s_total_qrrho_cal = s_trans_cal + s_rot_cal + s_elec_cal + s_vib_qrrho_cal

    g_corr_rrho_kcal = h_corr_kcal - temp_k * (s_total_rrho_cal / 1000.0)
    g_corr_qrrho_kcal = h_corr_kcal - temp_k * (s_total_qrrho_cal / 1000.0)

    return {
        "temp_k": float(temp_k),
        "pressure_atm": float(pressure_atm),
        "n_vib_modes": len(freqs_list),
        "n_imag": len(imag_freqs),
        "imag_freqs_cm1": [round(f, 2) for f in imag_freqs],
        "soft_neg_freqs_cm1": [round(f, 2) for f in soft_neg_freqs],
        "lowest_real_freq_cm1": round(min(real_freqs), 2) if real_freqs else None,
        "all_freqs_cm1": [round(f, 2) for f in freqs_list],
        "zpe_kcal": float(zpe_kcal),
        "zpe_ha": float(zpe_kcal * KCAL_TO_HARTREE),
        "h_corr_kcal": float(h_corr_kcal),
        "h_corr_ha": float(h_corr_kcal * KCAL_TO_HARTREE),
        "s_rrho_cal_mol_k": float(s_total_rrho_cal),
        "s_qrrho_cal_mol_k": float(s_total_qrrho_cal),
        "g_corr_rrho_kcal": float(g_corr_rrho_kcal),
        "g_corr_rrho_ha": float(g_corr_rrho_kcal * KCAL_TO_HARTREE),
        "g_corr_qrrho_kcal": float(g_corr_qrrho_kcal),
        "g_corr_qrrho_ha": float(g_corr_qrrho_kcal * KCAL_TO_HARTREE),
        "b_av_kg_m2": float(b_av_kg_m2),
    }


# =============================================================================
# 2. Charge-Aware AIMNet2 Engine (Opt, 11-Image CI-NEB, P-RFO/Sella TS, Hessian)
# =============================================================================


def build_aimnet2_calculator(
    charge: int = 0,
    multiplicity: int = 1,
    model: str = "aimnet2",
) -> Any:
    """Builds an official charge-aware AIMNet2ASE calculator (raises RuntimeError if unavailable)."""
    try:
        from aimnet.calculators import AIMNet2ASE
    except ImportError as exc:
        raise RuntimeError(
            "Charge-aware 'aimnet' package is not installed in this Python environment. "
            "Install aimnet>=0.2.0 or pass --aimnet-python pointing to the AIMNet2 venv."
        ) from exc
    return AIMNet2ASE(model, charge=int(charge), mult=int(multiplicity))


def _make_ase_atoms(symbols: list[str], coords: np.ndarray, charge: int, multiplicity: int, calc: Any) -> Any:
    from ase import Atoms

    atoms = Atoms(symbols=symbols, positions=np.asarray(coords, dtype=np.float64))
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(multiplicity)
    atoms.calc = calc
    return atoms


def evaluate_energy_grad_ase(
    symbols: list[str],
    coords: np.ndarray,
    charge: int,
    multiplicity: int,
    calc: Any,
) -> tuple[float, np.ndarray]:
    """Returns (energy_eV, gradient_eV_per_Angstrom) using an ASE calculator."""
    atoms = _make_ase_atoms(symbols, coords, charge, multiplicity, calc)
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), dtype=np.float64)
    return e, -f


def compute_numerical_hessian_ase(
    symbols: list[str],
    coords: np.ndarray,
    charge: int,
    multiplicity: int,
    calc: Any,
    delta_ang: float = 0.005,
) -> np.ndarray:
    """Computes central-difference Cartesian Hessian (3N x 3N, eV / A^2) from analytical forces."""
    c0 = np.asarray(coords, dtype=np.float64)
    n_atoms = len(symbols)
    n_dof = 3 * n_atoms
    hess = np.zeros((n_dof, n_dof), dtype=np.float64)

    atoms = _make_ase_atoms(symbols, c0, charge, multiplicity, calc)
    for idx in range(n_dof):
        atom_i = idx // 3
        comp_i = idx % 3

        c_plus = c0.copy()
        c_plus[atom_i, comp_i] += delta_ang
        atoms.set_positions(c_plus)
        g_plus = -np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)

        c_minus = c0.copy()
        c_minus[atom_i, comp_i] -= delta_ang
        atoms.set_positions(c_minus)
        g_minus = -np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)

        hess[:, idx] = (g_plus - g_minus) / (2.0 * delta_ang)

    return 0.5 * (hess + hess.T)


def relax_minimum_aimnet2(
    symbols: list[str],
    coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    calc: Any | None = None,
    fmax: float = 0.005,
    max_steps: int = 400,
) -> tuple[np.ndarray, float, float]:
    """Relaxes a minimum geometry with ASE LBFGS/BFGS on AIMNet2. Returns (coords, energy_eV, fmax)."""
    from ase.optimize import BFGS, LBFGS

    eff_calc = build_aimnet2_calculator(charge, multiplicity) if calc is None else calc
    atoms = _make_ase_atoms(symbols, coords, charge, multiplicity, eff_calc)
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)

    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    cur_fmax = float(np.max(np.linalg.norm(forces, axis=1)))
    if cur_fmax > fmax:
        opt2 = BFGS(atoms, logfile=None)
        opt2.run(fmax=fmax, steps=max_steps // 2)
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        cur_fmax = float(np.max(np.linalg.norm(forces, axis=1)))

    return np.asarray(atoms.get_positions(), dtype=np.float64), float(atoms.get_potential_energy()), cur_fmax


def discover_midpoint_intermediate_aimnet2(
    symbols: list[str],
    r_coords: np.ndarray,
    p_coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    temp_k: float = 298.15,
    pressure_atm: float = 1.0,
    calc: Any | None = None,
    rmsd_threshold_ang: float = 0.25,
    fmax_threshold_ev_ang: float = 0.02,
    opt_fmax: float = 0.005,
    max_steps: int = 400,
    min_clash_dist_ang: float = 0.45,
) -> dict[str, Any]:
    """
    Tests whether the midpoint between relaxed reactant R and product P relaxes
    under AIMNet2 to a genuine distinct local minimum (e.g. a tetrahedral oxyanion
    intermediate Int_tet in anionic acyl substitution):
      - converged optimization (fmax < fmax_threshold_ev_ang, default 0.02 eV/A)
      - distinct geometry from both R and P (Kabsch RMSD > rmsd_threshold_ang, default 0.25 A)
      - genuine minimum verified by Eckart-projected Hessian (N_imag == 0)

    Evaluates both the direct atom-mapped midpoint 0.5 * (R + P) and the Kabsch-aligned
    midpoint 0.5 * (R + P_aligned) whenever non-clashing, selecting the lowest-energy
    qualifying distinct intermediate. Never fabricates an intermediate if none exists.
    """
    t0 = time.perf_counter()
    eff_calc = build_aimnet2_calculator(charge, multiplicity) if calc is None else calc
    r_arr = np.asarray(r_coords, dtype=np.float64)
    p_arr = np.asarray(p_coords, dtype=np.float64)

    raw_mid = 0.5 * (r_arr + p_arr)
    p_aligned = kabsch_align(p_arr, r_arr)
    aligned_mid = 0.5 * (r_arr + p_aligned)

    guess_list: list[tuple[str, np.ndarray]] = []
    if _min_interatomic_distance(raw_mid) >= min_clash_dist_ang:
        guess_list.append(("direct_midpoint", raw_mid))
    if _min_interatomic_distance(aligned_mid) >= min_clash_dist_ang:
        if not guess_list or kabsch_rmsd(aligned_mid, raw_mid) > 0.05:
            guess_list.append(("kabsch_midpoint", aligned_mid))

    candidate_records: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_coords: np.ndarray | None = None
    best_hessian: dict[str, Any] | None = None

    for label, guess in guess_list:
        opt_coords, opt_e_ev, opt_fmax_actual = relax_minimum_aimnet2(
            symbols,
            guess,
            charge=charge,
            multiplicity=multiplicity,
            calc=eff_calc,
            fmax=opt_fmax,
            max_steps=max_steps,
        )
        rmsd_r = kabsch_rmsd(opt_coords, r_arr)
        rmsd_p = kabsch_rmsd(opt_coords, p_arr)
        heavy_rmsd_r = heavy_atom_rmsd(symbols, opt_coords, r_arr)
        heavy_rmsd_p = heavy_atom_rmsd(symbols, opt_coords, p_arr)

        converged_opt = bool(math.isfinite(opt_fmax_actual) and opt_fmax_actual < fmax_threshold_ev_ang)
        distinct_from_endpoints = bool(rmsd_r > rmsd_threshold_ang and rmsd_p > rmsd_threshold_ang)

        n_imag: int | None = None
        hess_res: dict[str, Any] | None = None
        qualifies = False
        if converged_opt and distinct_from_endpoints:
            hess_res = analyze_stationary_point_aimnet2(
                symbols=symbols,
                coords=opt_coords,
                charge=charge,
                multiplicity=multiplicity,
                temp_k=temp_k,
                pressure_atm=pressure_atm,
                calc=eff_calc,
            )
            n_imag = int(hess_res["thermo"]["n_imag"])
            qualifies = n_imag == 0

        rec = {
            "candidate": label,
            "energy_ev": float(opt_e_ev),
            "fmax_ev_ang": float(opt_fmax_actual),
            "converged_opt": converged_opt,
            "rmsd_from_r_ang": round(float(rmsd_r), 5),
            "rmsd_from_p_ang": round(float(rmsd_p), 5),
            "heavy_rmsd_from_r_ang": round(float(heavy_rmsd_r), 5),
            "heavy_rmsd_from_p_ang": round(float(heavy_rmsd_p), 5),
            "distinct_from_endpoints": distinct_from_endpoints,
            "n_imag": n_imag,
            "qualifies": qualifies,
        }
        candidate_records.append(rec)

        if qualifies and (best_candidate is None or opt_e_ev < best_candidate["energy_ev"]):
            best_candidate = rec
            best_coords = opt_coords
            best_hessian = hess_res

    wall_s = round(time.perf_counter() - t0, 3)
    return {
        "discovered": best_candidate is not None,
        "coords": best_coords,
        "hessian": best_hessian,
        "energy_ev": best_candidate["energy_ev"] if best_candidate is not None else None,
        "fmax_ev_ang": best_candidate["fmax_ev_ang"] if best_candidate is not None else None,
        "rmsd_from_r_ang": best_candidate["rmsd_from_r_ang"] if best_candidate is not None else None,
        "rmsd_from_p_ang": best_candidate["rmsd_from_p_ang"] if best_candidate is not None else None,
        "source_candidate": best_candidate["candidate"] if best_candidate is not None else None,
        "rmsd_threshold_ang": float(rmsd_threshold_ang),
        "fmax_threshold_ev_ang": float(fmax_threshold_ev_ang),
        "candidates": candidate_records,
        "wall_seconds": wall_s,
    }


def run_cineb_aimnet2(
    symbols: list[str],
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    n_images: int = 11,
    fmax: float = 0.08,
    max_steps: int = 250,
    k_spring: float = 0.1,
    calc: Any | None = None,
) -> dict[str, Any]:
    """
    Runs an N-image (default 11) ASE Climbing-Image NEB with IDPP interpolation
    and FIRE/LBFGS optimization using charge-aware AIMNet2.
    """
    from ase import Atoms
    from ase.mep import NEB
    from ase.optimize import FIRE

    if n_images < 3:
        raise ValueError("CI-NEB requires at least 3 images.")

    t0 = time.perf_counter()
    base_calc = build_aimnet2_calculator(charge, multiplicity) if calc is None else calc

    # Rigidly align end_coords onto start_coords (Kabsch) to prevent artificial rotation in IDPP interpolation
    a0 = np.asarray(start_coords, dtype=np.float64)
    b0 = np.asarray(end_coords, dtype=np.float64)
    b_aligned = kabsch_align(b0, a0)

    images: list[Any] = []
    for idx in range(n_images):
        pos = a0 if idx == 0 else (b_aligned if idx == n_images - 1 else a0.copy())
        at = Atoms(symbols=symbols, positions=pos.copy())
        at.info["charge"] = int(charge)
        at.info["spin"] = int(multiplicity)
        at.calc = base_calc
        images.append(at)

    def _make_neb(climb: bool) -> Any:
        try:
            return NEB(images, k=k_spring, climb=climb, method="improvedtangent", allow_shared_calculator=True)
        except TypeError:
            return NEB(images, k=k_spring, climb=climb, method="improvedtangent")

    neb_init = _make_neb(climb=False)
    try:
        neb_init.interpolate(method="idpp", mic=False)
    except Exception:
        neb_init.interpolate(method="linear", mic=False)

    # Stage 1: regular NEB pre-relaxation
    opt1 = FIRE(neb_init, logfile=None)
    opt1.run(fmax=max(fmax * 2.5, 0.20), steps=max_steps // 2)

    # Stage 2: Climbing-Image NEB
    neb_ci = _make_neb(climb=True)
    opt2 = FIRE(neb_ci, logfile=None)
    opt2.run(fmax=fmax, steps=max_steps)

    energies: list[float] = [float(img.get_potential_energy()) for img in images]
    peak_idx = int(np.argmax(energies[1:-1])) + 1
    peak_coords = np.asarray(images[peak_idx].get_positions(), dtype=np.float64)
    wall_s = time.perf_counter() - t0

    return {
        "n_images": n_images,
        "energies_ev": energies,
        "peak_index": peak_idx,
        "peak_energy_ev": float(energies[peak_idx]),
        "peak_coords": peak_coords,
        "all_coords": [np.asarray(img.get_positions(), dtype=np.float64) for img in images],
        "wall_seconds": round(wall_s, 3),
    }


def _solve_rfo_shift_min(evals: np.ndarray, grads: np.ndarray) -> float:
    """Solves exact minimization RFO secular equation for shift lambda_n < min(0, min(evals))."""
    if len(evals) == 0:
        return 0.0
    n = len(evals)
    aug = np.zeros((n + 1, n + 1), dtype=np.float64)
    aug[:n, :n] = np.diag(evals)
    aug[:n, n] = grads
    aug[n, :n] = grads
    eigvals = np.linalg.eigvalsh(aug)
    return float(eigvals[0])


def _solve_rfo_shift_max(eval_p: float, grad_p: float) -> float:
    """Solves exact 1D maximization RFO secular equation for shift lambda_p > max(0, eval_p)."""
    disc = math.sqrt(eval_p * eval_p + 4.0 * grad_p * grad_p)
    return 0.5 * (eval_p + disc)


def optimize_ts_prfo_aimnet2(
    symbols: list[str],
    initial_coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    calc: Any | None = None,
    fmax: float = 0.002,
    max_steps: int = 60,
    trust_radius_mw: float = 0.15,
    delta_ang: float = 0.005,
    target_bond_pair: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    First-order saddle-point (TS) optimizer using Eckart-projected Partitioned
    Rational Function Optimization (P-RFO / Baker eigenvector-following) in
    mass-weighted coordinates with exact central-difference Hessians.

    Guarantees uphill stepping along one chosen reaction mode and downhill
    minimization along all orthogonal vibrational degrees of freedom.
    """
    t0 = time.perf_counter()
    eff_calc = build_aimnet2_calculator(charge, multiplicity) if calc is None else calc
    masses = masses_from_symbols(symbols)
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    inv_sqrt_m = 1.0 / sqrt_m

    coords = np.asarray(initial_coords, dtype=np.float64).copy()
    prev_followed_mw: np.ndarray | None = None
    converged = False
    cur_fmax = float("inf")
    cur_energy = float("nan")
    steps_taken = 0

    for step in range(max_steps):
        steps_taken = step + 1
        cur_energy, grad_cart = evaluate_energy_grad_ase(symbols, coords, charge, multiplicity, eff_calc)
        cur_fmax = float(np.max(np.linalg.norm(grad_cart, axis=1)))

        hess_cart = compute_numerical_hessian_ase(symbols, coords, charge, multiplicity, eff_calc, delta_ang=delta_ang)
        projector, n_tr = build_eckart_projector(coords, masses)
        h_mw = (inv_sqrt_m[:, np.newaxis] * hess_cart) * inv_sqrt_m[np.newaxis, :]
        h_proj = 0.5 * (projector @ h_mw @ projector + (projector @ h_mw @ projector).T)

        evals, evecs = np.linalg.eigh(h_proj)
        tr_proj = np.eye(len(evals)) - projector
        tr_overlaps = np.array([float(evecs[:, k].T @ tr_proj @ evecs[:, k]) for k in range(len(evals))])
        tr_idx = set(np.argsort(tr_overlaps)[-n_tr:].tolist())
        vib_idx = [k for k in range(len(evals)) if k not in tr_idx]

        vib_evals = evals[vib_idx]
        vib_evecs = evecs[:, vib_idx]
        order = np.argsort(vib_evals)
        vib_evals = vib_evals[order]
        vib_evecs = vib_evecs[:, order]

        freqs_cm1 = np.sign(vib_evals) * np.sqrt(np.abs(vib_evals) * EIG_EV_AMU_A2_TO_CM1_SQ)
        n_imag = int(np.sum(freqs_cm1 < -15.0))

        if cur_fmax <= fmax and n_imag == 1:
            converged = True
            break

        # Choose which vibrational eigenvector to maximize along
        follow_idx = 0
        if prev_followed_mw is not None:
            overlaps = np.abs(vib_evecs.T @ prev_followed_mw)
            # Prefer negative eigenvalue with strong overlap, otherwise best overlap among lowest 4 modes
            cand_limit = min(4, len(vib_evals))
            best_c = int(np.argmax(overlaps[:cand_limit]))
            if overlaps[best_c] > 0.45:
                follow_idx = best_c
        elif target_bond_pair is not None:
            i_a, i_b = target_bond_pair
            bond_vec = coords[i_b] - coords[i_a]
            bond_norm = float(np.linalg.norm(bond_vec))
            if bond_norm > 1e-6:
                unit = bond_vec / bond_norm
                scores = []
                cand_limit = min(5, len(vib_evals))
                for k in range(cand_limit):
                    v_c = (vib_evecs[:, k] * inv_sqrt_m).reshape(-1, 3)
                    v_c /= max(float(np.linalg.norm(v_c)), 1e-12)
                    scores.append(abs(float(np.dot(v_c[i_b] - v_c[i_a], unit))))
                if max(scores) > 0.15:
                    follow_idx = int(np.argmax(scores))

        # If we already have a strong unique negative eigenvalue (< -80 cm^-1), follow mode 0
        if freqs_cm1[0] < -80.0 and (len(freqs_cm1) < 2 or freqs_cm1[1] > -30.0):
            follow_idx = 0

        prev_followed_mw = vib_evecs[:, follow_idx].copy()

        # Project gradient into vibrational eigenbasis
        g_mw = grad_cart.reshape(-1) * inv_sqrt_m
        g_vib = vib_evecs.T @ g_mw

        # Partitioned RFO step: maximize along follow_idx, minimize along all other vibrational modes
        lam_p = _solve_rfo_shift_max(float(vib_evals[follow_idx]), float(g_vib[follow_idx]))
        other_mask = np.ones(len(vib_evals), dtype=bool)
        other_mask[follow_idx] = False
        lam_n = _solve_rfo_shift_min(vib_evals[other_mask], g_vib[other_mask])

        step_vib = np.zeros_like(g_vib)
        denom_p = float(vib_evals[follow_idx] - lam_p)
        if abs(denom_p) > 1e-10:
            step_vib[follow_idx] = -g_vib[follow_idx] / denom_p

        for k in range(len(vib_evals)):
            if k == follow_idx:
                continue
            denom_k = float(vib_evals[k] - lam_n)
            if abs(denom_k) > 1e-10:
                step_vib[k] = -g_vib[k] / denom_k

        # If converged in gradient but N_imag > 1 (tiny soft mode), nudge slightly along mode 1
        if cur_fmax <= fmax * 2.0 and n_imag > 1:
            step_vib[1] += 0.04

        dq = vib_evecs @ step_vib
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > trust_radius_mw:
            dq *= trust_radius_mw / dq_norm

        dx = (dq * inv_sqrt_m).reshape(-1, 3)
        coords = coords + dx

    wall_s = time.perf_counter() - t0
    final_e, final_g = evaluate_energy_grad_ase(symbols, coords, charge, multiplicity, eff_calc)
    final_fmax = float(np.max(np.linalg.norm(final_g, axis=1)))

    return {
        "coords": coords,
        "energy_ev": float(final_e),
        "fmax_ev_ang": float(final_fmax),
        "converged": bool(converged),
        "steps": steps_taken,
        "method": "eckart_prfo",
        "wall_seconds": round(wall_s, 3),
    }


def optimize_ts_aimnet2(
    symbols: list[str],
    initial_coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    calc: Any | None = None,
    method: str = "auto",
    fmax: float = 0.002,
    max_steps: int = 60,
    target_bond_pair: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    Refines a TS guess using Sella (if method='sella' or 'auto' and Sella is installed)
    or Eckart-projected P-RFO eigenvector-following (default fallback and 'prfo').
    """
    eff_calc = build_aimnet2_calculator(charge, multiplicity) if calc is None else calc
    use_sella = False
    if method in {"sella", "auto"}:
        try:
            from sella import Sella  # type: ignore[import-not-found]

            use_sella = True
        except ImportError:
            if method == "sella":
                raise RuntimeError("Sella was explicitly requested but is not installed.")
            use_sella = False

    if use_sella:
        from sella import Sella  # type: ignore[import-not-found]

        t0 = time.perf_counter()
        atoms = _make_ase_atoms(symbols, initial_coords, charge, multiplicity, eff_calc)
        sella_error: str | None = None
        try:
            opt = Sella(atoms, internal=True, logfile=None)
            opt.run(fmax=fmax, steps=max_steps)
            pos = np.asarray(atoms.get_positions(), dtype=np.float64)
            e_ev = float(atoms.get_potential_energy())
            cur_fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
            n_steps = int(opt.nsteps)
        except Exception as exc:
            if method == "sella":
                raise
            sella_error = f"{type(exc).__name__}: {exc}"
            pos = np.asarray(initial_coords, dtype=np.float64)
            e_ev = float("nan")
            cur_fmax = float("inf")
            n_steps = 0

        sella_res: dict[str, Any] = {
            "coords": pos,
            "energy_ev": e_ev,
            "fmax_ev_ang": cur_fmax,
            "converged": bool(sella_error is None and cur_fmax <= fmax),
            "steps": n_steps,
            "method": "sella_internal",
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
        if sella_error is not None:
            sella_res["sella_error"] = sella_error

        if method == "sella":
            return sella_res

        # In auto mode, verify whether Sella actually reached a 1st-order saddle point (N_imag == 1).
        # If not, refine with exact-Hessian Eckart P-RFO; never fabricate N_imag == 1.
        sella_n_imag: int | None = None
        if sella_error is None:
            hess_check = analyze_stationary_point_aimnet2(
                symbols=symbols,
                coords=pos,
                charge=charge,
                multiplicity=multiplicity,
                calc=eff_calc,
            )
            sella_n_imag = int(hess_check["thermo"]["n_imag"])
            if sella_res["converged"] and sella_n_imag == 1:
                return sella_res

        prfo_start = pos if (sella_error is None and np.all(np.isfinite(pos))) else initial_coords
        prfo_res = optimize_ts_prfo_aimnet2(
            symbols=symbols,
            initial_coords=prfo_start,
            charge=charge,
            multiplicity=multiplicity,
            calc=eff_calc,
            fmax=fmax,
            max_steps=max_steps,
            target_bond_pair=target_bond_pair,
        )
        prfo_res["fallback_from"] = "sella_internal"
        prfo_res["sella_fmax_ev_ang"] = cur_fmax if math.isfinite(cur_fmax) else None
        prfo_res["sella_n_imag"] = sella_n_imag
        if sella_error is not None:
            prfo_res["sella_error"] = sella_error
        return prfo_res

    return optimize_ts_prfo_aimnet2(
        symbols=symbols,
        initial_coords=initial_coords,
        charge=charge,
        multiplicity=multiplicity,
        calc=eff_calc,
        fmax=fmax,
        max_steps=max_steps,
        target_bond_pair=target_bond_pair,
    )


def analyze_stationary_point_aimnet2(
    symbols: list[str],
    coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    temp_k: float = 298.15,
    pressure_atm: float = 1.0,
    calc: Any | None = None,
    delta_ang: float = 0.005,
    imag_threshold_cm1: float = -15.0,
) -> dict[str, Any]:
    """Evaluates AIMNet2 energy, gradient, Eckart-projected Hessian, frequencies, mode analysis, and RRHO/qRRHO thermo."""
    eff_calc = build_aimnet2_calculator(charge, multiplicity) if calc is None else calc
    t0 = time.perf_counter()
    energy_ev, grad = evaluate_energy_grad_ase(symbols, coords, charge, multiplicity, eff_calc)
    fmax = float(np.max(np.linalg.norm(grad, axis=1)))
    hess = compute_numerical_hessian_ase(symbols, coords, charge, multiplicity, eff_calc, delta_ang=delta_ang)
    masses = masses_from_symbols(symbols)
    freqs_cm1, cart_modes, _ = diagonalize_hessian_eckart(hess, coords, masses)

    thermo = compute_rrho_thermochemistry(
        symbols=symbols,
        coords=coords,
        freqs_cm1=freqs_cm1,
        temp_k=temp_k,
        pressure_atm=pressure_atm,
        multiplicity=multiplicity,
        imag_threshold_cm1=imag_threshold_cm1,
    )

    imag_mode_cart: np.ndarray | None = None
    imag_bond_activity: list[dict[str, Any]] = []
    if len(freqs_cm1) > 0 and freqs_cm1[0] < imag_threshold_cm1:
        imag_mode_cart = cart_modes[0]
        imag_bond_activity = analyze_mode_bond_activity(symbols, coords, imag_mode_cart)

    return {
        "aimnet2_energy_ev": float(energy_ev),
        "aimnet2_energy_ha": float(energy_ev * EV_TO_HARTREE),
        "fmax_ev_ang": float(fmax),
        "thermo": thermo,
        "imag_mode_cart": imag_mode_cart.tolist() if imag_mode_cart is not None else None,
        "imag_mode_bond_activity": imag_bond_activity,
        "hessian_wall_seconds": round(time.perf_counter() - t0, 3),
    }


def verify_ts_irc_aimnet2(
    symbols: list[str],
    ts_coords: np.ndarray,
    imag_mode_cart: np.ndarray,
    ref_a_coords: np.ndarray,
    ref_b_coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    calc: Any | None = None,
    step_size_amu_ang: float = 0.08,
    max_steps: int = 200,
    rmsd_tol: float = 0.15,
    grad_tol: float = 5e-4,
) -> dict[str, Any]:
    """Runs fixed mass-weighted bidirectional IRC + endpoint relaxation on AIMNet2."""
    eff_calc = build_aimnet2_calculator(charge, multiplicity) if calc is None else calc
    masses = masses_from_symbols(symbols)
    tracer = IRCTracerEngine(step_size_amu_ang=step_size_amu_ang, grad_tol=grad_tol)

    def _grad_fn(c: np.ndarray) -> tuple[float, np.ndarray]:
        return evaluate_energy_grad_ase(symbols, c, charge, multiplicity, eff_calc)

    def _relax_fn(c: np.ndarray) -> tuple[np.ndarray, float]:
        pos, e_ev, _ = relax_minimum_aimnet2(
            symbols, c, charge=charge, multiplicity=multiplicity, calc=eff_calc, fmax=0.001, max_steps=400
        )
        return pos, e_ev

    t0 = time.perf_counter()
    res = tracer.verify_endpoints(
        symbols=symbols,
        ts_coords=ts_coords,
        masses=masses,
        imag_mode_vector=np.asarray(imag_mode_cart, dtype=np.float64),
        grad_fn=_grad_fn,
        ref_a_coords=ref_a_coords,
        ref_b_coords=ref_b_coords,
        relax_fn=_relax_fn,
        max_steps=max_steps,
        rmsd_tol=rmsd_tol,
    )
    res["wall_seconds"] = round(time.perf_counter() - t0, 3)
    return res


# =============================================================================
# 3. PySCF / GPU4PySCF DF-wB97X-D3(0)/def2-TZVPP + PCM Single-Point Runner
# =============================================================================


def format_dft_method_label(xc: str, disp: str | None) -> str:
    """
    Formats a scientifically exact density-fitted DFT method label.

    In PySCF 2.14 / GPU4PySCF 1.9, ``scf.dispersion.parse_dft`` blacklists the
    Lin et al. (2013) reparameterized ``wb97x_d3`` / ``wb97x-d3`` keys and
    ``pyscf-dispersion`` does not ship a ``wb97xd3`` D3(0) parameter table.
    Consequently, ``xc="wb97x"`` (Chai & Head-Gordon 2008 bare wB97X, Libxc ID 464)
    paired with ``disp="d3zero"`` is explicitly labeled ``DF-wB97X-D3(0)`` — never
    mislabeled as reparameterized ``DF-wB97X-D3``.
    """
    xc_clean = str(xc or "wb97x").strip()
    xc_key = xc_clean.lower()
    xc_display_map = {
        "wb97x": "wB97X",
        "wb97x_v": "wB97X-V",
        "wb97x-v": "wB97X-V",
        "wb97m_v": "wB97M-V",
        "wb97m-v": "wB97M-V",
        "wb97x-d3bj": "wB97X-D3(BJ)",
        "wb97m-d3bj": "wB97M-D3(BJ)",
        "b3lyp": "B3LYP",
        "pbe0": "PBE0",
        "r2scan": "r2SCAN",
    }
    xc_label = xc_display_map.get(xc_key, xc_clean)

    disp_key = str(disp or "").strip().lower()
    if not disp_key or disp_key == "none":
        return f"DF-{xc_label}"

    disp_display_map = {
        "d3zero": "D3(0)",
        "d3bj": "D3(BJ)",
        "d4": "D4",
    }
    disp_label = disp_display_map.get(disp_key, disp_key.upper())
    return f"DF-{xc_label}-{disp_label}"


def run_pyscf_single_point(
    symbols: list[str],
    coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    basis: str = "def2-tzvpp",
    # Bare 2008 Chai & Head-Gordon wB97X (Libxc ID 464) + Grimme D3(0) -> DF-wB97X-D3(0).
    # PySCF 2.14 blacklists wb97x_d3/wb97x-d3 in pyscf.scf.dispersion.parse_dft.
    xc: str = "wb97x",
    disp: str = "d3zero",
    solvent_name: str = "acetonitrile",
    eps: float | None = None,
    solvation_model: str = "pcm",
    engine: str = "cpu",
    grid_level: int = 3,
    conv_tol: float = 1e-9,
    max_cycle: int = 150,
    gas_phase_ftmw_lock: bool = False,
) -> dict[str, Any]:
    """
    Executes a genuine PySCF or GPU4PySCF DF-DFT + D3 + PCM/ddCOSMO single-point
    energy calculation (default: DF-wB97X-D3(0)/def2-TZVPP using bare 2008 wB97X
    + Grimme D3(0), distinct from Lin et al. 2013 reparameterized wB97X-D3) and
    returns full electronic/solvation/dispersion telemetry.
    """
    eff_solvent, eff_eps = resolve_solvent_eps(solvent_name, eps)
    phase_mode = "gas" if (solvation_model.lower() == "gas" or eff_solvent in {"gas", "vacuum"}) else solvation_model.lower()

    # Enforce phase-aware CP-FTMW lock via canonical thermo guard
    if phase_mode != "gas":
        apply_implicit_solvation_correction(
            delta_g_gas=0.0,
            solvent_name=eff_solvent,
            delta_g_solv_ts=0.0,
            delta_g_solv_reactants=0.0,
            phase=phase_mode,
            gas_phase_ftmw_lock=gas_phase_ftmw_lock,
        )

    try:
        import pyscf
        from pyscf import gto
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is not installed in this Python environment. "
            "Install pyscf + pyscf-dispersion or pass --pyscf-python."
        ) from exc

    disp_version = None
    try:
        import pyscf.dispersion as pdisp

        disp_version = getattr(pdisp, "__version__", "installed")
    except ImportError:
        if disp and disp.lower() not in {"none", ""}:
            raise RuntimeError("pyscf-dispersion is required for D3 dispersion corrections.")

    gpu4pyscf_version = None
    engine_norm = engine.strip().lower()
    if engine_norm == "gpu":
        try:
            import gpu4pyscf
            from gpu4pyscf import dft as gpu_dft

            gpu4pyscf_version = getattr(gpu4pyscf, "__version__", "installed")
        except ImportError as exc:
            raise RuntimeError("engine='gpu' requested but gpu4pyscf is not installed.") from exc
    elif engine_norm != "cpu":
        raise ValueError(f"Unsupported DFT engine '{engine}'. Expected 'cpu' or 'gpu'.")

    t0 = time.perf_counter()
    atom_spec = [[sym, [float(xyz[0]), float(xyz[1]), float(xyz[2])]] for sym, xyz in zip(symbols, coords)]
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.basis = basis
    mol.charge = int(charge)
    mol.spin = int(multiplicity) - 1
    mol.verbose = 0
    mol.build()

    if engine_norm == "gpu":
        if mol.spin == 0:
            mf = gpu_dft.rks.RKS(mol, xc=xc).density_fit()
        else:
            mf = gpu_dft.uks.UKS(mol, xc=xc).density_fit()
    else:
        from pyscf import dft as cpu_dft

        if mol.spin == 0:
            mf = cpu_dft.RKS(mol, xc=xc).density_fit()
        else:
            mf = cpu_dft.UKS(mol, xc=xc).density_fit()

    mf.grids.level = int(grid_level)
    mf.conv_tol = float(conv_tol)
    mf.max_cycle = int(max_cycle)

    if disp and disp.lower() not in {"none", ""}:
        # Attach explicit Grimme dispersion (e.g. d3zero -> D3(0)) to the base xc functional.
        mf.disp = disp.lower()

    if phase_mode == "pcm":
        mf = mf.PCM()
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.eps = eff_eps
    elif phase_mode == "ddcosmo":
        mf = mf.ddCOSMO()
        mf.with_solvent.eps = eff_eps
    elif phase_mode != "gas":
        raise ValueError(f"Unsupported solvation_model '{solvation_model}'. Expected 'pcm', 'ddcosmo', or 'gas'.")

    e_tot = float(mf.kernel())
    converged = bool(mf.converged)

    # Second-order Newton-Raphson SCF retry if first DIIS run did not converge
    if not converged and engine_norm == "cpu":
        mf_nr = mf.newton()
        e_tot = float(mf_nr.kernel())
        converged = bool(mf_nr.converged)
        mf = mf_nr

    wall_s = time.perf_counter() - t0
    summary = getattr(mf, "scf_summary", {}) or {}
    e_disp = summary.get("dispersion", 0.0)
    if hasattr(e_disp, "item"):
        e_disp = float(e_disp.item())
    else:
        e_disp = float(e_disp) if e_disp is not None else 0.0

    e_solv = summary.get("e_solvent", 0.0)
    if hasattr(e_solv, "item"):
        e_solv = float(e_solv.item())
    else:
        e_solv = float(e_solv) if e_solv is not None else 0.0

    return {
        "engine": engine_norm,
        "converged": converged,
        "e_tot_ha": e_tot,
        "e_solvent_ha": e_solv,
        "e_dispersion_ha": e_disp,
        "e_scf_no_disp_ha": float(e_tot - e_disp),
        "basis": basis,
        "xc": xc,
        "disp": disp.lower() if disp else "none",
        "method_label": format_dft_method_label(xc, disp),
        "solvation_model": phase_mode,
        "solvent_name": eff_solvent,
        "eps": eff_eps,
        "grid_level": int(grid_level),
        "n_ao": int(mol.nao_nr()),
        "n_electrons": int(mol.nelectron),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "dispersion_version": disp_version,
        "gpu4pyscf_version": gpu4pyscf_version,
        "wall_seconds": round(wall_s, 3),
    }


# =============================================================================
# 4. Stationary Table & Provenance Manifest Export
# =============================================================================


def assemble_and_export_profile(
    output_dir: str | Path,
    case_name: str,
    reaction_equation: str,
    states_data: dict[str, dict[str, Any]],
    stoich_report: dict[str, Any],
    irc_reports: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Assembles relative energies (Delta E, ZPE, Delta H, Delta G_RRHO, Delta G_qRRHO)
    strictly from raw Hartree totals, exports converged XYZs, stationary_table.csv,
    stationary_table.md, results.json, and manifest_sha256.json.

    Any unconverged state is explicitly marked PENDING / NaN — never fabricated.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz_dir = out / "geometries"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    requested_order = metadata.get("stationary_points")
    if requested_order:
        ordered_states = [s for s in requested_order if s in states_data]
    elif set(states_data.keys()) == set(ELEMENTARY_STATIONARY_STATES):
        ordered_states = list(ELEMENTARY_STATIONARY_STATES)
    else:
        ordered_states = [s for s in STATIONARY_STATES if s in states_data]
    for s in states_data:
        if s not in ordered_states:
            ordered_states.append(s)

    pathway_mode = str(
        metadata.get("pathway_mode")
        or ("elementary_3pt" if ordered_states == list(ELEMENTARY_STATIONARY_STATES) else "stepwise_5pt")
    )
    input_mode = str(metadata.get("input_mode") or "5pt_explicit")

    # Export converged XYZs
    xyz_paths: dict[str, Path] = {}
    for st in ordered_states:
        entry = states_data[st]
        comment = (
            f"{case_name} | {st} | Q={stoich_report['charge']} M={stoich_report['multiplicity']} "
            f"| E_DFT_Ha={entry.get('dft', {}).get('e_tot_ha', 'PENDING')}"
        )
        p = write_xyz(xyz_dir / f"{st}.xyz", entry["symbols"], np.asarray(entry["coords"], dtype=np.float64), comment)
        xyz_paths[st] = p

    # Export top-level ts_polished.xyz for the primary (or highest-energy) TS state
    ts_states = [s for s in ordered_states if s.startswith("TS")]
    primary_ts_state: str | None = None
    if "TS" in states_data:
        primary_ts_state = "TS"
    elif ts_states:
        def _ts_sort_energy(st_name: str) -> float:
            d = states_data[st_name].get("dft", {})
            if d.get("converged") and "e_tot_ha" in d and math.isfinite(float(d["e_tot_ha"])):
                return float(d["e_tot_ha"])
            h = states_data[st_name].get("hessian", {})
            return float(h.get("aimnet2_energy_ha", float("-inf")))

        primary_ts_state = max(ts_states, key=_ts_sort_energy)

    ts_polished_path: Path | None = None
    if primary_ts_state is not None:
        ts_entry = states_data[primary_ts_state]
        ts_thermo = ts_entry.get("hessian", {}).get("thermo", {})
        ts_imag_list = ts_thermo.get("imag_freqs_cm1", [])
        ts_imag_lbl = f"{ts_imag_list[0]:.2f} cm^-1" if ts_imag_list else "none"
        ts_comment = (
            f"{case_name} | {primary_ts_state} ({pathway_mode}) | Q={stoich_report['charge']} "
            f"M={stoich_report['multiplicity']} | N_imag={ts_thermo.get('n_imag', -1)} ({ts_imag_lbl}) "
            f"| E_DFT_Ha={ts_entry.get('dft', {}).get('e_tot_ha', 'PENDING')}"
        )
        ts_polished_path = write_xyz(
            out / "ts_polished.xyz",
            ts_entry["symbols"],
            np.asarray(ts_entry["coords"], dtype=np.float64),
            ts_comment,
        )

    # Reference state R
    ref = states_data.get("R", {})
    ref_dft = ref.get("dft", {})
    ref_thermo = ref.get("hessian", {}).get("thermo", {})
    ref_converged = bool(ref_dft.get("converged", False))
    ref_e_ha = float(ref_dft["e_tot_ha"]) if ref_converged else float("nan")
    ref_h_ha = (ref_e_ha + float(ref_thermo["h_corr_ha"])) if (ref_converged and "h_corr_ha" in ref_thermo) else float("nan")
    ref_g_rrho_ha = (ref_e_ha + float(ref_thermo["g_corr_rrho_ha"])) if (ref_converged and "g_corr_rrho_ha" in ref_thermo) else float("nan")
    ref_g_qrrho_ha = (ref_e_ha + float(ref_thermo["g_corr_qrrho_ha"])) if (ref_converged and "g_corr_qrrho_ha" in ref_thermo) else float("nan")

    rows: list[dict[str, Any]] = []
    for st in ordered_states:
        entry = states_data[st]
        dft = entry.get("dft", {})
        hess = entry.get("hessian", {})
        thermo = hess.get("thermo", {})
        conv = bool(dft.get("converged", False))

        e_ha = float(dft["e_tot_ha"]) if conv else float("nan")
        zpe_ha = float(thermo["zpe_ha"]) if "zpe_ha" in thermo else float("nan")
        h_corr_ha = float(thermo["h_corr_ha"]) if "h_corr_ha" in thermo else float("nan")
        g_rrho_corr_ha = float(thermo["g_corr_rrho_ha"]) if "g_corr_rrho_ha" in thermo else float("nan")
        g_qrrho_corr_ha = float(thermo["g_corr_qrrho_ha"]) if "g_corr_qrrho_ha" in thermo else float("nan")

        h_tot_ha = e_ha + h_corr_ha if (conv and math.isfinite(h_corr_ha)) else float("nan")
        g_rrho_ha = e_ha + g_rrho_corr_ha if (conv and math.isfinite(g_rrho_corr_ha)) else float("nan")
        g_qrrho_ha = e_ha + g_qrrho_corr_ha if (conv and math.isfinite(g_qrrho_corr_ha)) else float("nan")

        de_kcal = (e_ha - ref_e_ha) * HARTREE_TO_KCAL if (conv and ref_converged) else float("nan")
        dh_kcal = (h_tot_ha - ref_h_ha) * HARTREE_TO_KCAL if (math.isfinite(h_tot_ha) and math.isfinite(ref_h_ha)) else float("nan")
        dg_rrho_kcal = (g_rrho_ha - ref_g_rrho_ha) * HARTREE_TO_KCAL if (math.isfinite(g_rrho_ha) and math.isfinite(ref_g_rrho_ha)) else float("nan")
        dg_qrrho_kcal = (g_qrrho_ha - ref_g_qrrho_ha) * HARTREE_TO_KCAL if (math.isfinite(g_qrrho_ha) and math.isfinite(ref_g_qrrho_ha)) else float("nan")

        imag_freqs = thermo.get("imag_freqs_cm1", [])
        n_imag = int(thermo.get("n_imag", -1)) if thermo else -1
        imag_str = f"{imag_freqs[0]:.2f}" if imag_freqs else "0.00"

        # Top forming/breaking bond distance if TS (prefer C-O for acyl/carbonate, else top heavy-atom pair)
        top_co = ""
        bond_activity = hess.get("imag_mode_bond_activity", [])
        for item in bond_activity:
            if item.get("elements") == ["C", "O"]:
                top_co = f"{item['label']}={item['distance_ang']:.3f}A"
                break
        if not top_co and bond_activity:
            for item in bond_activity:
                if "H" not in item.get("elements", []):
                    top_co = f"{item['label']}={item['distance_ang']:.3f}A"
                    break
            if not top_co:
                item0 = bond_activity[0]
                top_co = f"{item0['label']}={item0['distance_ang']:.3f}A"

        rows.append(
            {
                "state": st,
                "status": "CONVERGED" if conv else "PENDING",
                "n_atoms": stoich_report["n_atoms"],
                "charge": stoich_report["charge"],
                "multiplicity": stoich_report["multiplicity"],
                "n_imag": n_imag,
                "imag_freq_cm1": imag_str,
                "forming_breaking_co": top_co,
                "e_dft_ha": f"{e_ha:.10f}" if math.isfinite(e_ha) else "NaN",
                "zpe_ha": f"{zpe_ha:.10f}" if math.isfinite(zpe_ha) else "NaN",
                "h_tot_ha": f"{h_tot_ha:.10f}" if math.isfinite(h_tot_ha) else "NaN",
                "g_rrho_ha": f"{g_rrho_ha:.10f}" if math.isfinite(g_rrho_ha) else "NaN",
                "g_qrrho_ha": f"{g_qrrho_ha:.10f}" if math.isfinite(g_qrrho_ha) else "NaN",
                "delta_e_kcal": f"{de_kcal:.4f}" if math.isfinite(de_kcal) else "NaN",
                "delta_h_kcal": f"{dh_kcal:.4f}" if math.isfinite(dh_kcal) else "NaN",
                "delta_g_rrho_kcal": f"{dg_rrho_kcal:.4f}" if math.isfinite(dg_rrho_kcal) else "NaN",
                "delta_g_qrrho_kcal": f"{dg_qrrho_kcal:.4f}" if math.isfinite(dg_qrrho_kcal) else "NaN",
                "dft_wall_s": dft.get("wall_seconds", "NaN"),
                "dft_engine": dft.get("engine", "PENDING"),
            }
        )

    # Write stationary_table.csv
    csv_path = out / "stationary_table.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write stationary_table.md
    n_pts = len(ordered_states)
    dft_method_label = format_dft_method_label(
        str(metadata.get("xc", "wb97x")),
        str(metadata.get("disp", "d3zero")),
    )
    md_lines = [
        f"# {case_name} — {n_pts}-Stationary-Point Reaction Profile",
        "",
        f"- **Reaction:** `{reaction_equation}`",
        f"- **Pathway Mode:** `{pathway_mode}` (`{input_mode}`)",
        f"- **Formula / Charge / Spin:** `{stoich_report['formula']}` ({stoich_report['n_atoms']} atoms, Q = {stoich_report['charge']}, M = {stoich_report['multiplicity']})",
        f"- **DFT Level:** `{dft_method_label}/{metadata.get('basis', 'def2-tzvpp')}` + `{metadata.get('solvation_model', 'pcm').upper()}({metadata.get('solvent_name', 'acetonitrile')}, eps={metadata.get('eps', 'N/A')})`",
        f"- **Thermochemistry Temperature / Pressure:** `{metadata.get('temp_k', 298.15)} K`, `{metadata.get('pressure_atm', 1.0)} atm` (AIMNet2 Hessian + RRHO / Grimme qRRHO)",
        "",
        "| State | Status | N_imag | Imag Freq (cm^-1) | Key Active Bond | E_DFT (Ha) | dE (kcal/mol) | dH (kcal/mol) | dG_RRHO (kcal/mol) | dG_qRRHO (kcal/mol) | DFT Time (s) |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['state']} | {r['status']} | {r['n_imag']} | {r['imag_freq_cm1']} | {r['forming_breaking_co'] or '-'} | "
            f"{r['e_dft_ha']} | {r['delta_e_kcal']} | {r['delta_h_kcal']} | {r['delta_g_rrho_kcal']} | {r['delta_g_qrrho_kcal']} | {r['dft_wall_s']} |"
        )

    if irc_reports:
        md_lines.extend(["", "## Bidirectional Mass-Weighted IRC Connectivity Verification", ""])
        md_lines.append("| TS | Orientation | Max COM Drift (A) | RMSD End A (A) | RMSD End B (A) | Graph A | Graph B | Strict Connected |")
        md_lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
        for ts_name, irc in irc_reports.items():
            md_lines.append(
                f"| {ts_name} | {irc.get('orientation', '-')} | {irc.get('max_com_drift_ang', float('nan')):.3e} | "
                f"{irc.get('rmsd_a_ang', float('nan')):.4f} | {irc.get('rmsd_b_ang', float('nan')):.4f} | "
                f"{irc.get('bond_graph_match_a')} | {irc.get('bond_graph_match_b')} | {irc.get('connected_strict')} |"
            )

    cpu_bench_rows = [
        (st, states_data[st].get("dft", {}), states_data[st].get("dft_cpu_benchmark"))
        for st in ordered_states
        if states_data[st].get("dft_cpu_benchmark")
    ]
    if cpu_bench_rows:
        md_lines.extend(["", "## PySCF CPU vs GPU4PySCF Single-Point Benchmark", ""])
        md_lines.append("| State | N_AO | GPU Time (s) | CPU Time (s) | Speedup | |E_GPU - E_CPU| (Ha) | 5-Point Est. CPU Time (min) |")
        md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for st, gpu_d, cpu_d in cpu_bench_rows:
            g_t = float(gpu_d.get("wall_seconds", float("nan")))
            c_t = float(cpu_d.get("wall_seconds", float("nan")))
            sp = c_t / g_t if (math.isfinite(g_t) and g_t > 0 and math.isfinite(c_t)) else float("nan")
            diff_ha = abs(float(gpu_d.get("e_tot_ha", float("nan"))) - float(cpu_d.get("e_tot_ha", float("nan"))))
            est_5pt_min = (5.0 * c_t) / 60.0 if math.isfinite(c_t) else float("nan")
            md_lines.append(
                f"| {st} | {cpu_d.get('n_ao', '-')} | {g_t:.3f} | {c_t:.3f} | {sp:.2f}x | {diff_ha:.2e} | {est_5pt_min:.2f} |"
            )

    md_path = out / "stationary_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Write full machine-readable results.json (strip huge arrays from IRC if present)
    clean_irc: dict[str, Any] = {}
    for k, v in irc_reports.items():
        clean_irc[k] = {
            ik: iv
            for ik, iv in v.items()
            if ik not in {"path_coords", "path_energies", "endpoint_a_coords", "endpoint_b_coords"}
        }
        if "path_energies" in v:
            clean_irc[k]["path_energies_ev"] = [float(x) for x in v["path_energies"]]

    clean_states: dict[str, Any] = {}
    for st in ordered_states:
        entry = states_data[st]
        clean_states[st] = {
            "xyz_file": f"geometries/{st}.xyz",
            "hessian": {
                hk: hv for hk, hv in entry.get("hessian", {}).items() if hk != "imag_mode_cart"
            },
            "dft": entry.get("dft", {}),
            "dft_cpu_benchmark": entry.get("dft_cpu_benchmark"),
        }

    ts_n_imag_by_state: dict[str, int] = {
        st: int(states_data[st].get("hessian", {}).get("thermo", {}).get("n_imag", -1))
        for st in ts_states
    }
    ts_n_imag_value: int | dict[str, int]
    if len(ts_states) == 1:
        ts_n_imag_value = ts_n_imag_by_state[ts_states[0]]
    else:
        ts_n_imag_value = ts_n_imag_by_state

    eff_pressure_atm = float(
        metadata.get("pressure_atm", ref_thermo.get("pressure_atm", 1.0))
    )
    metadata_out = dict(metadata)
    metadata_out.setdefault("pressure_atm", eff_pressure_atm)

    results_payload: dict[str, Any] = {
        "schema_version": "cochem.kinetic.rxn-profile/1",
        "case_name": case_name,
        "reaction_equation": reaction_equation,
        "pathway_mode": pathway_mode,
        "input_mode": input_mode,
        "pressure_atm": eff_pressure_atm,
        "stationary_points": ordered_states,
        "ts_n_imaginary_modes": ts_n_imag_value,
        "ts_n_imaginary_modes_by_state": ts_n_imag_by_state,
        "ts_all_first_order": bool(ts_states and all(v == 1 for v in ts_n_imag_by_state.values())),
        "primary_ts_state": primary_ts_state,
        "metadata": metadata_out,
        "stoichiometry": stoich_report,
        "stationary_table": rows,
        "states": clean_states,
        "irc_verification": clean_irc,
    }
    if metadata.get("intermediate_discovery") is not None:
        results_payload["intermediate_discovery"] = metadata["intermediate_discovery"]
    if metadata.get("ts_diagnostics") is not None:
        results_payload["ts_diagnostics"] = metadata["ts_diagnostics"]
    results_path = out / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")

    # Write SHA-256 provenance manifest covering every artifact
    manifest_files: dict[str, str] = {}
    for fpath in sorted(out.rglob("*")):
        if fpath.is_file() and fpath.name != "manifest_sha256.json":
            rel = fpath.relative_to(out).as_posix()
            manifest_files[rel] = sha256_file(fpath)

    manifest_payload = {
        "schema_version": "cochem.provenance-manifest/1",
        "case_name": case_name,
        "files_sha256": manifest_files,
    }
    manifest_path = out / "manifest_sha256.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    return {
        "pathway_mode": pathway_mode,
        "input_mode": input_mode,
        "stationary_points": ordered_states,
        "ts_n_imaginary_modes": ts_n_imag_value,
        "ts_all_first_order": results_payload["ts_all_first_order"],
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "results_path": str(results_path),
        "ts_polished_path": str(ts_polished_path) if ts_polished_path is not None else None,
        "manifest_path": str(manifest_path),
        "rows": rows,
    }
