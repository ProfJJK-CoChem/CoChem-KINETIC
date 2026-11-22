#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 3: Intrinsic Reaction Coordinate (IRC) Path Tracer
-------------------------------------------------------------------------
Implements mass-weighted Gonzalez-Schlegel / Heun predictor-corrector IRC integration
starting from a transition state geometry along forward and reverse mass-weighted
imaginary-mode vectors, plus Kabsch heavy-atom RMSD and bond-graph endpoint verification.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

# Standard Pyykko / Alvarez covalent radii (Angstrom) for bond-graph verification
COVALENT_RADII_ANG: dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def compute_center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Computes the mass-weighted center of mass (3,) in Angstrom."""
    c = np.asarray(coords, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    total_mass = float(np.sum(m))
    if total_mass <= 0.0:
        raise ValueError("Total atomic mass must be strictly positive.")
    return np.sum(c * m[:, np.newaxis], axis=0) / total_mass


def project_out_com_translation(mode_cart: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Projects out net center-of-mass translation from a Cartesian displacement vector."""
    v = np.asarray(mode_cart, dtype=np.float64).copy()
    m = np.asarray(masses, dtype=np.float64)
    total_mass = float(np.sum(m))
    if total_mass <= 0.0:
        raise ValueError("Total atomic mass must be strictly positive.")
    com_shift = np.sum(v * m[:, np.newaxis], axis=0) / total_mass
    return v - com_shift[np.newaxis, :]


def kabsch_rmsd(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    atom_mask: np.ndarray | None = None,
) -> float:
    """
    Computes optimal rigid-body (Kabsch-aligned) RMSD in Angstrom between two
    same-indexed geometries, optionally restricted to atom_mask (e.g., heavy atoms).
    """
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Expected matching (N, 3) coordinate arrays, got {a.shape} and {b.shape}.")
    if atom_mask is not None:
        mask = np.asarray(atom_mask, dtype=bool)
        a = a[mask]
        b = b[mask]
    if len(a) == 0:
        raise ValueError("Cannot compute Kabsch RMSD on zero selected atoms.")

    a_centered = a - np.mean(a, axis=0)
    b_centered = b - np.mean(b, axis=0)

    cov = a_centered.T @ b_centered
    u_mat, _, vt_mat = np.linalg.svd(cov)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0.0:
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T

    a_aligned = a_centered @ rot.T
    diff = a_aligned - b_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def heavy_atom_rmsd(symbols: list[str], coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Computes Kabsch-aligned heavy-atom (non-hydrogen) RMSD in Angstrom."""
    heavy_mask = np.array([sym != "H" for sym in symbols], dtype=bool)
    if not np.any(heavy_mask):
        heavy_mask = np.ones(len(symbols), dtype=bool)
    return kabsch_rmsd(coords_a, coords_b, atom_mask=heavy_mask)


def build_connectivity_graph(
    symbols: list[str],
    coords: np.ndarray,
    covalent_scale: float = 1.25,
) -> set[tuple[int, int]]:
    """Builds covalent bond adjacency edge set {(i, j) | i < j} from covalent radii."""
    c = np.asarray(coords, dtype=np.float64)
    n_atoms = len(symbols)
    edges: set[tuple[int, int]] = set()
    for i in range(n_atoms):
        r_i = COVALENT_RADII_ANG.get(symbols[i], 0.85)
        for j in range(i + 1, n_atoms):
            r_j = COVALENT_RADII_ANG.get(symbols[j], 0.85)
            dist = float(np.linalg.norm(c[i] - c[j]))
            if 0.4 <= dist <= covalent_scale * (r_i + r_j):
                edges.add((i, j))
    return edges


class IRCTracerEngine:
    """Bidirectional mass-weighted IRC Path Integrator (Heun Predictor-Corrector)."""

    def __init__(self, step_size_amu_ang: float = 0.1, grad_tol: float = 1e-3) -> None:
        self.step_size = float(step_size_amu_ang)
        self.grad_tol = float(grad_tol)
        self.logger = logging.getLogger("CoChem_KINETIC_IRC")

    def trace_irc_path(
        self,
        ts_coords: np.ndarray,
        masses: np.ndarray,
        imag_mode_vector: np.ndarray,
        grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
        max_steps: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Traces forward and reverse mass-weighted IRC pathways from a TS geometry.

        Mass-weighted coordinates are q_i = sqrt(m_i) * x_i, so:
          g_q = dV/dq = M^{-1/2} g_x
          dq  = -ds * g_q / ||g_q||
          dx  = M^{-1/2} dq = -ds * (M^{-1} g_x) / ||M^{-1/2} g_x||
        which strictly preserves center of mass (sum_i m_i dx_i = 0) for any
        translationally invariant potential.

        Returns:
            (all_path_coords, path_energies) ordered reverse_end -> ... -> TS -> ... -> forward_end.
        """
        ts_c = np.asarray(ts_coords, dtype=np.float64)
        masses_arr = np.asarray(masses, dtype=np.float64)
        if masses_arr.ndim != 1 or len(masses_arr) != len(ts_c):
            raise ValueError("masses must be a 1D array matching the number of atoms in ts_coords.")
        if np.any(masses_arr <= 0.0):
            raise ValueError("All atomic masses must be strictly positive.")

        mw = np.sqrt(masses_arr)[:, np.newaxis]  # M^{1/2}
        inv_mw = 1.0 / mw                        # M^{-1/2}

        # Ensure initial displacement vector has zero net COM translation
        mode_cart = project_out_com_translation(imag_mode_vector, masses_arr)
        mode_mw = mode_cart * mw
        mode_norm = float(np.linalg.norm(mode_mw))
        if mode_norm < 1e-12:
            raise ValueError("imag_mode_vector has zero non-translational norm.")
        mode_mw /= mode_norm

        def _step_direction(direction_sign: float) -> list[np.ndarray]:
            # Initial step in mass-weighted coordinates mapped back to Cartesian via M^{-1/2}
            coords = ts_c.copy() + direction_sign * self.step_size * (mode_mw * inv_mw)
            path = [coords.copy()]
            cur_step = self.step_size

            for step in range(max_steps):
                try:
                    e_curr, grad = grad_fn(coords)
                except Exception as e:
                    self.logger.error(f"Gradient evaluation failed at step {step}: {e}")
                    raise RuntimeError(f"Gradient evaluation failed at step {step}") from e

                grad_arr = np.asarray(grad, dtype=np.float64)
                # Remove numerical (e.g. Float32 MLFF) net translational force residual so sum_i g_i = 0 strictly
                grad_arr = grad_arr - np.mean(grad_arr, axis=0, keepdims=True)
                grad_mw = grad_arr * inv_mw
                g_norm = float(np.linalg.norm(grad_mw))

                if g_norm < self.grad_tol:
                    self.logger.info(f"IRC path step converged to minimum at step {step} (||g_mw||={g_norm:.3e}).")
                    break

                # Heun Predictor-Corrector in mass-weighted coordinates, mapped via M^{-1/2}
                accepted = False
                for _ in range(8):
                    dq_pred = -(cur_step / max(g_norm, 1e-12)) * grad_mw
                    coords_pred = coords + dq_pred * inv_mw

                    try:
                        _, grad_pred = grad_fn(coords_pred)
                    except Exception as e:
                        self.logger.error(f"Gradient predictor evaluation failed at step {step}: {e}")
                        raise RuntimeError(f"Gradient predictor evaluation failed at step {step}") from e

                    grad_pred_arr = np.asarray(grad_pred, dtype=np.float64)
                    grad_pred_arr = grad_pred_arr - np.mean(grad_pred_arr, axis=0, keepdims=True)
                    grad_mw_pred = grad_pred_arr * inv_mw
                    grad_avg = 0.5 * (grad_mw + grad_mw_pred)
                    g_avg_norm = float(np.linalg.norm(grad_avg))
                    if g_avg_norm < 1e-12:
                        accepted = True
                        break

                    dq_corr = -(cur_step / g_avg_norm) * grad_avg
                    coords_cand = coords + dq_corr * inv_mw
                    e_cand, _ = grad_fn(coords_cand)

                    if float(e_cand) <= float(e_curr) + 1e-4:
                        coords = coords_cand
                        path.append(coords.copy())
                        cur_step = min(self.step_size, cur_step * 1.25)
                        accepted = True
                        break
                    cur_step *= 0.5

                if not accepted:
                    self.logger.info(f"IRC step damped at minimum vicinity at step {step}.")
                    break

            return path

        self.logger.info("Tracing forward IRC path...")
        forward_path = _step_direction(+1.0)
        self.logger.info("Tracing reverse IRC path...")
        reverse_path = _step_direction(-1.0)

        # Assemble full trajectory: reverse (reversed) + TS + forward
        full_path = reverse_path[::-1] + [ts_c.copy()] + forward_path
        path_array = np.array(full_path, dtype=np.float64)

        # Compute energies along path
        energies_array = np.zeros(len(path_array), dtype=np.float64)
        for idx in range(len(path_array)):
            try:
                energy, _ = grad_fn(path_array[idx])
                energies_array[idx] = float(energy)
            except Exception as e:
                self.logger.error(f"Failed to compute energy along path at index {idx}: {e}")
                raise RuntimeError(f"Energy computation failed at index {idx}") from e

        self.logger.info(f"IRC path tracing complete ({len(path_array)} path images).")
        return path_array, energies_array

    def verify_endpoints(
        self,
        symbols: list[str],
        ts_coords: np.ndarray,
        masses: np.ndarray,
        imag_mode_vector: np.ndarray,
        grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
        ref_a_coords: np.ndarray,
        ref_b_coords: np.ndarray,
        relax_fn: Callable[[np.ndarray], tuple[np.ndarray, float]] | None = None,
        max_steps: int = 80,
        rmsd_tol: float = 0.15,
        covalent_scale: float = 1.25,
    ) -> dict[str, Any]:
        """
        Runs bidirectional mass-weighted IRC, optionally relaxes both endpoints to
        their local minima, and verifies heavy-atom RMSD and covalent bond-graph
        connectivity against reference stationary points (ref_a, ref_b).
        """
        path_coords, path_energies = self.trace_irc_path(
            ts_coords=ts_coords,
            masses=masses,
            imag_mode_vector=imag_mode_vector,
            grad_fn=grad_fn,
            max_steps=max_steps,
        )

        ts_com = compute_center_of_mass(ts_coords, masses)
        com_drifts = [
            float(np.linalg.norm(compute_center_of_mass(frame, masses) - ts_com))
            for frame in path_coords
        ]
        max_com_drift = max(com_drifts) if com_drifts else 0.0

        rev_end = path_coords[0].copy()
        fwd_end = path_coords[-1].copy()
        rev_energy = float(path_energies[0])
        fwd_energy = float(path_energies[-1])

        if relax_fn is not None:
            rev_end, rev_energy = relax_fn(rev_end)
            fwd_end, fwd_energy = relax_fn(fwd_end)

        # Evaluate both pairings: (rev->A, fwd->B) vs (rev->B, fwd->A)
        rmsd_ra = heavy_atom_rmsd(symbols, rev_end, ref_a_coords)
        rmsd_fb = heavy_atom_rmsd(symbols, fwd_end, ref_b_coords)
        rmsd_rb = heavy_atom_rmsd(symbols, rev_end, ref_b_coords)
        rmsd_fa = heavy_atom_rmsd(symbols, fwd_end, ref_a_coords)

        if (rmsd_ra + rmsd_fb) <= (rmsd_rb + rmsd_fa):
            orientation = "reverse->A, forward->B"
            end_a, end_b = rev_end, fwd_end
            rmsd_a, rmsd_b = rmsd_ra, rmsd_fb
        else:
            orientation = "forward->A, reverse->B"
            end_a, end_b = fwd_end, rev_end
            rmsd_a, rmsd_b = rmsd_fa, rmsd_rb

        graph_ref_a = build_connectivity_graph(symbols, ref_a_coords, covalent_scale=covalent_scale)
        graph_ref_b = build_connectivity_graph(symbols, ref_b_coords, covalent_scale=covalent_scale)
        graph_end_a = build_connectivity_graph(symbols, end_a, covalent_scale=covalent_scale)
        graph_end_b = build_connectivity_graph(symbols, end_b, covalent_scale=covalent_scale)

        graph_match_a = graph_end_a == graph_ref_a
        graph_match_b = graph_end_b == graph_ref_b

        connected = (rmsd_a < rmsd_tol and rmsd_b < rmsd_tol) or (
            graph_match_a and graph_match_b and rmsd_a < rmsd_tol and rmsd_b < rmsd_tol
        )
        # Strict check: both heavy-atom RMSD < rmsd_tol AND bond graph match
        strict_connected = (rmsd_a < rmsd_tol) and (rmsd_b < rmsd_tol) and graph_match_a and graph_match_b

        return {
            "path_coords": path_coords,
            "path_energies": path_energies,
            "n_images": int(len(path_coords)),
            "max_com_drift_ang": float(max_com_drift),
            "orientation": orientation,
            "endpoint_a_coords": end_a,
            "endpoint_b_coords": end_b,
            "reverse_relaxed_energy": float(rev_energy),
            "forward_relaxed_energy": float(fwd_energy),
            "rmsd_a_ang": float(rmsd_a),
            "rmsd_b_ang": float(rmsd_b),
            "bond_graph_match_a": bool(graph_match_a),
            "bond_graph_match_b": bool(graph_match_b),
            "connected_rmsd": bool(connected),
            "connected_strict": bool(strict_connected),
        }
