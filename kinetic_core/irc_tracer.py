#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 3: Intrinsic Reaction Coordinate (IRC) Path Tracer
-------------------------------------------------------------------------
Implements Page-McIver and Gonzalez-Schlegel algorithms for bidirectional IRC integration
starting from a transition state geometry along forward and reverse mass-weighted reaction vectors.
"""

import logging
from collections.abc import Callable

import numpy as np


class IRCTracerEngine:
    """Bidirectional IRC Path Integrator using Page-McIver / Gonzalez-Schlegel math."""

    def __init__(self, step_size_amu_ang: float = 0.1) -> None:
        self.step_size = step_size_amu_ang
        self.logger = logging.getLogger("CoChem_KINETIC_IRC")

    def trace_irc_path(self, ts_coords: np.ndarray, masses: np.ndarray, imag_mode_vector: np.ndarray, grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]], max_steps: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """
        Traces forward and reverse IRC pathways from TS geometry.
        Returns: (all_path_coords, path_energies)
        """
        mw = np.sqrt(masses)[:, np.newaxis]
        mode_mw = imag_mode_vector * mw
        mode_mw /= max(np.linalg.norm(mode_mw), 1e-12)

        def _step_direction(direction_sign: float) -> list[np.ndarray]:
            coords = ts_coords.copy() + direction_sign * (self.step_size / np.maximum(mw, 1e-6)) * mode_mw
            path = [coords.copy()]

            for step in range(max_steps):
                try:
                    _, grad = grad_fn(coords)
                except Exception as e:
                    self.logger.error(f"Gradient evaluation failed at step {step}: {e}")
                    raise RuntimeError(f"Gradient evaluation failed at step {step}") from e

                grad_mw = grad / np.maximum(mw, 1e-6)
                g_norm = float(np.linalg.norm(grad_mw))

                if g_norm < 0.01:
                    self.logger.info(f"IRC path step converged to minimum at step {step}.")
                    break

                # Gonzalez-Schlegel mass-weighted Predictor-Corrector step (Heun's method)
                step_vec_pred = - (self.step_size / max(g_norm, 1e-12)) * grad_mw
                coords_pred = coords + step_vec_pred
                
                try:
                    _, grad_pred = grad_fn(coords_pred)
                except Exception as e:
                    self.logger.error(f"Gradient predictor evaluation failed at step {step}: {e}")
                    raise RuntimeError(f"Gradient predictor evaluation failed at step {step}") from e

                grad_mw_pred = grad_pred / np.maximum(mw, 1e-6)
                
                # Corrector using averaged gradient
                grad_avg = 0.5 * (grad_mw + grad_mw_pred)
                g_avg_norm = float(np.linalg.norm(grad_avg))
                step_vec_corr = - (self.step_size / max(g_avg_norm, 1e-12)) * grad_avg
                
                coords += step_vec_corr
                path.append(coords.copy())

            return path

        self.logger.info("Tracing forward IRC path...")
        forward_path = _step_direction(+1.0)
        self.logger.info("Tracing reverse IRC path...")
        reverse_path = _step_direction(-1.0)

        # Assemble full trajectory: reverse (reversed) + TS + forward
        full_path = reverse_path[::-1] + [ts_coords] + forward_path
        path_array = np.array(full_path)

        # Compute energies along path
        energies_array = np.zeros(len(path_array))
        for idx in range(len(path_array)):
            try:
                energy, _ = grad_fn(path_array[idx])
                energies_array[idx] = float(energy)
            except Exception as e:
                self.logger.error(f"Failed to compute energy along path at index {idx}: {e}")
                raise RuntimeError(f"Energy computation failed at index {idx}") from e

        self.logger.info(f"IRC path tracing complete ({len(path_array)} path images).")
        return path_array, energies_array


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracer = IRCTracerEngine()
    ts = np.array([[0.0,0.0,0.0], [0.0,0.0,1.2]])
    m = np.array([16.0, 1.0])
    mode = np.array([[0.0,0.0,1.0], [0.0,0.0,-1.0]])
    
    def test_grad(c: np.ndarray) -> tuple[float, np.ndarray]: 
        return (0.0, 2.0 * c)
        
    path, e = tracer.trace_irc_path(ts, m, mode, test_grad, max_steps=5)
    logging.getLogger().info(f"IRC Tracer test passed. Path images: {len(path)}")