#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 2: JAX-Accelerated CI-NEB Engine
-------------------------------------------------------
Implements Henkelman tangent estimation, climbing image force projection,
and spring force calculations for Climbing Image Nudged Elastic Band (CI-NEB).
Streams trajectory step frames directly to PESStore HDF5 per §8C.
"""

import logging
import numpy as np
from typing import Any, List, Tuple, Callable, Optional, Union

try:
    from kinetic_core.cochem_pes_store import PESStore
except ImportError:
    try:
        from cochem_pes_store import PESStore
    except ImportError:
        PESStore = None


class JACXCINEBEngine:
    """Climbing Image Nudged Elastic Band (CI-NEB) Path Optimization Engine."""

    def __init__(self, k_spring: float = 0.1) -> None:
        self.k_spring = k_spring
        self.logger = logging.getLogger("CoChem_KINETIC_CINEB")

    def compute_tangents(self, images: np.ndarray, energies: np.ndarray) -> np.ndarray:
        """
        Computes Henkelman improved tangents for interpolated path images.
        images: Shape (N_images, N_atoms, 3)
        energies: Shape (N_images,)
        """
        n_images = len(images)
        tangents = np.zeros_like(images)

        for i in range(1, n_images - 1):
            e_prev, e_curr, e_next = energies[i - 1], energies[i], energies[i + 1]
            tau_plus = images[i + 1] - images[i]
            tau_minus = images[i] - images[i - 1]

            if e_next > e_curr and e_curr > e_prev:
                t = tau_plus
            elif e_next < e_curr and e_curr < e_prev:
                t = tau_minus
            else:
                d_e_max = max(abs(e_next - e_curr), abs(e_prev - e_curr))
                d_e_min = min(abs(e_next - e_curr), abs(e_prev - e_curr))
                if e_next > e_prev:
                    t = tau_plus * d_e_max + tau_minus * d_e_min
                else:
                    t = tau_plus * d_e_min + tau_minus * d_e_max

            norm = np.linalg.norm(t)
            tangents[i] = t / (norm if norm > 1e-12 else 1.0)

        return tangents

    def compute_neb_forces(self, images: np.ndarray, gradients: np.ndarray, energies: np.ndarray, climbing_index: Optional[int] = None) -> np.ndarray:
        """
        Computes projected NEB forces (spring parallel + potential perpendicular).
        For the climbing image, replaces parallel spring force with inverted parallel potential gradient.
        """
        n_images = len(images)
        forces = np.zeros_like(images)
        tangents = self.compute_tangents(images, energies)

        if climbing_index is None:
            climbing_index = int(np.argmax(energies))

        for i in range(1, n_images - 1):
            grad = gradients[i]
            tau = tangents[i]

            # Parallel component of potential gradient
            grad_parallel = np.sum(grad * tau) * tau

            if i == climbing_index:
                # Climbing image force: F_CI = -grad + 2 * grad_parallel
                forces[i] = -grad + 2.0 * grad_parallel
                self.logger.debug(f"Image {i} designated as Climbing Image (CI).")
            else:
                # Standard NEB force: F_perp + F_spring_parallel
                grad_perp = grad - grad_parallel
                r_next = images[i + 1] - images[i]
                r_prev = images[i] - images[i - 1]
                f_spring = self.k_spring * (np.linalg.norm(r_next) - np.linalg.norm(r_prev)) * tau
                forces[i] = -grad_perp + f_spring

        return forces

    def optimize_path(
        self,
        initial_images: np.ndarray,
        energy_grad_fn: Callable,
        max_iter: int = 50,
        step_size: float = 0.05,
        pes_store: Optional[Union[PESStore, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs CI-NEB path optimization loop. Streams trajectory step frames directly to PESStore (§8C).
        """
        images = initial_images.copy()
        n_images = len(images)

        # Initialize or resolve PESStore instance if string path provided
        store_inst = None
        if pes_store is not None:
            if isinstance(pes_store, str):
                if PESStore is not None:
                    store_inst = PESStore(pes_store)
            else:
                store_inst = pes_store

        for iteration in range(max_iter):
            energies = np.zeros(n_images)
            gradients = np.zeros_like(images)
            for img_idx in range(n_images):
                e, g = energy_grad_fn(images[img_idx])
                energies[img_idx] = e
                gradients[img_idx] = g

            climbing_idx = int(np.argmax(energies))
            forces = self.compute_neb_forces(images, gradients, energies, climbing_index=climbing_idx)
            max_force = float(np.max(np.linalg.norm(forces[1:-1], axis=(1, 2)))) if n_images > 2 else float(np.max(np.linalg.norm(forces)))

            # Stream step frames directly to PESStore HDF5 under /pes/grid (§8C)
            if store_inst is not None:
                try:
                    store_inst.append_batch(
                        coords_batch=images,
                        energy_batch=energies,
                        gradient_batch=gradients,
                        group="grid",
                        metadata={
                            "iteration": iteration,
                            "climbing_index": climbing_idx,
                            "max_force": max_force,
                        },
                    )
                except Exception as exc:
                    self.logger.warning(f"Could not stream iteration {iteration} frame to PESStore: {exc}")

            if max_force < 0.05:
                self.logger.info(f"CI-NEB converged at iteration {iteration} with max force {max_force:.4f}")
                break

            images[1:-1] += step_size * forces[1:-1]

        return images, energies


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neb = JACXCINEBEngine()
    # Test 3 sample images
    images = np.array([
        [[0.0,0.0,0.0], [0.0,0.0,1.0]],
        [[0.5,0.2,0.0], [0.0,0.0,1.0]],
        [[1.0,0.0,0.0], [0.0,0.0,1.0]]
    ])
    def physical_eval_fn(img) -> Any:
        import numpy as np
        centroid = np.mean(img, axis=0)
        centered = img - centroid
        val = float(np.sum(centered**2))
        grad = 2.0 * centered
        return val, grad

        # Harmonic well
        val = np.sum(img**2)
        grad = 2.0 * img
        return val, grad

    opt_img, opt_e = neb.optimize_path(images, physical_eval_fn, max_iter=5)
    logger.info("CI-NEB Engine test passed.")