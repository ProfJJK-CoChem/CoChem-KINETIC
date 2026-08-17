#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 2: JAX-Accelerated CI-NEB Engine
-------------------------------------------------------
Implements Henkelman tangent estimation, climbing image force projection,
and spring force calculations for Climbing Image Nudged Elastic Band (CI-NEB).
Streams trajectory step frames directly to PESStore HDF5 per §8C.
"""
import logging
import os
from collections.abc import Callable
from typing import Any
os.environ["JAX_ENABLE_X64"] = "True"
try:
    import jax.numpy as jnp
    np = jnp
    HAS_JAX = True
except ImportError:
    import numpy as np
    HAS_JAX = False

try:
    from kinetic_core.cochem_pes_store import PESStore
except ImportError:
    try:
        from cochem_pes_store import PESStore
    except ImportError:
        PESStore = 'PESStore' # or Any, since we just use it in typing

def _set(arr: Any, idx: Any, val: Any) -> Any:
    if HAS_JAX and hasattr(arr, 'at'):
        return arr.at[idx].set(val)
    arr[idx] = val
    return arr

def _add(arr: Any, idx: Any, val: Any) -> Any:
    if HAS_JAX and hasattr(arr, 'at'):
        return arr.at[idx].add(val)
    arr[idx] += val
    return arr


class JACXCINEBEngine:
    """Climbing Image Nudged Elastic Band (CI-NEB) Path Optimization Engine."""

    def __init__(self, k_spring: float = 0.1) -> None:
        self.k_spring = k_spring
        self.logger = logging.getLogger("CoChem_KINETIC_CINEB")

    def compute_tangents(self, images: Any, energies: Any) -> Any:
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
            tangents = _set(tangents, i, t / (norm if norm > 1e-12 else 1.0))

        return tangents

    def compute_neb_forces(self, images: Any, gradients: Any, energies: Any, climbing_index: int | None = None) -> Any:
        n_images = len(images)
        forces = np.zeros_like(images)
        tangents = self.compute_tangents(images, energies)

        if climbing_index is None:
            climbing_index = int(np.argmax(energies))

        for i in range(1, n_images - 1):
            grad = gradients[i]
            tau = tangents[i]
            grad_parallel = np.sum(grad * tau) * tau

            if i == climbing_index:
                forces = _set(forces, i, -grad + 2.0 * grad_parallel)
                self.logger.debug(f"Image {i} designated as CI.")
            else:
                grad_perp = grad - grad_parallel
                r_next = images[i + 1] - images[i]
                r_prev = images[i] - images[i - 1]
                f_spring = self.k_spring * (np.linalg.norm(r_next) - np.linalg.norm(r_prev)) * tau
                forces = _set(forces, i, -grad_perp + f_spring)

        return forces

    def optimize_path(
        self,
        initial_images: Any,
        energy_grad_fn: Callable[..., tuple[float, Any]],
        max_iter: int = 50,
        step_size: float = 0.05,
        pes_store: Any | str | None = None,
    ) -> tuple[Any, Any]:
        images = initial_images if HAS_JAX else initial_images.copy()
        n_images = len(images)
        store_inst = None
        if pes_store is not None:
            if isinstance(pes_store, str):
                if PESStore is not None:
                    store_inst = PESStore(pes_store)
            else:
                store_inst = pes_store

        # FIRE parameters
        dt = step_size
        dt_max = step_size * 10.0
        dt_min = step_size * 0.1
        f_inc = 1.1
        f_dec = 0.5
        f_a = 0.99
        a_start = 0.1
        a = a_start
        N_steps = 0
        v = np.zeros_like(images)

        for iteration in range(max_iter):
            energies_list = []
            gradients_list = []
            for img_idx in range(n_images):
                e, g = energy_grad_fn(images[img_idx])
                energies_list.append(e)
                gradients_list.append(g)
            
            energies = np.array(energies_list)
            gradients = np.array(gradients_list)

            climbing_idx = int(np.argmax(energies))
            forces = self.compute_neb_forces(images, gradients, energies, climbing_index=climbing_idx)
            
            if n_images > 2:
                force_norms = np.linalg.norm(forces[1:-1], axis=(1, 2)) if forces.ndim == 3 else np.linalg.norm(forces[1:-1], axis=-1)
                max_force = float(np.max(force_norms))
            else:
                max_force = float(np.max(np.linalg.norm(forces)))

            if store_inst is not None:
                try:
                    store_inst.append_batch(
                        coords_batch=images if not HAS_JAX else np.array(images),
                        energy_batch=energies if not HAS_JAX else np.array(energies),
                        gradient_batch=gradients if not HAS_JAX else np.array(gradients),
                        group="grid",
                        metadata={"iteration": iteration, "climbing_index": climbing_idx, "max_force": max_force},
                    )
                except (OSError, ValueError) as exc:
                    self.logger.warning(f"Store streaming failed: {exc}")

            if max_force < 0.05:
                self.logger.info(f"CI-NEB FIRE converged at iteration {iteration} with max force {max_force:.4f}")
                break

            # FIRE Step
            P = float(np.sum(forces[1:-1] * v[1:-1]))
            if P > 0:
                v_norm = np.linalg.norm(v[1:-1])
                f_norm = np.linalg.norm(forces[1:-1])
                if f_norm > 1e-12:
                    v_dir = forces[1:-1] / f_norm
                    v = _set(v, slice(1, -1), (1 - a) * v[1:-1] + a * v_norm * v_dir)
                N_steps += 1
                if N_steps > 5:
                    dt = min(dt * f_inc, dt_max)
                    a = a * f_a
            else:
                v = _set(v, slice(1, -1), 0.0)
                a = a_start
                dt = max(dt * f_dec, dt_min)
                N_steps = 0

            # Velocity Verlet Integration
            v = _add(v, slice(1, -1), dt * forces[1:-1])
            images = _add(images, slice(1, -1), dt * v[1:-1])

        return images, energies

