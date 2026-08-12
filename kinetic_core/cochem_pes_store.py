import hashlib
#!/usr/bin/env python3
"""
CoChem-KINETIC - PES Store HDF5 Class Interface
------------------------------------------------
Implements PESStore HDF5 class interface conforming to Method Matrix v4 §8C.
Stores PES grid points, fitted surfaces, and surrogate epistemic uncertainties
under /pes/grid, /pes/fit, and /pes/uncertainty with:
  - 512-point chunking
  - gzip level 4 compression
  - byte shuffle filter
  - Fletcher32 checksums
  - qcschema_version = 2
  - provenance_tag = '[M]' (or custom)
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False


class PESStore:
    """
    HDF5 PES Store interface for managing Potential Energy Surface (PES) grids,
    fitted surrogate models, and epistemic uncertainties (§8C).
    """

    _lock = threading.Lock()

    def __init__(self, filename: Union[str, Path], mode: str = "a", provenance_tag: str = "[M]") -> None:
        self.filename = Path(filename)
        self.mode = mode
        self.provenance_tag = provenance_tag
        self.logger = logging.getLogger("CoChem_KINETIC_PESStore")

        if not H5PY_AVAILABLE:
            raise RuntimeError("h5py library is required to initialize PESStore HDF5 interface.")

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._init_groups()

    def _init_groups(self) -> Any:
        """Initializes standard group structure and metadata attributes per §8C."""
        if self.mode != "r":
            with self._lock:
                with h5py.File(self.filename, self.mode) as f:
                    for grp_name in ["pes/grid", "pes/fit", "pes/uncertainty"]:
                        grp = f.require_group(grp_name)
                        grp.attrs["qcschema_version"] = 2
                        grp.attrs["provenance_tag"] = self.provenance_tag

    def _get_or_create_dataset(
        self,
        group: h5py.Group,
        name: str,
        sample_data: np.ndarray,
        chunk_size_dim0: int = 512,
    ) -> h5py.Dataset:
        """Helper to retrieve or create resizable 512-chunked gzip-level-4 Fletcher32 dataset."""
        sample_arr = np.asarray(sample_data, dtype=np.float64)
        item_shape = sample_arr.shape[1:]

        if name in group:
            return group[name]

        shape = (0, *item_shape)
        maxshape = (None, *item_shape)
        chunks = (chunk_size_dim0, *item_shape)

        try:
            return group.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                dtype=np.float64,
            )
        except (ValueError, RuntimeError):
            if name in group:
                return group[name]
            raise

    def append_batch(
        self,
        coords_batch: np.ndarray,
        energy_batch: np.ndarray,
        gradient_batch: Optional[np.ndarray] = None,
        variance_batch: Optional[np.ndarray] = None,
        group: str = "grid",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Appends a batch of points to the specified PES group ('grid', 'fit', or 'uncertainty').
        """
        coords_arr = np.asarray(coords_batch, dtype=np.float64)
        energy_arr = np.asarray(energy_batch, dtype=np.float64)

        if coords_arr.ndim == 1:
            coords_arr = coords_arr[np.newaxis, ...]
        if energy_arr.ndim == 0:
            energy_arr = energy_arr[np.newaxis]

        n_points = len(coords_arr)

        target_group_path = f"pes/{group}" if not group.startswith("pes/") else group

        with self._lock:
            with h5py.File(self.filename, "a") as f:
                grp = f.require_group(target_group_path)
                grp.attrs["qcschema_version"] = 2
                grp.attrs["provenance_tag"] = self.provenance_tag

                if metadata:
                    for k, v in metadata.items():
                        try:
                            grp.attrs[k] = v
                        except Exception:
                            grp.attrs[k] = str(v)

                # Coordinates
                ds_coords = self._get_or_create_dataset(grp, "coordinates", coords_arr)
                curr_len = ds_coords.shape[0]
                new_shape = (curr_len + n_points, *coords_arr.shape[1:])
                ds_coords.resize(new_shape)
                ds_coords[curr_len:] = coords_arr

                # Energies
                ds_name_e = "fit_energies" if group == "fit" else "energies"
                ds_energies = self._get_or_create_dataset(grp, ds_name_e, energy_arr)
                curr_len_e = ds_energies.shape[0]
                new_shape_e = (curr_len_e + n_points, *energy_arr.shape[1:])
                ds_energies.resize(new_shape_e)
                ds_energies[curr_len_e:] = energy_arr

                # Gradients
                if gradient_batch is not None:
                    grad_arr = np.asarray(gradient_batch, dtype=np.float64)
                    if grad_arr.ndim == 1 and coords_arr.ndim > 1:
                        grad_arr = grad_arr[np.newaxis, ...]
                    ds_grads = self._get_or_create_dataset(grp, "gradients", grad_arr)
                    curr_len_g = ds_grads.shape[0]
                    new_shape_g = (curr_len_g + n_points, *grad_arr.shape[1:])
                    ds_grads.resize(new_shape_g)
                    ds_grads[curr_len_g:] = grad_arr

                # Variances
                if variance_batch is not None:
                    var_arr = np.asarray(variance_batch, dtype=np.float64)
                    if var_arr.ndim == 0:
                        var_arr = var_arr[np.newaxis]
                    ds_var = self._get_or_create_dataset(grp, "variance", var_arr)
                    curr_len_v = ds_var.shape[0]
                    new_shape_v = (curr_len_v + n_points, *var_arr.shape[1:])
                    ds_var.resize(new_shape_v)
                    ds_var[curr_len_v:] = var_arr

        self.logger.debug(f"Appended {n_points} points to PESStore target group '{target_group_path}'.")

    def append_point(
        self,
        coords: np.ndarray,
        energy: float,
        gradient: Optional[np.ndarray] = None,
        variance: Optional[float] = None,
        group: str = "grid",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Appends a single PES data point."""
        c_batch = coords[np.newaxis, ...] if coords.ndim > 0 else coords
        e_batch = np.array([energy], dtype=np.float64)
        g_batch = gradient[np.newaxis, ...] if gradient is not None else None
        v_batch = np.array([variance], dtype=np.float64) if variance is not None else None
        self.append_batch(c_batch, e_batch, g_batch, v_batch, group=group, metadata=metadata)

    def get_grid(self) -> Dict[str, np.ndarray]:
        """Retrieves grid datasets from /pes/grid."""
        return self._read_group("pes/grid")

    def get_fit(self) -> Dict[str, np.ndarray]:
        """Retrieves fitted datasets from /pes/fit."""
        return self._read_group("pes/fit")

    def get_uncertainty(self) -> Dict[str, np.ndarray]:
        """Retrieves epistemic uncertainty datasets from /pes/uncertainty."""
        return self._read_group("pes/uncertainty")

    def _read_group(self, group_path: str) -> Dict[str, np.ndarray]:
        """Reads all datasets in an HDF5 group into a dictionary of NumPy arrays."""
        result = {}
        with self._lock:
            with h5py.File(self.filename, "r") as f:
                if group_path in f:
                    grp = f[group_path]
                    for key in grp.keys():
                        if isinstance(grp[key], h5py.Dataset):
                            result[key] = grp[key][:]
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = PESStore("test_pes_store.h5")
    store.append_point(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), energy=-76.2, gradient=np.zeros((2, 3)))
    grid = store.get_grid()
    logger.info(f"PESStore test passed. Grid shape: {grid['coordinates'].shape}")
    import os
    os.remove("test_pes_store.h5")
def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()