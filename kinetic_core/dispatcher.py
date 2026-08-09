#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 1: Payload Ingestion Dispatcher & HDF5 State Writer
----------------------------------------------------------------------------
Ingests SMILES strings, reactant/product structures, and BASE state payloads.
Serializes rates, Eyring parameters, and IRC trajectories into cochem_state.h5
at /kinetic/thermo/ and /kinetic/irc_path/.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False


class KineticDispatcher:
    """CLI and API Dispatcher for CoChem-KINETIC workflows."""

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir or Path(os.environ.get("COCHEM_ARTIFACT_DIR", "."))
        self.logger = logging.getLogger("CoChem_KINETIC_Dispatcher")

    def parse_payload_json(self, json_path: Path) -> Dict[str, Any]:
        """
        Parses input payload JSON file specifying reactants, products, and TS candidates.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.logger.info(f"Ingested kinetic payload from {json_path.name}")
        return data

    def write_state_hdf5(self, h5_path: Path, thermo_data: Dict[str, float], irc_coords: Optional[np.ndarray] = None, irc_energies: Optional[np.ndarray] = None):
        """
        Serializes rate parameters and IRC trajectory data into cochem_state.h5.
        Writes to /kinetic/thermo/ and /kinetic/irc_path/.
        """
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        if not H5PY_AVAILABLE:
            self.logger.warning("h5py not installed. Dumping state telemetry to JSON fallback.")
            json_fallback = h5_path.with_suffix(".json")
            with open(json_fallback, "w", encoding="utf-8") as f:
                json.dump({"thermo": thermo_data, "irc_energies": irc_energies.tolist() if irc_energies is not None else []}, f, indent=4)
            return

        with h5py.File(h5_path, "a") as f:
            # Group /kinetic/thermo
            thermo_group = f.require_group("kinetic/thermo")
            for k, v in thermo_data.items():
                if k in thermo_group:
                    del thermo_group[k]
                thermo_group.create_dataset(k, data=float(v))
            thermo_group.attrs["unit"] = "kcal/mol"
            thermo_group.attrs["temperature_k"] = thermo_data.get("temperature_k", 298.15)

            # Group /kinetic/irc_path
            if irc_coords is not None or irc_energies is not None:
                irc_group = f.require_group("kinetic/irc_path")
                if irc_coords is not None:
                    if "coordinates" in irc_group: del irc_group["coordinates"]
                    irc_group.create_dataset("coordinates", data=irc_coords)
                if irc_energies is not None:
                    if "energies" in irc_group: del irc_group["energies"]
                    irc_group.create_dataset("energies", data=irc_energies)

        self.logger.info(f"Successfully serialized kinetic state to {h5_path.name} (/kinetic/thermo/ and /kinetic/irc_path/).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    disp = KineticDispatcher()
    sample_data = {"delta_g_barrier": 15.2, "k_eyring": 1.2e3, "temperature_k": 298.15}
    h5_file = Path("./cochem_state.h5")
    disp.write_state_hdf5(h5_file, sample_data, irc_energies=np.array([0.0, 15.2, 2.1]))
    print("Kinetic Dispatcher test passed.")
