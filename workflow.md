# CoChem-KINETIC: Software Engineering Specification
**Target Phase:** Python Implementation

This document serves as the exact coding blueprint for the next LLM agent to construct the `CoChem-KINETIC` repository.

## 1. Directory & File Architecture
```text
CoChem-KINETIC/
├── kinetic_core/
│   ├── __init__.py
│   ├── dispatcher.py      # Entry point for BASE payload ingestion
│   ├── mace_pre_opt.py    # Active learning neural network NEB sweeps
│   ├── jax_cineb.py       # CI-NEB implementation using JAX
│   ├── thermo.py          # Wigner corrections and free energy mapping
│   └── aimd_hoover.py     # Nose-Hoover thermal sampling
├── tests/
│   ├── test_wigner.py
│   └── test_cineb.py
├── requirements.txt       # jax, mace, h5py, numpy, scipy
└── README.md
```

## 2. File-by-File Blueprint

### `kinetic_core/jax_cineb.py`
- **Purpose:** Computes the highest-energy image mathematically.
- **Functions:**
  - `def climb_image(energies: np.ndarray, gradients: np.ndarray) -> np.ndarray:`
    - *Returns:* The isolated tangential force array pushing the highest image toward the strict saddle point.

### `kinetic_core/thermo.py`
- **Purpose:** Applies physical thermodynamic corrections.
- **Functions:**
  - `def wigner_correction(imaginary_freq: float, temp: float = 298.15) -> float:`
    - *Returns:* The unitless Wigner transmission multiplier. If $freq < 0$ (real), return $1.0$.
  - `def calculate_eyring_rate(delta_g: float, wigner_coeff: float) -> float:`
    - *Returns:* The rate constant $k$.

### `kinetic_core/aimd_hoover.py`
- **Purpose:** Executes canonical sampling.
- **Functions:**
  - `def nvt_step(coords: np.ndarray, momenta: np.ndarray, forces: np.ndarray, dt: float) -> tuple:`
    - *Returns:* Updated coordinates and momenta using the Nose-Hoover chain algorithm.

## 3. Execution Data Flow (The Payload Trace)
1. **Payload Ingest:** `dispatcher.py` reads `SMILES` for Reactants and Products.
2. **Pre-Sweep:** `mace_pre_opt.py` interpolates 11 images and performs an instant 10,000-step MLFF relaxation to pull the string tight.
3. **Saddle Optimization:** ORCA executes CI-NEB on the highest image. `jax_cineb.py` processes the forces.
4. **Frequency Check:** Parses the ORCA frequency job; throws an error if exactly one imaginary frequency isn't found.
5. **Thermodynamics:** Extracts the ZPE. Passes the imaginary frequency to `thermo.wigner_correction()`.
6. **Serialization:** Writes the kinetic rate constant and activation energy to `/kinetic/thermo/`. Streams the IRC pathway structures to `/kinetic/irc_path/`.

## 4. PyTest Roadmap
- **Test 1 (`test_wigner.py`):** Assert that an imaginary frequency of $1500$ cm$^{-1}$ at $298.15$ K returns the analytically expected transmission multiplier.
- **Test 2 (`test_cineb.py`):** Provide a mock 1D potential energy surface. Assert that `climb_image` reverses the force direction exactly at the peak to lock onto the saddle point.
