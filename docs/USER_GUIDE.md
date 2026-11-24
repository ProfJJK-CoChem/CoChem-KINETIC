# CoChem-KINETIC User Guide

This guide documents the theoretical engines, Python APIs, CHEM 312L reaction-profile CLI (`rxn_profile.py`), ecosystem handoffs (including `CoChem-TOPOS`, `CoChem-TORQ`, and `CoChem-SPYFY`), and provenance/warning guarantees in **CoChem-KINETIC v5.0.0**.

---

## Table of Contents

1. [Architecture & Ecosystem Context](#1-architecture--ecosystem-context)
2. [Microkinetic Modeling & Eyring TST](#2-microkinetic-modeling--eyring-tst)
3. [Quantum Tunneling (Eckart, Wigner & Skodje–Truhlar)](#3-quantum-tunneling-eckart-wigner--skodjetruhlar)
4. [Smoluchowski Diffusion Limits](#4-smoluchowski-diffusion-limits)
5. [Microcanonical RRKM & Master Equation](#5-microcanonical-rrkm--master-equation)
6. [CHEM 312L Reaction-Profile Pipeline (`rxn_profile.py`)](#6-chem-312l-reaction-profile-pipeline-rxn_profilepy)
7. [Interpreting `[M]` vs `[E]` Provenance Tags & Fallback Warnings](#7-interpreting-m-vs-e-provenance-tags--fallback-warnings)

---

## 1. Architecture & Ecosystem Context

`CoChem-KINETIC` is organized into two cooperating packages:

- **`cochem_kinetic/`** — Modern Pipeline 5.0.0 public package:
  - `cochem_kinetic.cli` / `cochem_kinetic.rxn_profile` / `cochem_kinetic.chem312_specs`: Charge-aware reaction-profile CLI, stoichiometry validation, AIMNet2 CI-NEB + TS polish + Hessian + IRC, and PySCF/GPU4PySCF single-point assembly.
  - `cochem_kinetic.math.thermodynamics`: `ThermodynamicSymmetryEnforcer` (detailed balance) and `grimes_qrrho_entropy` (Grimme qRRHO entropy).
  - `cochem_kinetic.math.tunneling`: Asymmetric Eckart tunneling (`calculate_eckart_tunneling`) and Wigner tunneling (`calculate_wigner_correction`) with explicit `UserWarning` + `[E]` fallback reporting.
  - `cochem_kinetic.literature.diffusion_mapping`: `SmoluchowskiDiffusionEngine` with explicit unknown-solvent viscosity fallback warnings and provenance helpers.
  - `cochem_kinetic.engine.rrkm_theory` & `master_equation`: Exact Beyer–Swinehart microcanonical RRKM (`RRKMSolver`) and bath-gas collisional energy-transfer bounds (`MasterEquation`).
  - `cochem_kinetic.provider`: Entry-point provider (`cochem.modules:kinetic`) for `CoChem-NODE`.
- **`kinetic_core/`** — Core physical kernels and storage:
  - `kinetic_core.thermo`: Eyring TST (`calculate_eyring_rate`), variational TST (VTST), Skodje–Truhlar tunneling, Kinetic Isotope Effects (KIE), Landau–Zener crossings, and implicit solvation corrections.
  - `kinetic_core.irc_tracer`: Bidirectional mass-weighted IRC integration, Kabsch/heavy-atom RMSD, and covalent bond-graph verification.
  - `kinetic_core.cochem_pes_store`: FAIR HDF5 `PESStore` (gzip-4, shuffle, Fletcher32 checksums, QCSchema metadata).

### Handoff with `CoChem-TOPOS`, `CoChem-TORQ`, and `CoChem-SPYFY`

- **`CoChem-TOPOS`** supplies conformer ensembles and non-covalent complex geometries for reactant/intermediate/product states.
- **`CoChem-TORQ`** provides harmonic and anharmonic (VPT2) vibrational frequencies for qRRHO thermochemistry and RRKM direct state counting.
- **`CoChem-SPYFY`** performs rotational spectroscopy (FTMW / Pickett `SPFIT`/`SPCAT`) fitting for gas-phase molecules and complexes. When validating gas-phase conformer energetics or reaction barriers against `CoChem-SPYFY` experimental assignments, run `cochem-kinetic rxn-profile` with `--gas-phase-ftmw-lock` so continuum solvation is locked to vacuum (`eps = 1.0`).

---

## 2. Microkinetic Modeling & Eyring TST

### Eyring Transition State Theory

Canonical TST rate constants are computed via:

$$
k(T) = \kappa(T) \, \frac{k_B T}{h} \, \exp\!\left(-\frac{\Delta G^\ddagger(T)}{R T}\right)
$$

with ideal-gas standard-state volume conversion ($V_m = R T / P$) applied automatically for bimolecular (`reaction_order=2`) reactions in `kinetic_core.thermo.calculate_eyring_rate`:

```python
from kinetic_core.thermo import calculate_eyring_rate

k_uni = calculate_eyring_rate(
    delta_g=18.5,       # kcal/mol at 1 atm standard state
    wigner_coeff=1.25,  # tunneling factor kappa
    temp=298.15,        # K
    reaction_order=1,   # returns s^-1 (order 2 returns L mol^-1 s^-1)
)
```

### Detailed Balance & Thermodynamic Symmetry

Before evaluating coupled forward and reverse microkinetic rates, `ThermodynamicSymmetryEnforcer` verifies that both directions use identical partition-function methodologies (e.g., `RRHO` vs `HinderedRotor`), raising `DetailedBalanceError` on asymmetric setups that would violate microscopic reversibility:

```python
from cochem_kinetic.math.thermodynamics import (
    ThermodynamicSymmetryEnforcer,
    grimes_qrrho_entropy,
)

enforcer = ThermodynamicSymmetryEnforcer()
kf, kr = enforcer.compute_rates_with_symmetry_check(
    dg_forward=15.0,
    dg_reverse=22.0,
    forward_params={"method": "qRRHO"},
    reverse_params={"method": "qRRHO"},
    temp=298.15,
)

# Grimme quasi-RRHO vibrational entropy (cal mol^-1 K^-1)
S_vib = grimes_qrrho_entropy(
    frequencies_cm1=[35.0, 120.0, 480.0, 1720.0, 3050.0],
    temp=298.15,
    cutoff_freq=100.0,
)
```

Grimme's damping function $w(\nu) = [1 + (\nu_0 / \nu)^4]^{-1}$ smoothly interpolates low-frequency vibrational modes ($\nu < \nu_0 = 100\text{ cm}^{-1}$) from harmonic oscillator entropy to free-rotor entropy, preventing unphysical entropy divergence for floppy modes.

---

## 3. Quantum Tunneling (Eckart, Wigner & Skodje–Truhlar)

### Asymmetric Eckart Tunneling

`cochem_kinetic.math.tunneling.calculate_eckart_tunneling(imag_freq, temp, V_f, V_r)` integrates the exact asymmetric Eckart transmission probability $P(E)$ over a Boltzmann energy distribution using `scipy.integrate.quad`:

```python
from cochem_kinetic.math.tunneling import calculate_eckart_tunneling

res = calculate_eckart_tunneling(
    imag_freq=520.0,  # |nu_imag| in cm^-1
    temp=298.15,      # K
    V_f=12.4,         # forward barrier in kcal/mol
    V_r=18.9,         # reverse barrier in kcal/mol
)
# res -> {"kappa": 1.31..., "provenance_tag": "[M]", "warning": None}
```

The returned dictionary always includes:
- `"kappa"` (`float`): Tunneling transmission factor ($\ge 1.0$).
- `"provenance_tag"` (`"[M]"` or `"[E]"`): `"[M]"` for a nominal Eckart evaluation, `"[E]"` if any safety gate or numerical fallback was triggered.
- `"warning"` (`str | None`): `None` on nominal `[M]` runs, or the exact diagnostic message explaining why an `[E]` fallback occurred.

#### Safety Gates in `calculate_eckart_tunneling`

1. **Negative barrier height (`V_f < 0` or `V_r < 0`):** Emits `UserWarning` + log warning, falls back to classical VTST (`kappa = 1.0`, `provenance_tag = "[E]"`).
2. **Multi-reference binding gate (`|imag_freq| > 3000 cm^-1` and `V_f < 1.0 kcal/mol`):** Emits `UserWarning` + log warning, falls back to the Wigner surrogate (`provenance_tag = "[E]"`).
3. **Quadrature failure (`scipy.integrate.quad` raises an exception or returns a non-finite value):** Emits `UserWarning` + log warning, falls back to `kappa = 1.0` (`provenance_tag = "[E]"`).

### Wigner Tunneling & Cryogenic Safeguard

`cochem_kinetic.math.tunneling.calculate_wigner_correction(imag_freq, temp)` evaluates the leading-order Wigner tunneling factor:

$$
\kappa_{\text{Wigner}}(T) = 1 + \frac{1}{24}\left(\frac{h c |\nu^\ddagger|}{k_B T}\right)^2
$$

Because the perturbative Wigner expansion diverges at cryogenic temperatures, any call with **`temp < 50.0 K`** intercepts the divergence, emits a `UserWarning` and `[E]` log entry, and returns `1.0`. For high barriers or deep tunneling regimes, prefer `calculate_eckart_tunneling` or `kinetic_core.thermo.skodje_truhlar_tunneling_correction`.

---

## 4. Smoluchowski Diffusion Limits

For solution-phase bimolecular reactions, the observable association rate cannot exceed the diffusion-controlled encounter limit predicted by the Smoluchowski and Stokes–Einstein equations:

$$
k_D = 4 \pi (D_A + D_B)(R_A + R_B) N_A \times 10^3 \quad [\text{M}^{-1}\text{s}^{-1}],
\qquad
D_i = \frac{k_B T}{6 \pi \eta R_i}
$$

```python
from cochem_kinetic.literature.diffusion_mapping import SmoluchowskiDiffusionEngine

engine = SmoluchowskiDiffusionEngine()

# Nominal water lookup (eta = 8.9e-4 Pa*s, no warning, provenance_tag == "[M]")
k_d_water = engine.compute_diffusion_limit(
    radius_a_angstrom=2.5,
    radius_b_angstrom=3.0,
    temp=298.15,
    solvent="water",
)

# Structured provenance helper (recommended for CLI / UI / pipeline callers)
prov = engine.compute_diffusion_limit_with_provenance(
    radius_a_angstrom=2.5,
    radius_b_angstrom=3.0,
    temp=298.15,
    solvent="acetonitrile",  # unknown -> emits UserWarning + logger.warning
)
# prov["viscosity_pa_s"] == 1e-3
# prov["fallback_used"]  == True
# prov["provenance_tag"] == "[E]"
# prov["warning"]        -> "[E] Unknown solvent 'acetonitrile' ..."
```

- `compute_diffusion_limit(...)` retains its pure `float` return type (`M^-1 s^-1`) for backward compatibility while populating `engine.last_fallback_warning`, `engine.last_provenance_tag`, and `engine.last_viscosity_metadata`.
- To add custom solvent viscosities cleanly without triggering a fallback, register them in `engine.known_viscosities["acetonitrile"] = 3.43e-4` (in `Pa*s`) before calling the engine.
- `process_association_rate(raw_rate, radius_a, radius_b, state_dict, temp=298.15, solvent="water")` caps `raw_rate` at `k_D`, sets `state_dict["tag"] = "DiffusionLimited"` when capped, and records `state_dict["warning"]` / `state_dict["provenance_tag"] = "[E]"` whenever a viscosity fallback occurs.

---

## 5. Microcanonical RRKM & Master Equation

### Exact Beyer–Swinehart RRKM (`RRKMSolver`)

`cochem_kinetic.engine.rrkm_theory.RRKMSolver` computes microcanonical unimolecular rate constants:

$$
k(E) = \frac{\sigma \, N^\ddagger(E - E_0)}{h \, \rho(E)}
$$

using exact Beyer–Swinehart direct counting of reactant density of states $\rho(E)$ and transition-state sum of states $N^\ddagger(E - E_0)$:

```python
from cochem_kinetic.engine.rrkm_theory import RRKMSolver

solver = RRKMSolver(
    frequencies=[400.0, 850.0, 1300.0, 3000.0],
    ts_frequencies=[380.0, 820.0, 2950.0],
    e0_cm1=5000.0,
    reaction_degeneracy=1.0,
)
k_E = solver.compute_kE(energy=9000.0, dE=2.0)  # s^-1
```

If the requested grain size `dE` exceeds the Nyquist limit `min(10.0, min_freq / 2.0)`, `RRKMSolver` emits a `UserWarning` and clamps `self.dE` to prevent state-counting aliasing.

### Bath-Gas Collision Bounds (`MasterEquation`)

`cochem_kinetic.engine.master_equation.MasterEquation(bath_gas, delta_e_down)` checks collisional downward energy transfer $\langle \Delta E_{\text{down}} \rangle$ against physical $k_B T$ limits for monoatomic bath gases (e.g., Argon), issuing a `CollisionParameterWarning(UserWarning)` and clamping unphysical values.

---

## 6. CHEM 312L Reaction-Profile Pipeline (`rxn_profile.py`)

`cochem_kinetic/rxn_profile.py` and the `cochem-kinetic` CLI implement the end-to-end charge-aware reaction-profile workflow developed for Tennessee Tech University's **CHEM 312L** course (and general organic/catalytic pathways).

### Built-In Specifications & Stoichiometry Validation

Inspect the locked CHEM 312 Spring 2027 specifications (neutral formic acid / formate esterifications in Family A; anionic cyclic/open carbonate pathways and $d_4$-EC isotopologues in Family B):

```bash
cochem-kinetic list-specs
cochem-kinetic validate-stoich --input-dir benchmarks/chem312_teams_matrix/team4_ethylene_carbonate/week10 --spec team4_ec_cyclization
```

`validate-stoich` strictly verifies identical atom counts, element formula, atom ordering (unless `--allow-reordered`), electron-count parity with `(charge, multiplicity)`, finite coordinates, and absence of nuclear clashes.

### Automatic Pathway Detection (5-Point Stepwise vs. 2-Endpoint 3-Point/5-Point)

`cochem-kinetic rxn-profile` accepts both canonical filenames (`R.xyz`, `P.xyz`, `Int_tet.xyz`, `TS1.xyz`, `TS2.xyz`, `TS.xyz`) and student 2-endpoint aliases (`reactant.xyz`, `product.xyz`):

1. **Explicit 5-point (`5pt_explicit`):** When `Int_tet.xyz` is present, runs `R -> TS1 -> Int_tet -> TS2 -> P`.
2. **2-endpoint auto-discovery (`2pt_auto`):** When only `R.xyz` + `P.xyz` (or `reactant.xyz` + `product.xyz`) are present, relaxes midpoint candidates with AIMNet2 (`discover_midpoint_intermediate_aimnet2`). If a distinct local minimum is found (`RMSD > 0.25 A`, `fmax < 0.02 eV/A`, `N_imag == 0`), it automatically promotes to the 5-point stepwise pathway; otherwise it runs an elementary 3-point pathway (`R -> TS -> P`). Override explicitly with `--pathway-mode 3pt` or `--pathway-mode 5pt`.

### Five Pipeline Stages

1. **Validation (`--stage validate`):** Stoichiometry, charge/spin parity, and input-mode detection.
2. **MLFF Stage (`--stage mlff`):**
   - Charge-aware AIMNet2 (`Q = -1, 0, +1`) endpoint relaxation.
   - 11-image ASE Climbing-Image NEB with IDPP interpolation (`--neb-images 11`).
   - Saddle-point polish via Sella or Eckart-projected P-RFO (`--ts-method auto|prfo|sella`).
   - Central-difference mass-weighted Eckart-projected Hessian verifying `N_imag == 0` on minima and `N_imag == 1` on transition states, plus RRHO and Grimme qRRHO thermochemistry.
   - Bidirectional mass-weighted IRC verification (`IRCTracerEngine`) requiring heavy-atom `RMSD < 0.15 A` and matching covalent bond graphs.
3. **DFT Single-Point Stage (`--stage dft`):**
   - PySCF CPU or GPU4PySCF (`--dft-engine cpu|gpu|both`) DF-wB97X-D3/def2-TZVPP (configurable via `--xc`, `--disp`, `--basis`, `--grid-level`) with PCM, ddCOSMO, or gas phase (`--solvation-model pcm|ddcosmo|gas`, `--solvent`, `--eps`, or `--gas-phase-ftmw-lock`).
4. **Assembly & FAIR Export (`--stage assemble`):**
   - Combines MLFF thermochemistry and DFT electronic energies into:
     - `stationary_table.csv` and `stationary_table.md`
     - `results.json` (`schema_version: "cochem.kinetic.rxn-profile/1"`)
     - `ts_polished.xyz`, `neb_path.xyz`, `irc_path.xyz`, and `geometries/<state>.xyz`
     - `manifest_sha256.json` (`schema_version: "cochem.provenance-manifest/1"`) hashing every output artifact.
5. **Cross-Environment Execution (`--stage all`):** Pass `--aimnet-python` and `--pyscf-python` when AIMNet2 and PySCF live in separate Python environments.

```bash
cochem-kinetic rxn-profile \
    --input-dir ./my_reaction \
    --output-dir ./my_reaction_out \
    --charge -1 --multiplicity 1 \
    --solvent acetonitrile --solvation-model pcm \
    --pathway-mode auto \
    --dft-engine cpu \
    --aimnet-python /path/to/aimnet/python \
    --pyscf-python /path/to/pyscf/python
```

---

## 7. Interpreting `[M]` vs `[E]` Provenance Tags & Fallback Warnings

### Provenance Tag Definitions

| Tag | Meaning | Interpretation |
| --- | --- | --- |
| **`[M]`** | **Model-authentic** | Exact target physics/lookup succeeded within its validated domain (e.g., known solvent viscosity, converged Eckart quadrature with positive barriers). No fallback warning emitted. |
| **`[D]`** | **Derived** | Quantity derived deterministically from upstream `[M]` data (e.g., thermodynamic free-energy differences, Arrhenius fits). |
| **`[E]`** | **Empirical / Fallback surrogate** | Target calculation hit a physical boundary, missing parameter, or numerical failure and fell back to a safe bounded surrogate. **Always accompanied by a `UserWarning` and `logger.warning(...)`.** |

### Complete Fallback Warning Matrix

| Module & Function | Trigger Condition | Fallback Behavior | Emitted Signals & Metadata |
| --- | --- | --- | --- |
| `cochem_kinetic.literature.diffusion_mapping.SmoluchowskiDiffusionEngine` | Solvent not in `known_viscosities` | Uses generic viscosity `1.0e-3 Pa*s` | `UserWarning`, `logger.warning`, `engine.last_fallback_warning`, `engine.last_provenance_tag == "[E]"`, `compute_diffusion_limit_with_provenance()["warning"]`, `state_dict["warning"]` |
| `cochem_kinetic.math.tunneling.calculate_eckart_tunneling` | `V_f < 0.0` or `V_r < 0.0` | Returns `kappa = 1.0` (classical VTST) | `UserWarning`, `logger.warning`, `result["provenance_tag"] == "[E]"`, `result["warning"]` |
| `cochem_kinetic.math.tunneling.calculate_eckart_tunneling` | `abs(imag_freq) > 3000.0` and `V_f < 1.0` | Calls `calculate_wigner_correction(...)` | `UserWarning`, `logger.warning`, `result["provenance_tag"] == "[E]"`, `result["warning"]` |
| `cochem_kinetic.math.tunneling.calculate_eckart_tunneling` | `scipy.integrate.quad` raises exception or returns non-finite value | Returns `kappa = 1.0` | `UserWarning`, `logger.warning`, `result["provenance_tag"] == "[E]"`, `result["warning"]` |
| `cochem_kinetic.math.tunneling.calculate_wigner_correction` | `temp < 50.0 K` (cryogenic divergence) | Returns `kappa = 1.0` | `UserWarning`, `logger.warning` (`[E] Fallback triggered: Cryogenic temperature ...`) |
| `cochem_kinetic.engine.rrkm_theory.RRKMSolver.compute_kE` | `dE > min(10.0, min_freq / 2.0)` | Clamps `self.dE` to Nyquist bound | `UserWarning` |
| `cochem_kinetic.engine.master_equation.MasterEquation` | Argon `delta_e_down > 1.2 * kT` | Downscales `delta_e_down` to `1.2 * kT` | `CollisionParameterWarning(UserWarning)` |

### Best Practices for Callers & Pipelines

- **Surface warnings in CLI/UI layers:** Always inspect `result["warning"]` from `calculate_eckart_tunneling` and `prov["warning"]` from `compute_diffusion_limit_with_provenance`, and avoid globally silencing `UserWarning`.
- **Promote `[E]` results to review before publication:** Treat any `[E]`-tagged rate constant as a diagnostic placeholder until the missing physical input (e.g., exact solvent viscosity or properly converged barrier/temperature regime) is supplied and re-run to `[M]`.
- **Verify zero-warning nominal execution in tests:** Wrap nominal runs in `warnings.catch_warnings(record=True)` to guarantee no unintended fallbacks occurred silently.
