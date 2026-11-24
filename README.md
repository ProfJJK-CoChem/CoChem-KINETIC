# CoChem-KINETIC: Microkinetic Modeling, TST & Reaction-Profile Engine

**Version:** 5.0.0  
**Author/PI:** Dr. Joshua John Klaassen  
**ORCiD:** [https://orcid.org/0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)  
**GitHub Organization:** [https://github.com/ProfJJK-CoChem](https://github.com/ProfJJK-CoChem)  
**User Guide:** [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## 1. Module Overview

`CoChem-KINETIC` is the microkinetic modeling, transition-state theory (TST), tunneling, RRKM, and reaction-coordinate profile engine for the **CoChem Pipeline 5.0.0** ecosystem. It bridges electronic-structure stationary points and spectroscopic properties into macroscopic, temperature- and pressure-dependent rate constants and auditable reaction profiles.

### Ecosystem Integration

- **Upstream conformer & PES discovery:** Ingests geometries and potential energy surfaces from `CoChem-TOPOS` and FAIR-compliant `PESStore` HDF5 archives (`kinetic_core.cochem_pes_store`).
- **Spectroscopic handoff (`CoChem-TORQ` & `CoChem-SPYFY`):** Pairs harmonic/anharmonic vibrational thermochemistry (`CoChem-TORQ`) and gas-phase rotational/FTMW spectroscopic fitting (`CoChem-SPYFY`) with reaction-profile analysis — including `--gas-phase-ftmw-lock` to enforce vacuum (`eps = 1.0`) consistency with FTMW rotational constants.
- **Workflow orchestration:** Registers as the `cochem.modules` entry point `kinetic = cochem_kinetic.provider:kinetic_provider` for execution within `CoChem-NODE`.

---

## 2. Capabilities & Theoretical Framework

- **Transition State Theory & Thermochemistry:** Eyring TST (`kinetic_core.thermo.calculate_eyring_rate`), detailed-balance symmetry enforcement (`cochem_kinetic.math.thermodynamics.ThermodynamicSymmetryEnforcer`), and Grimme's quasi-RRHO (`grimes_qrrho_entropy`) interpolation for low-frequency vibrational modes.
- **Quantum Tunneling Corrections:** Asymmetric Eckart tunneling (`cochem_kinetic.math.tunneling.calculate_eckart_tunneling`), Wigner tunneling with cryogenic safeguard (`calculate_wigner_correction`), and Skodje–Truhlar high-barrier tunneling (`kinetic_core.thermo.skodje_truhlar_tunneling_correction`).
- **Microcanonical RRKM & Master Equation:** Exact Beyer–Swinehart direct-count density and sum of states (`cochem_kinetic.engine.rrkm_theory.RRKMSolver`) and collision energy-transfer bounding (`cochem_kinetic.engine.master_equation.MasterEquation`).
- **Solution-Phase Diffusion Limits:** Stokes–Einstein / Smoluchowski bimolecular encounter limits (`cochem_kinetic.literature.diffusion_mapping.SmoluchowskiDiffusionEngine`).
- **Charge-Aware Reaction-Profile Pipeline (`cochem_kinetic/rxn_profile.py`):** End-to-end AIMNet2 CI-NEB, Sella / Eckart-projected P-RFO saddle polish, central-difference Eckart-projected Hessian, bidirectional mass-weighted IRC verification, and PySCF / GPU4PySCF DF-wB97X-D3/def2-TZVPP + PCM/ddCOSMO single-point assembly for CHEM 312L workflows (both 5-point stepwise and 2-endpoint auto 3-point/5-point modes).

---

## 3. Warning & Provenance Guarantees (`[M]` / `[D]` / `[E]`)

`CoChem-KINETIC` enforces **zero silent fallbacks**. Every calculation carries an explicit provenance tag (`[M]` model-authentic, `[D]` derived, or `[E]` empirical/fallback surrogate):

- **Unknown solvent viscosity (`SmoluchowskiDiffusionEngine`):** Falls back to the generic viscosity `1.0e-3 Pa*s` while emitting **both** `warnings.warn(..., UserWarning)` and `logger.warning(...)`, recording structured provenance metadata on `engine.last_fallback_warning`, `engine.last_provenance_tag == "[E]"`, `engine.last_viscosity_metadata`, and via `compute_diffusion_limit_with_provenance(...)`.
- **Tunneling fallbacks (`calculate_eckart_tunneling` & `calculate_wigner_correction`):** Negative barrier heights, extreme high-frequency/low-barrier multi-reference failures, `scipy.integrate.quad` integration failures, and cryogenic `T < 50 K` Wigner divergence all emit `warnings.warn(..., UserWarning)` + `logger.warning(...)`, tag provenance as `"[E]"`, and expose an explicit `"warning"` message field in the returned Eckart dictionary (while normal `[M]` regimes emit no fallback warning and set `"warning": None`).
- **Honest TS & IRC diagnostics:** Unconverged saddles or missing DFT stages are reported honestly (`FAILED_SADDLE_CONVERGENCE` / `PENDING` / `NaN`) and hashed into `manifest_sha256.json`.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for the complete fallback-warning matrix and handling patterns.

---

## 4. Installation

```bash
pip install -e .
```

Optional extras in `pyproject.toml`:

```bash
pip install -e ".[aimnet,dft,ts]"   # AIMNet2 + ASE, PySCF + D3, Sella TS optimizer
pip install -e ".[jax,mace,viz]"    # JAX CI-NEB, MACE pre-opt, py3Dmol visualization
```

---

## 5. Command-Line Interface (`cochem-kinetic`)

`cochem-kinetic` provides three subcommands:

```bash
# 1. List built-in CHEM 312 Spring 2027 balanced reaction specifications
cochem-kinetic list-specs

# 2. Strictly validate atom count, element order, charge, spin, and clash balance
cochem-kinetic validate-stoich --input-dir path/to/geometries --spec team4_ec_cyclization

# 3. Run an end-to-end charge-aware reaction profile (auto-detects 5pt vs 2-endpoint 3pt/5pt)
cochem-kinetic rxn-profile \
    --input-dir path/to/geometries \
    --output-dir path/to/output \
    --spec team4_ec_cyclization \
    --solvent acetonitrile \
    --solvation-model pcm \
    --dft-engine cpu
```

For vacuum FTMW-consistent benchmarking alongside `CoChem-SPYFY`, add `--gas-phase-ftmw-lock` (forces gas phase with `eps = 1.0`). To split AIMNet2 and PySCF across separate Python environments, use `--stage validate|mlff|dft|assemble` or supply `--aimnet-python` / `--pyscf-python`.

---

## 6. Documentation & Testing

- **Full Guide:** Read [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for detailed equations, Python API examples, CHEM 312L reaction-profile workflows, and provenance interpretation.
- **Test Suite:** Run the zero-mock pytest suite:
  ```bash
  python -B -m pytest -q
  ```

---

## 7. Citation & License

If `CoChem-KINETIC` is used in published work, cite the CoChem ecosystem:
- *Klaassen, J. J. et al. "CoChem: A Differentiable Tensor Framework for Heterogeneous Ab Initio Workflows." (2026)*

Licensed under the CoChem Academic License. See [LICENSE](LICENSE).
