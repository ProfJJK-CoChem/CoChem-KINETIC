---
schema: cochem-srs/5.0
repo: CoChem-KINETIC
status: ACTIVE_CORE (CHEM 312 rxn-profile with COCHEM_AIMNET_PYTHON discovery operational; VTST/MECP and CLI expansion pending)
pipeline_baseline: 5.0.0
source_commit: 1efc990b2ff1213411ad9d3efd7d7e4fb856b094
decision_ref: swarm-1126-8efd
---

# Software Requirements Specification: CoChem-KINETIC (Pipeline 5.0.0)

## 1. Purpose, Scope & System Classification

`CoChem-KINETIC` (`ProfJJK-CoChem/CoChem-KINETIC`, Python distribution `cochem-kinetic` v5.0.0) is classified as an **`ACTIVE_CORE`** (`wave1_core_9`) repository in the CoChem Pipeline 5.0.0 ecosystem. It delivers charge-aware reaction-path profiling (CHEM 312L workflows), Eyring transition-state theory (TST) thermochemistry, quantum tunneling corrections, unimolecular RRKM/Master Equation kinetics, and MLFF/AIMD trajectory utilities.

The package ships two Python trees configured in `pyproject.toml` (`include = ["cochem_kinetic*", "kinetic_core*"]`, 40 production `.py` files, 0 `NotImplementedError` stubs):
- **Primary v5.0.0 Package (`cochem_kinetic/`):** `cochem_kinetic/cli.py`, `cochem_kinetic/rxn_profile.py`, `cochem_kinetic/chem312_specs.py`, `cochem_kinetic/provider.py`, `cochem_kinetic/math/thermodynamics.py`, `cochem_kinetic/math/tunneling.py`, `cochem_kinetic/math/laplace.py`, `cochem_kinetic/engine/rrkm_theory.py`, `cochem_kinetic/engine/master_equation.py`, `cochem_kinetic/engine/nonadiabatic.py`, `cochem_kinetic/engine/hpc_dispatcher.py`, and `cochem_kinetic/literature/diffusion_mapping.py`.
- **Legacy v4 Core (`kinetic_core/`):** `kinetic_core/thermo.py`, `kinetic_core/jax_cineb.py`, `kinetic_core/mace_pre_opt.py`, `kinetic_core/aimd_hoover.py`, `kinetic_core/irc_tracer.py`, `kinetic_core/cochem_pes_store.py`, and `kinetic_core/dispatcher.py`.
- **CLI Entrypoint:** `cochem-kinetic = "cochem_kinetic.cli:main"` exposing 3 subcommands (`list-specs`, `validate-stoich`, `rxn-profile`), with `--aimnet-python` defaulting via `$COCHEM_AIMNET_PYTHON` auto-discovery in `cochem_kinetic/cli.py` (commit `1efc990` / B-04).

## 2. Pipeline 5.0.0 Architecture Alignment & Resolved Conflicts

`CoChem-KINETIC` aligns with signed Portal decision `swarm-1126-8efd` and resolves legacy v4.x drift:
- **`CONTRA-01` (Compute Topology & `HC-1`):** Local WSL2 (`host-guard`), Colab Pro+ GPU (`w51..w52`), and GCP Spot CPU execute all AIMNet2, MACE, PySCF/GPU4PySCF, and Sella saddle calculations. `.github/workflows/zero_mock_ci.yml` operates strictly with zero cron triggers and zero heavy QM compute.
- **`CONTRA-03` (Portable Storage Paths):** Hardcoded legacy Windows drive paths in `notebooks/True_Research_CoChem-KINETIC.ipynb` were eliminated in commit `49fda21d` (`NB-8`) in favor of portable path resolution.
- **`CONTRA-05` (Method Matrix & Provenance):** All kinetic rate predictions tag outputs explicitly with `[M]` (model), `[D]` (DFT/QM), or `[E]` (empirical/approximate) provenance labels and enforce `DLPNOExclusionFirewall` on any coupled-cluster refinement requests.
- **`CONTRA-06` (Contract Alignment):** `cochem_kinetic/provider.py` currently emits `cochem.module-handoff-result/1`; downstream ecosystem exchanges align to the shared `CoChem-BASE` DRAFT schemas (`cochem-job-handoff/1`, `cochem-quantities/1`, `cochem-portal-dossier/1.1`).

## 3. Authoritative Scientific Engines, Methods & Method Matrix Constraints

- **Reaction Profile Pipeline (`cochem_kinetic/rxn_profile.py`, `cochem_kinetic/chem312_specs.py`):** Runs charge-aware AIMNet2 CI-NEB path generation, Sella/Eckart P-RFO transition-state saddle refinement, IRC tracing (`kinetic_core/irc_tracer.py`), and PySCF/GPU4PySCF single-point energies (`benchmarks/chem312_t_kin_01/README.md`, `benchmarks/chem312_teams_matrix/README.md`).
- **Thermochemistry & Tunneling (`kinetic_core/thermo.py`, `cochem_kinetic/math/thermodynamics.py`, `cochem_kinetic/math/tunneling.py`, `cochem_kinetic/math/laplace.py`):** Implements Eyring TST, Grimme qRRHO entropic treatment, and Wigner, asymmetric Eckart, and Skodje-Truhlar tunneling factors.
- **Unimolecular & Diffusion Kinetics (`cochem_kinetic/engine/rrkm_theory.py`, `cochem_kinetic/engine/master_equation.py`, `cochem_kinetic/literature/diffusion_mapping.py`):** Implements Beyer-Swinehart direct state counting, RRKM microcanonical rates, Master Equation falloff, and Smoluchowski diffusion limits.
- **MLFF & AIMD Utilities (`kinetic_core/jax_cineb.py`, `kinetic_core/mace_pre_opt.py`, `kinetic_core/aimd_hoover.py`):** Provides Bofill-updated JAX CI-NEB, MACE-OFF24m pre-optimization, and Nose-Hoover AIMD (0.5 fs transition-metal timestep enforcement with `[E]` rotational-constant warnings).
- **Method Matrix Constraints:** Transition states MUST verify exactly one imaginary normal mode aligned with the reaction coordinate; fallback calculations emit explicit provenance warnings tested by `tests/test_fallback_warnings.py`.

## 4. Cross-Repo Interfaces, Contracts & Canonical Schemas

- **CoChem-BASE Module Provider:** `pyproject.toml` registers `[project.entry-points."cochem.modules"]` `kinetic = "cochem_kinetic.provider:kinetic_provider"` via `cochem_kinetic/provider.py`.
- **CoChem-SCRIBE Consumption:** Reaction-profile JSON artifacts emitted by `cochem_kinetic/rxn_profile.py` are consumed directly by `CoChem-SCRIBE`'s `render-kinetic` verified renderer.
- **HDF5 PES Persistence:** `kinetic_core/cochem_pes_store.py` provides structured HDF5 trajectory and PES array serialization.
- **Canonical Pipeline 5.0.0 Schemas:** Cross-repo workstation/cloud calculation packages target `cochem-job-handoff/1`, `cochem-quantities/1` (`OWN-117`), and `cochem-portal-dossier/1.1` (hosted as DRAFT schemas in `CoChem-BASE`). Legacy `cochem.module-handoff-result/1` payloads remain supported during migration.

## 5. Verification, Zero-Mock CI Gates & Acceptance Criteria

- **Zero-Mock CI Gate:** `tests/test_zero_mock_and_no_sim_gate.py` enforces AST scanning across all `.py` modules and `.ipynb` notebooks (`.anti_spoof_amnesty.json` strictly `[]`), executed on PRs via `.github/workflows/zero_mock_ci.yml` (`zero_mock_gate` `SUCCESS` on PR #5).
- **Functional & Regression Suites:** 8 domain test modules verify physical kinetics and CHEM 312 workflows: `tests/test_2endpoint_rxn_profile.py`, `tests/test_chem312_all_teams_matrix.py`, `tests/test_fallback_warnings.py`, `tests/test_hpc_dispatcher_and_dft_label.py`, `tests/test_kinetic_pipeline.py`, `tests/test_kinetic_stress.py`, `tests/test_modernization.py`, and `tests/test_t_kin_01_fixes.py`.
- **Branch Protection:** `main` requires authentic `blind-s3-p10` parity before PR #5 merges; stacked consolidation changes remain in Draft PR targeting `fix/zero-mock-authentic-engines`.

## 6. Verified Implementation Gaps, Known Holds & Non-Goals

The following verified gaps from the 2026-11-26 Opus-verified audit remain open:
1. **No True VTST / CVTST or MECP Landau-Zener Solver:** `cochem_kinetic/engine/nonadiabatic.py` (43 lines) implements only a `kappa_NA` warning threshold guard, not a minimum-energy crossing point (MECP) or Landau-Zener surface-crossing calculator; variational dividing-surface optimization (VTST/CVTST) is not yet implemented.
2. **Dual-Tree Fallback Import Bugs:** `kinetic_core/dispatcher.py` (line 28) and `kinetic_core/jax_cineb.py` (line 26) contain broken `except ImportError: from cochem_pes_store import PESStore` fallbacks referencing a nonexistent top-level module instead of `kinetic_core.cochem_pes_store`.
3. **Unwired CLI Subcommands:** `cochem_kinetic/cli.py` exposes only the 3 CHEM 312 subcommands (`list-specs`, `validate-stoich`, `rxn-profile`); Eyring TST, Eckart tunneling, RRKM, Master Equation, Nose-Hoover AIMD, and JAX CI-NEB engines are callable via Python API only.
4. **Empty Placeholder Constructors:** `cochem_kinetic/interfaces/web_contours.py`, `cochem_kinetic/interfaces/web_mcp.py`, and `cochem_kinetic/math/thermodynamics.py` retain minor empty `__init__` pass bodies.
5. **Non-Goals:** `CoChem-KINETIC` does not perform excited-state FSSH non-adiabatic dynamics (`CoChem-PULSE`), CBS composite thermochemistry (`CoChem-BENCH`), or conformer enumeration (`CoChem-TOPOS`).

## 7. Legacy Specification Disposition (Non-Normative Archive)

Prior to `swarm-1126-8efd`, `CoChem-KINETIC` had no formal SRS despite bumping `pyproject.toml` to `5.0.0`. The following historical repository notes remain non-normative:
- `20260807_kinetic_architectural_changes.md` (references renamed/deleted pre-v5 paths `kinetic_core/neb_optimizer.py`, `kinetic_core/pre_optimizer.py`, and `kinetic_core/aimd.py`, which were replaced by `kinetic_core/jax_cineb.py`, `kinetic_core/mace_pre_opt.py`, and `kinetic_core/aimd_hoover.py`).
- `20260807_kinetic_workflow.md` and `workflow.md` (pre-v5 workflow notes).
- `docs/USER_GUIDE.md` remains the complementary operational user guide subordinate to this normative `docs/SRS.md`.
