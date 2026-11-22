# T-KIN-01 — CHEM 312 Spring 2027 Real Scientific Qualification Benchmarks

**Institution:** Tennessee Tech University (CHEM 312 Spring 2027 CURE)  
**Engine:** `cochem-kinetic rxn-profile` (`CoChem-KINETIC v5.0.0`)  
**MLFF Geometry / CI-NEB / TS / Hessian / IRC:** Charge-aware `AIMNet2` (`aimnet==0.2.5`, `ase==3.29.0`, RTX 3090 GPU)  
**DFT Single-Point & Continuum Solvation:** PySCF `2.14.0` / GPU4PySCF `1.9.0` `DF-wB97X-D3ZERO/def2-TZVPP` + `C-PCM(acetonitrile, eps=35.688)`  

---

## Qualified Cases

### 1. Case A — Team 1 Neutral Formate Esterification (`case_a_team1_formate/`)
- **Balanced Reaction:** `HC#C-CH2OH + HCOOH + CH3SO3H -> HCOOCH2C#CH + H2O + CH3SO3H`
- **Stoichiometry:** `C5H10O6S` (22 atoms, `Q = 0`, `M = 1`, `N_AO = 523`)
- **Stationary Points (`5/5 CONVERGED`, `N_imag = 0, 1, 0, 1, 0`):**
  - `R`: `E = -1046.1912066606 Ha` (`dG_qRRHO = 0.00 kcal/mol`)
  - `TS1`: `E = -1046.1646956875 Ha` (`dG_qRRHO = +18.13 kcal/mol`, `nu_imag = -309.55 cm^-1`, `O0-C8 = 1.780 A`)
  - `Int_tet`: `E = -1046.1816099784 Ha` (`dG_qRRHO = +7.95 kcal/mol`)
  - `TS2`: `E = -1046.1623866568 Ha` (`dG_qRRHO = +18.33 kcal/mol`, `nu_imag = -332.90 cm^-1`, `C8-O10 = 1.800 A`)
  - `P`: `E = -1046.1844287591 Ha` (`dG_qRRHO = +1.75 kcal/mol`)
- **Bidirectional Mass-Weighted IRC:**
  - `TS1`: `COM drift = 1.84e-15 A`, `RMSD(R) = 0.0013 A`, `RMSD(Int_tet) = 0.0027 A`, `graph_match = (True, True)`, `strict = True`
  - `TS2`: `COM drift = 1.90e-15 A`, `RMSD(P) = 0.0084 A`, `RMSD(Int_tet) = 0.5840 A`, `graph_match = (True, True)` (honest physical observation: `TS2` IRC lands on the `CH3SO3H` H-bond rotamer of `Int_tet` shifted by `0.58 A` from `O0` toward departing `O10`, `+0.28 kcal/mol` above `Int_tet`, with identical covalent bond graph).

### 2. Case B — Team 4 Anionic 5-exo-trig Cyclization (`case_b_team4_ec_cyclization/`)
- **Balanced Reaction:** `-O-CH2CH2-OC(=O)OMe -> TS1 -> Int_tet -> TS2 -> EC + MeO-`
- **Stoichiometry:** `C4H7O4-` (15 atoms, `Q = -1`, `M = 1`, `N_AO = 346`)
- **Stationary Points (`5/5 CONVERGED`, `N_imag = 0, 1, 0, 1, 0`):**
  - `R`: `E = -457.7123149741 Ha` (`dG_qRRHO = 0.00 kcal/mol`)
  - `TS1`: `E = -457.7113666476 Ha` (`dG_qRRHO = +0.93 kcal/mol`, `nu_imag = -63.13 cm^-1`, pre-reactive 5-exo-trig attack gate)
  - `Int_tet`: `E = -457.7337023584 Ha` (`dG_qRRHO = -11.75 kcal/mol`, cyclic hemiorthocarbonate resting state)
  - `TS2`: `E = -457.7114418460 Ha` (`dG_qRRHO = +0.71 kcal/mol` vs `R`, `+12.45 kcal/mol` vs `Int_tet`, `nu_imag = -183.59 cm^-1`, `C0-O4 = 2.599 A`)
  - `P`: `E = -457.7131785666 Ha` (`dG_qRRHO = -1.18 kcal/mol`, `EC + MeO-`)
- **Bidirectional Mass-Weighted IRC:**
  - `TS1`: `COM drift = 9.09e-16 A`, `RMSD(R) = 0.0001 A`, `RMSD(Int_tet) = 0.0000 A`, `graph_match = (True, True)`, `strict = True`
  - `TS2`: `COM drift = 5.84e-16 A`, `RMSD(Int_tet) = 0.0001 A`, `RMSD(P) = 0.1339 A`, `graph_match = (True, True)`, `strict = True`

### 3. Case B Intermolecular Step Balanced Input Spec (`case_b_intermolecular_21atom_spec/`)
- **Balanced Reaction:** `HO-CH2CH2-O- + (CH3O)2C=O -> TS1 -> Int_tet -> TS2 -> HO-CH2CH2-OC(=O)OMe + MeO-`
- **Stoichiometry:** `C5H11O5-` (21 atoms, `Q = -1`, `M = 1`, `82` electrons), atom-index-consistent `inputs/{R,Int_tet,P}.xyz` + `spec_validation.json`.

---

## Empirical CPU vs GPU4PySCF Wall-Clock Analysis (Student Public-Fork Actions `<90 min` Gate)

| Case | Atoms | Q | N_AO (`def2-TZVPP`) | GPU4PySCF (s/pt) | CPU PySCF Workstation (s/pt) | GPU Speedup | `|E_GPU - E_CPU|` (Ha) | 5-Pt Workstation CPU (min) | Est. 4-vCPU Public Actions (min) | Routing Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Case B (Team 4) | 15 | -1 | 346 | 12.82 (10.7 warm) | 90.42 | 7.05x | 3.41e-11 | 7.53 | ~25–35 | Fits `<90 min` on student public-fork CPU Actions |
| Case A (Team 1) | 22 | 0 | 523 | 31.80 (31.9 avg) | 307.26 | 9.66x | 1.02e-11 | 25.61 | ~90–130 | Exceeds comfortable `<90 min` on 4-vCPU CPU Actions; offload `def2-TZVPP + PCM` refinement via Pipeline 5.0.0 GPU (`2.66 min` total on RTX 3090) |
