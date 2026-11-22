# CHEM 312 Spring 2027 — All-6-Team Input Matrix & Charge-Aware AIMNet2 Qualification

**Institution:** Tennessee Tech University (CHEM 312 Spring 2027 CURE)  
**Engine:** `cochem-kinetic` (`CoChem-KINETIC v5.0.0`)  
**MLFF Qualification Backend:** Charge-aware `AIMNet2` (`aimnet==0.2.5`, `ase==3.29.0`, `torch==2.13.0+cu129`, `NVIDIA GeForce RTX 3090`)  
**Elements Qualified:** `C, H, D, O, F, S` | **Charges Qualified:** `Q = 0, -1` | **Convergence Gate:** L-BFGS `fmax < 0.02 eV/Å`

---

## All-6-Team Qualification Matrix

| Team | Family | Monomer (`Q=0`) | Week 10 Reaction Complex | `Q` | Elements | AIMNet2 `fmax` (eV/Å) | Full TS/IRC/DFT+PCM Status |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| **Team 1** | Family A (neutral formate) | Propargyl formate (`C4H4O2`, 10) | `C5H10O6S` (22, 5-state `R..P`) | `0` | `C,H,O,S` | `0.0095 / 0.0010 / 0.0009` | **VERIFIED** (`benchmarks/chem312_t_kin_01/case_a_team1_formate/`) |
| **Team 2** | Family A (fluorinated formate) | 2,2-Difluoroethyl formate (`C3H4F2O2`, 11) | `C4H10F2O6S` (23, `R` & `P`) | `0` | `C,H,F,O,S` | `0.0064 / 0.0094 / 0.0084` | `PENDING_STUDENT_RUN` |
| **Team 3** | Family A (trifluorinated formate) | 2,2,2-Trifluoroethyl formate (`C3H3F3O2`, 11) | `C4H9F3O6S` (23, `R` & `P`) | `0` | `C,H,F,O,S` | `0.0099 / 0.0100 / 0.0091` | `PENDING_STUDENT_RUN` |
| **Team 4** | Family B (anionic cyclic carbonate) | Ethylene carbonate (`C3H4O3`, 10) | `C4H7O4-` (15, 5-state `R..P`) + 21-atom `C5H11O5-` | `-1` | `C,H,O` | `0.0078 / 0.0005 / 0.0007` | **VERIFIED** (`benchmarks/chem312_t_kin_01/case_b_team4_ec_cyclization/`) |
| **Team 5** | Family B (anionic chiral carbonate) | (±)-Propylene carbonate (`C4H6O3`, 13) | `C5H9O4-` (18, `R` & `P`) + 24-atom `C6H13O5-` | `-1` | `C,H,O` | `0.0089 / 0.0099 / 0.0080` | `PENDING_STUDENT_RUN` |
| **Team 6** | Family B (open carbonate + `d4-EC`) | Ethyl methyl carbonate (`C4H8O3`, 15) + `d4-EC` (`C3D4O3`, 10) | `C5H11O4-` (20, `R` & `P`) | `-1` | `C,H,D,O` | `0.0079 / 0.0066 / 0.0086` | `PENDING_STUDENT_RUN` |

All real AIMNet2 energies, gradients (`fmax`, `frms`), wall-clock timings, and SHA-256 hashes for all 29 staged XYZ files are recorded in [`teams_qualification_manifest.json`](teams_qualification_manifest.json).

---

## CLI Verification Commands

```bash
cochem-kinetic list-specs
cochem-kinetic validate-stoich --input-dir benchmarks/chem312_teams_matrix/team2_difluoroethyl_formate/week10 --spec team2_difluoroethyl_formate
cochem-kinetic validate-stoich --input-dir benchmarks/chem312_teams_matrix/team6_emc_d4ec/d4_ec_isotopologue --spec team6_d4_ec_isotopologue
```
