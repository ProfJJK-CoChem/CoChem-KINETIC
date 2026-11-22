# team1_propargyl_formate — 5-Stationary-Point Reaction Profile

- **Reaction:** `HC#C-CH2OH + HCOOH + CH3SO3H -> HCOOCH2C#CH + H2O + CH3SO3H`
- **Formula / Charge / Spin:** `C5H10O6S` (22 atoms, Q = 0, M = 1)
- **DFT Level:** `DF-WB97X-D3ZERO/def2-tzvpp` + `PCM(acetonitrile, eps=35.688)`
- **Thermochemistry Temperature:** `298.15 K` (AIMNet2 Hessian + RRHO / Grimme qRRHO)

| State | Status | N_imag | Imag Freq (cm^-1) | Key C-O Bond | E_DFT (Ha) | dE (kcal/mol) | dH (kcal/mol) | dG_RRHO (kcal/mol) | dG_qRRHO (kcal/mol) | DFT Time (s) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | CONVERGED | 0 | 0.00 | - | -1046.1912066606 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 31.803 |
| TS1 | CONVERGED | 1 | -309.55 | O0-C8=1.780A | -1046.1646956875 | 16.6359 | 15.7326 | 18.4138 | 18.1319 | 30.081 |
| Int_tet | CONVERGED | 0 | 0.00 | - | -1046.1816099784 | 6.0220 | 6.8071 | 7.7474 | 7.9460 | 32.048 |
| TS2 | CONVERGED | 1 | -332.90 | C8-O10=1.800A | -1046.1623866568 | 18.0848 | 15.9041 | 18.7603 | 18.3319 | 31.843 |
| P | CONVERGED | 0 | 0.00 | - | -1046.1844287591 | 4.2532 | 3.7746 | 0.8700 | 1.7504 | 33.978 |

## Bidirectional Mass-Weighted IRC Connectivity Verification

| TS | Orientation | Max COM Drift (A) | RMSD End A (A) | RMSD End B (A) | Graph A | Graph B | Strict Connected |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| TS1 | forward->A, reverse->B | 1.844e-15 | 0.0013 | 0.0027 | True | True | True |
| TS2 | forward->A, reverse->B | 1.897e-15 | 0.5840 | 0.0084 | True | True | False |

## PySCF CPU vs GPU4PySCF Single-Point Benchmark

| State | N_AO | GPU Time (s) | CPU Time (s) | Speedup | |E_GPU - E_CPU| (Ha) | 5-Point Est. CPU Time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 523 | 31.803 | 307.262 | 9.66x | 1.02e-11 | 25.61 |
