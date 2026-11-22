# team4_ec_cyclization — 5-Stationary-Point Reaction Profile

- **Reaction:** `-O-CH2CH2-OC(=O)OMe -> TS1 -> Int_tet -> TS2 -> EC + MeO-`
- **Formula / Charge / Spin:** `C4H7O4-` (15 atoms, Q = -1, M = 1)
- **DFT Level:** `DF-WB97X-D3ZERO/def2-tzvpp` + `PCM(acetonitrile, eps=35.688)`
- **Thermochemistry Temperature:** `298.15 K` (AIMNet2 Hessian + RRHO / Grimme qRRHO)

| State | Status | N_imag | Imag Freq (cm^-1) | Key C-O Bond | E_DFT (Ha) | dE (kcal/mol) | dH (kcal/mol) | dG_RRHO (kcal/mol) | dG_qRRHO (kcal/mol) | DFT Time (s) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | CONVERGED | 0 | 0.00 | - | -457.7123149741 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.822 |
| TS1 | CONVERGED | 1 | -63.13 | - | -457.7113666476 | 0.5951 | -0.0288 | 1.2450 | 0.9262 | 10.666 |
| Int_tet | CONVERGED | 0 | 0.00 | - | -457.7337023584 | -13.4208 | -12.8806 | -11.7879 | -11.7458 | 9.949 |
| TS2 | CONVERGED | 1 | -183.59 | C0-O4=2.599A | -457.7114418460 | 0.5479 | -0.4329 | 0.9630 | 0.7090 | 9.882 |
| P | CONVERGED | 0 | 0.00 | - | -457.7131785666 | -0.5419 | -1.1472 | -1.0603 | -1.1760 | 10.115 |

## Bidirectional Mass-Weighted IRC Connectivity Verification

| TS | Orientation | Max COM Drift (A) | RMSD End A (A) | RMSD End B (A) | Graph A | Graph B | Strict Connected |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| TS1 | forward->A, reverse->B | 9.089e-16 | 0.0001 | 0.0000 | True | True | True |
| TS2 | reverse->A, forward->B | 5.839e-16 | 0.0001 | 0.1339 | True | True | True |

## PySCF CPU vs GPU4PySCF Single-Point Benchmark

| State | N_AO | GPU Time (s) | CPU Time (s) | Speedup | |E_GPU - E_CPU| (Ha) | 5-Point Est. CPU Time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 346 | 12.822 | 90.415 | 7.05x | 3.41e-11 | 7.53 |
