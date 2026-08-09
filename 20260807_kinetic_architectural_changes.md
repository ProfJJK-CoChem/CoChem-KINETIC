# CoChem-KINETIC: Architectural Changes (2026-08-07)

## 1. JAX-Accelerated CI-NEB
**Target File:** `kinetic_core/neb_optimizer.py`
**Required Architectural Change:**
- Implement JAX-accelerated Climbing Image Nudged Elastic Band (CI-NEB). Ensure the highest energy image correctly targets the saddle point. Use Bofill Hessian updates for TS convergence.

## 2. Active Learning Sweeps
**Target File:** `kinetic_core/pre_optimizer.py`
**Required Architectural Change:**
- Utilize `MACE-OFF24m` to execute rapid initial sweeps of the NEB band. Only pass the mathematically taut spring coordinates to DFT (r2SCAN-3c) for the final rigorous geometric relaxation.

## 3. Dynamic Wigner Tunneling
**Target File:** `kinetic_core/thermo.py`
**Required Architectural Change:**
- KINETIC must isolate the single imaginary frequency of the TS. If $\nu_i > 1000 \text{ cm}^{-1}$ (e.g., hydrogen transfer), it automatically applies the Wigner tunneling coefficient correction to the final rate constant.

## 4. AIMD Enforcements
**Target File:** `kinetic_core/aimd.py`
**Required Architectural Change:**
- For Ab Initio Molecular Dynamics (AIMD), KINETIC must strictly enforce the Nose-Hoover thermostat for canonical NVT sampling. It must parse the system for Transition Metals and lower the time step to 0.5 fs automatically to prevent energy drift.
