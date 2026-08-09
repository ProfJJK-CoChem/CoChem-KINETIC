# CoChem-KINETIC: Execution Workflow (2026-08-07)

## Phase 1: Reaction Coordinate Generation
1. **Band Initialization:** The user selects Reactants and Products in the UI. KINETIC interpolates 11 initial images.
2. **MACE Pre-Relaxation:** MACE-OFF24m executes an instant 10,000-step NEB sweep, yielding a smooth structural band.

## Phase 2: Transition State Search
1. **CI-NEB Execution:** ORCA refines the highest-energy image using CI-NEB.
2. **Frequency Verification:** A numerical frequency calculation confirms exactly one imaginary mode.
3. **IRC Tracing:** KINETIC calculates the Intrinsic Reaction Coordinate (IRC) strictly down both sides of the saddle point to guarantee the Reactant/Product connection.

## Phase 3: AIMD & Thermodynamics
1. **Thermal Sampling:** If requested, Nose-Hoover AIMD at 298.15 K explores the entropic basin. JAX active learning aborts the trajectory early if the conformational space is exhausted.
2. **Free Energy Calculation:** KINETIC extracts Zero-Point Energy (ZPE) and explicit solvent umbrella sampling statistics. It applies Wigner tunneling if $\nu_i > 1000$.
3. **UI Rendering:** The user receives a continuous 3D HTML animation of the IRC pathway and a 2D plot of the energy barrier in the Jupyter dashboard.
