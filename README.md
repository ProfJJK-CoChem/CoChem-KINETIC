# CoChem-KINETIC

**CoChem-KINETIC** is the Transition State and Thermal Dynamics engine of the extended CoChem suite.

It is responsible for:
- Executing JAX-accelerated Climbing Image Nudged Elastic Band (CI-NEB) algorithms to isolate precise Minimum Energy Pathways.
- Leveraging `MACE-OFF24m` Active Learning for rapid pre-optimization of kinetic bands before handing taut strings to ORCA.
- Enforcing dynamic Wigner tunneling corrections if transition state imaginary frequencies cross defined energetic thresholds (e.g., $1000$ cm$^{-1}$).
- Conducting strict Nose-Hoover Ab Initio Molecular Dynamics (AIMD) at elevated temperatures to sample the entropic conformational space, extracting critical thermal parameters.

## Usage
Please refer to the authoritative `CoChem_Master_User_Manual.md` located in the `CoChem-BASE` repository for full execution instructions across the entire pipeline.