"""
CHEM 312 Spring 2027 Balanced Elementary Reaction Specifications & Stoichiometry Validator
------------------------------------------------------------------------------------------
Institution: Tennessee Tech University (CHEM 312 Spring 2027 CURE)

Provides locked, atom- and charge-balanced monomer and Week 10 reaction specifications for
all 6 CHEM 312 Spring 2027 teams:
  - Team 1 (Family A, neutral):
      Propargyl formate monomer (C4H4O2, 10 atoms, Q=0) &
      HC#C-CH2OH + HCOOH + CH3SO3H -> HCOOCH2C#CH + H2O + CH3SO3H (C5H10O6S, 22 atoms, Q=0)
  - Team 2 (Family A, fluorinated neutral):
      2,2-Difluoroethyl formate monomer (C3H4F2O2, 11 atoms, Q=0) &
      CHF2CH2OH + HCOOH + CH3SO3H -> HCOOCH2CHF2 + H2O + CH3SO3H (C4H10F2O6S, 23 atoms, Q=0)
  - Team 3 (Family A, trifluorinated neutral):
      2,2,2-Trifluoroethyl formate monomer (C3H3F3O2, 11 atoms, Q=0) &
      CF3CH2OH + HCOOH + CH3SO3H -> HCOOCH2CF3 + H2O + CH3SO3H (C4H9F3O6S, 23 atoms, Q=0)
  - Team 4 (Family B, anionic cyclic carbonate):
      Ethylene carbonate monomer (C3H4O3, 10 atoms, Q=0),
      -O-CH2CH2-OC(=O)OMe -> EC + MeO- (C4H7O4-, 15 atoms, Q=-1), &
      HO-CH2CH2-O- + DMC -> HO-CH2CH2-OC(=O)OMe + MeO- (C5H11O5-, 21 atoms, Q=-1)
  - Team 5 (Family B, anionic chiral cyclic carbonate):
      (+-)-Propylene carbonate monomer (C4H6O3, 13 atoms, Q=0),
      -O-CH2CH(CH3)-OC(=O)OMe -> PC + MeO- (C5H9O4-, 18 atoms, Q=-1), &
      HO-CH2CH(CH3)-O- + DMC -> HO-CH2CH(CH3)-OC(=O)OMe + MeO- (C6H13O5-, 24 atoms, Q=-1)
  - Team 6 (Family B, anionic open carbonate + deuterated EC):
      Ethyl methyl carbonate monomer (C4H8O3, 15 atoms, Q=0),
      EtO- + DMC -> EMC + MeO- (C5H11O4-, 20 atoms, Q=-1), &
      Weeks 6-8 d4-EC isotopologue (C3D4O3, 10 atoms, Q=0)

All stationary points (R, TS1, Int_tet, TS2, P) and monomers are strictly validated for
identical atom count, element formula, charge, multiplicity, finite coordinates, and
absence of nuclear clashes.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

STATIONARY_STATES: tuple[str, ...] = ("R", "TS1", "Int_tet", "TS2", "P")

# Static dielectric constants (298.15 K) for continuum PCM / ddCOSMO solvation
SOLVENT_DIELECTRICS: dict[str, float] = {
    "gas": 1.0,
    "vacuum": 1.0,
    "water": 78.3553,
    "acetonitrile": 35.688,
    "mecn": 35.688,
    "methanol": 32.613,
    "meoh": 32.613,
    "ethanol": 24.852,
    "formic_acid": 51.1,
    "dmc": 3.09,
    "ec": 89.78,
    "thf": 7.4257,
    "dmso": 46.826,
}

# Standard atomic masses (IUPAC atomic weights, amu) for mass-weighting and RRHO
ATOMIC_MASSES_AMU: dict[str, float] = {
    "H": 1.00782503223,
    "D": 2.01410177812,
    "B": 11.00930536,
    "C": 12.0000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Br": 78.9183376,
    "I": 126.9044719,
}

# Atomic numbers for electron count parity validation
ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "D": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Br": 35,
    "I": 53,
}


BUILTIN_SPECS: dict[str, dict[str, Any]] = {
    "team1_propargyl_formate_monomer": {
        "name": "team1_propargyl_formate_monomer",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family A (Neutral Formate Monomer)",
        "team": "Team 1",
        "reaction_equation": "Propargyl formate monomer (HCOOCH2C#CH)",
        "formula": "C4H4O2",
        "element_counts": {"C": 4, "H": 4, "O": 2},
        "n_atoms": 10,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "gas",
        "default_eps": 1.0,
        "default_temp_k": 298.15,
        "stationary_states": ["monomer"],
        "required_states": ["monomer"],
    },
    "team1_propargyl_formate": {
        "name": "team1_propargyl_formate",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family A (Formates, Amberlyst-15 modeled as CH3SO3H proton shuttle)",
        "team": "Team 1",
        "reaction_equation": "HC#C-CH2OH + HCOOH + CH3SO3H -> HCOOCH2C#CH + H2O + CH3SO3H",
        "formula": "C5H10O6S",
        "element_counts": {"C": 5, "H": 10, "O": 6, "S": 1},
        "n_atoms": 22,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": list(STATIONARY_STATES),
        "ts1_description": "Nucleophilic attack of propargyl alcohol O onto formic acid carbonyl C assisted by CH3SO3H proton shuttle",
        "int_description": "Neutral tetrahedral orthoformate intermediate (HC#C-CH2-O-CH(OH)2 ... CH3SO3H)",
        "ts2_description": "Elimination of H2O from tetrahedral intermediate assisted by CH3SO3H proton shuttle",
        "reactant_fragments": ["HC#C-CH2OH (C3H4O, 8)", "HCOOH (CH2O2, 5)", "CH3SO3H (CH4O3S, 9)"],
        "product_fragments": ["HCOOCH2C#CH (C4H4O2, 10)", "H2O (H2O, 3)", "CH3SO3H (CH4O3S, 9)"],
    },
    "team2_difluoroethyl_formate_monomer": {
        "name": "team2_difluoroethyl_formate_monomer",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family A (Fluorinated Neutral Formate Monomer)",
        "team": "Team 2",
        "reaction_equation": "2,2-Difluoroethyl formate monomer (HCOOCH2CHF2)",
        "formula": "C3H4F2O2",
        "element_counts": {"C": 3, "H": 4, "F": 2, "O": 2},
        "n_atoms": 11,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "gas",
        "default_eps": 1.0,
        "default_temp_k": 298.15,
        "stationary_states": ["monomer"],
        "required_states": ["monomer"],
    },
    "team2_difluoroethyl_formate": {
        "name": "team2_difluoroethyl_formate",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family A (Fluorinated Formates, Amberlyst-15 modeled as CH3SO3H proton shuttle)",
        "team": "Team 2",
        "reaction_equation": "CHF2CH2OH + HCOOH + CH3SO3H -> HCOOCH2CHF2 + H2O + CH3SO3H",
        "formula": "C4H10F2O6S",
        "element_counts": {"C": 4, "H": 10, "F": 2, "O": 6, "S": 1},
        "n_atoms": 23,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": ["R", "P"],
        "ts1_description": "Nucleophilic attack of 2,2-difluoroethanol O onto formic acid carbonyl C assisted by CH3SO3H proton shuttle",
        "int_description": "Neutral tetrahedral orthoformate intermediate (CHF2CH2-O-CH(OH)2 ... CH3SO3H)",
        "ts2_description": "Elimination of H2O from tetrahedral intermediate assisted by CH3SO3H proton shuttle",
        "reactant_fragments": ["CHF2CH2OH (C2H4F2O, 9)", "HCOOH (CH2O2, 5)", "CH3SO3H (CH4O3S, 9)"],
        "product_fragments": ["HCOOCH2CHF2 (C3H4F2O2, 11)", "H2O (H2O, 3)", "CH3SO3H (CH4O3S, 9)"],
    },
    "team3_trifluoroethyl_formate_monomer": {
        "name": "team3_trifluoroethyl_formate_monomer",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family A (Trifluorinated Neutral Formate Monomer)",
        "team": "Team 3",
        "reaction_equation": "2,2,2-Trifluoroethyl formate monomer (HCOOCH2CF3)",
        "formula": "C3H3F3O2",
        "element_counts": {"C": 3, "H": 3, "F": 3, "O": 2},
        "n_atoms": 11,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "gas",
        "default_eps": 1.0,
        "default_temp_k": 298.15,
        "stationary_states": ["monomer"],
        "required_states": ["monomer"],
    },
    "team3_trifluoroethyl_formate": {
        "name": "team3_trifluoroethyl_formate",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family A (Trifluorinated Formates, Amberlyst-15 modeled as CH3SO3H proton shuttle)",
        "team": "Team 3",
        "reaction_equation": "CF3CH2OH + HCOOH + CH3SO3H -> HCOOCH2CF3 + H2O + CH3SO3H",
        "formula": "C4H9F3O6S",
        "element_counts": {"C": 4, "H": 9, "F": 3, "O": 6, "S": 1},
        "n_atoms": 23,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": ["R", "P"],
        "ts1_description": "Nucleophilic attack of 2,2,2-trifluoroethanol O onto formic acid carbonyl C assisted by CH3SO3H proton shuttle",
        "int_description": "Neutral tetrahedral orthoformate intermediate (CF3CH2-O-CH(OH)2 ... CH3SO3H)",
        "ts2_description": "Elimination of H2O from tetrahedral intermediate assisted by CH3SO3H proton shuttle",
        "reactant_fragments": ["CF3CH2OH (C2H3F3O, 9)", "HCOOH (CH2O2, 5)", "CH3SO3H (CH4O3S, 9)"],
        "product_fragments": ["HCOOCH2CF3 (C3H3F3O2, 11)", "H2O (H2O, 3)", "CH3SO3H (CH4O3S, 9)"],
    },
    "team4_ec_monomer": {
        "name": "team4_ec_monomer",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Cyclic Carbonate Monomer)",
        "team": "Team 4",
        "reaction_equation": "Ethylene carbonate monomer (C3H4O3)",
        "formula": "C3H4O3",
        "element_counts": {"C": 3, "H": 4, "O": 3},
        "n_atoms": 10,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "gas",
        "default_eps": 1.0,
        "default_temp_k": 298.15,
        "stationary_states": ["monomer"],
        "required_states": ["monomer"],
    },
    "team4_ec_cyclization": {
        "name": "team4_ec_cyclization",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Anionic Carbonate Transesterification - 5-exo-trig Cyclization)",
        "team": "Team 4",
        "reaction_equation": "-O-CH2CH2-OC(=O)OMe -> TS1 -> Int_tet -> TS2 -> EC + MeO-",
        "formula": "C4H7O4-",
        "element_counts": {"C": 4, "H": 7, "O": 4},
        "n_atoms": 15,
        "charge": -1,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": list(STATIONARY_STATES),
        "ts1_description": "Intramolecular 5-exo-trig nucleophilic attack of alkoxide O- onto carbonate carbonyl C",
        "int_description": "Cyclic anionic tetrahedral hemiorthocarbonate intermediate (2-methoxy-1,3-dioxolan-2-olate)",
        "ts2_description": "Ring-preserving C-OMe bond cleavage expelling methoxide MeO- to yield ethylene carbonate (EC)",
        "reactant_fragments": ["-O-CH2CH2-OC(=O)OMe (C4H7O4-, 15)"],
        "product_fragments": ["Ethylene carbonate (C3H4O3, 10)", "Methoxide MeO- (CH3O-, 5)"],
    },
    "team4_diol_dmc_intermolecular": {
        "name": "team4_diol_dmc_intermolecular",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Anionic Carbonate Transesterification - Intermolecular Step)",
        "team": "Team 4 (Step 1)",
        "reaction_equation": "HO-CH2CH2-O- + (CH3O)2C=O -> TS1 -> Int_tet -> TS2 -> HO-CH2CH2-OC(=O)OMe + MeO-",
        "formula": "C5H11O5-",
        "element_counts": {"C": 5, "H": 11, "O": 5},
        "n_atoms": 21,
        "charge": -1,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": ["R", "P"],
        "ts1_description": "Intermolecular nucleophilic attack of HO-CH2CH2-O- alkoxide onto dimethyl carbonate (DMC) carbonyl C",
        "int_description": "Acyclic anionic tetrahedral intermediate (HO-CH2CH2-O-C(O-)(OMe)2)",
        "ts2_description": "Elimination of methoxide MeO- yielding monocarbonate intermediate HO-CH2CH2-OC(=O)OMe",
        "reactant_fragments": ["HO-CH2CH2-O- (C2H5O2-, 9)", "DMC (C3H6O3, 12)"],
        "product_fragments": ["HO-CH2CH2-OC(=O)OMe (C4H8O4, 16)", "Methoxide MeO- (CH3O-, 5)"],
    },
    "team5_pc_monomer": {
        "name": "team5_pc_monomer",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Chiral Cyclic Carbonate Monomer)",
        "team": "Team 5",
        "reaction_equation": "(+-)-Propylene carbonate monomer (C4H6O3)",
        "formula": "C4H6O3",
        "element_counts": {"C": 4, "H": 6, "O": 3},
        "n_atoms": 13,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "gas",
        "default_eps": 1.0,
        "default_temp_k": 298.15,
        "stationary_states": ["monomer"],
        "required_states": ["monomer"],
    },
    "team5_pc_cyclization": {
        "name": "team5_pc_cyclization",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Anionic Chiral Cyclic Carbonate - 5-exo-trig Cyclization)",
        "team": "Team 5",
        "reaction_equation": "-O-CH2CH(CH3)-OC(=O)OMe -> TS1 -> Int_tet -> TS2 -> PC + MeO-",
        "formula": "C5H9O4-",
        "element_counts": {"C": 5, "H": 9, "O": 4},
        "n_atoms": 18,
        "charge": -1,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": ["R", "P"],
        "ts1_description": "Intramolecular 5-exo-trig nucleophilic attack of primary alkoxide O- onto carbonate carbonyl C",
        "int_description": "Chiral cyclic anionic tetrahedral hemiorthocarbonate intermediate (4-methyl-2-methoxy-1,3-dioxolan-2-olate)",
        "ts2_description": "Ring-preserving C-OMe bond cleavage expelling methoxide MeO- to yield (+-)-propylene carbonate (PC)",
        "reactant_fragments": ["-O-CH2CH(CH3)-OC(=O)OMe (C5H9O4-, 18)"],
        "product_fragments": ["Propylene carbonate (C4H6O3, 13)", "Methoxide MeO- (CH3O-, 5)"],
    },
    "team5_diol_dmc_intermolecular": {
        "name": "team5_diol_dmc_intermolecular",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Anionic Chiral Carbonate Transesterification - Intermolecular Step)",
        "team": "Team 5 (Step 1)",
        "reaction_equation": "HO-CH2CH(CH3)-O- + (CH3O)2C=O -> TS1 -> Int_tet -> TS2 -> HO-CH2CH(CH3)-OC(=O)OMe + MeO-",
        "formula": "C6H13O5-",
        "element_counts": {"C": 6, "H": 13, "O": 5},
        "n_atoms": 24,
        "charge": -1,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": ["R", "P"],
        "ts1_description": "Intermolecular nucleophilic attack of 1,2-propanediol monoalkoxide onto DMC carbonyl C",
        "int_description": "Acyclic anionic tetrahedral intermediate (HO-CH2CH(CH3)-O-C(O-)(OMe)2)",
        "ts2_description": "Elimination of methoxide MeO- yielding monocarbonate intermediate HO-CH2CH(CH3)-OC(=O)OMe",
        "reactant_fragments": ["HO-CH2CH(CH3)-O- (C3H7O2-, 12)", "DMC (C3H6O3, 12)"],
        "product_fragments": ["HO-CH2CH(CH3)-OC(=O)OMe (C5H10O4, 19)", "Methoxide MeO- (CH3O-, 5)"],
    },
    "team6_emc_monomer": {
        "name": "team6_emc_monomer",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Open-Chain Carbonate Monomer)",
        "team": "Team 6",
        "reaction_equation": "Ethyl methyl carbonate monomer (CH3CH2OC(=O)OCH3)",
        "formula": "C4H8O3",
        "element_counts": {"C": 4, "H": 8, "O": 3},
        "n_atoms": 15,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "gas",
        "default_eps": 1.0,
        "default_temp_k": 298.15,
        "stationary_states": ["monomer"],
        "required_states": ["monomer"],
    },
    "team6_emc_transesterification": {
        "name": "team6_emc_transesterification",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Anionic Open Carbonate Transesterification)",
        "team": "Team 6",
        "reaction_equation": "EtO- + (CH3O)2C=O -> TS1 -> Int_tet -> TS2 -> EMC + MeO-",
        "formula": "C5H11O4-",
        "element_counts": {"C": 5, "H": 11, "O": 4},
        "n_atoms": 20,
        "charge": -1,
        "multiplicity": 1,
        "default_solvent": "acetonitrile",
        "default_eps": 35.688,
        "default_temp_k": 298.15,
        "stationary_states": list(STATIONARY_STATES),
        "required_states": ["R", "P"],
        "ts1_description": "Nucleophilic attack of ethoxide EtO- onto dimethyl carbonate (DMC) carbonyl C",
        "int_description": "Acyclic anionic tetrahedral orthocarbonate intermediate (EtO-C(O-)(OMe)2)",
        "ts2_description": "Elimination of methoxide MeO- yielding ethyl methyl carbonate (EMC)",
        "reactant_fragments": ["Ethoxide EtO- (C2H5O-, 8)", "DMC (C3H6O3, 12)"],
        "product_fragments": ["Ethyl methyl carbonate (C4H8O3, 15)", "Methoxide MeO- (CH3O-, 5)"],
    },
    "team6_d4_ec_isotopologue": {
        "name": "team6_d4_ec_isotopologue",
        "course": "CHEM 312 Spring 2027",
        "institution": "Tennessee Tech University",
        "family": "Family B (Weeks 6-8 Deuterated Ethylene Carbonate Isotopologue)",
        "team": "Team 6 (Weeks 6-8)",
        "reaction_equation": "d4-Ethylene carbonate isotopologue (C3D4O3)",
        "formula": "C3D4O3",
        "element_counts": {"C": 3, "D": 4, "O": 3},
        "n_atoms": 10,
        "charge": 0,
        "multiplicity": 1,
        "default_solvent": "gas",
        "default_eps": 1.0,
        "default_temp_k": 298.15,
        "stationary_states": ["monomer"],
        "required_states": ["monomer"],
    },
}


def list_builtin_specs() -> list[str]:
    """Returns sorted names of all built-in CHEM 312 reaction specifications."""
    return sorted(BUILTIN_SPECS.keys())


def get_builtin_spec(name: str) -> dict[str, Any]:
    """Returns a copy of a built-in CHEM 312 reaction specification."""
    if name not in BUILTIN_SPECS:
        raise KeyError(f"Unknown built-in spec '{name}'. Available: {list_builtin_specs()}")
    return dict(BUILTIN_SPECS[name])


def resolve_solvent_eps(solvent_name: str, eps_override: float | None = None) -> tuple[str, float]:
    """Resolves solvent name and static dielectric constant."""
    if eps_override is not None:
        eps_val = float(eps_override)
        if eps_val < 1.0 or not math.isfinite(eps_val):
            raise ValueError(f"Dielectric constant eps must be finite and >= 1.0, got {eps_override}.")
        return solvent_name.strip().lower(), eps_val
    key = solvent_name.strip().lower()
    if key not in SOLVENT_DIELECTRICS:
        raise ValueError(
            f"Unknown solvent '{solvent_name}'. Known solvents: {sorted(SOLVENT_DIELECTRICS.keys())}, "
            "or pass an explicit eps_override."
        )
    return key, float(SOLVENT_DIELECTRICS[key])


def formula_from_symbols(symbols: list[str], charge: int = 0) -> str:
    """Formats Hill-system molecular formula (C, H, then alphabetical) with optional charge suffix."""
    counts = Counter(symbols)
    parts: list[str] = []
    if "C" in counts:
        parts.append("C" if counts["C"] == 1 else f"C{counts['C']}")
        if "H" in counts:
            parts.append("H" if counts["H"] == 1 else f"H{counts['H']}")
        for el in sorted(k for k in counts if k not in {"C", "H"}):
            parts.append(el if counts[el] == 1 else f"{el}{counts[el]}")
    else:
        for el in sorted(counts):
            parts.append(el if counts[el] == 1 else f"{el}{counts[el]}")
    suffix = ""
    if charge > 0:
        suffix = "+" if charge == 1 else f"{charge}+"
    elif charge < 0:
        suffix = "-" if charge == -1 else f"{abs(charge)}-"
    return "".join(parts) + suffix


def masses_from_symbols(symbols: list[str]) -> np.ndarray:
    """Returns atomic mass vector (amu) for element symbols."""
    masses = []
    for sym in symbols:
        if sym not in ATOMIC_MASSES_AMU:
            raise ValueError(f"Unsupported element symbol '{sym}' for atomic mass lookup.")
        masses.append(ATOMIC_MASSES_AMU[sym])
    return np.asarray(masses, dtype=np.float64)


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray, str]:
    """Reads a single-frame XYZ file, returning (symbols, coords_angstrom, comment)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"XYZ file not found: {p}")
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file {p} has fewer than 3 lines.")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid atom count on line 1 of {p}: {lines[0]!r}") from exc
    if n_atoms <= 0:
        raise ValueError(f"Atom count must be positive in {p}, got {n_atoms}.")
    comment = lines[1].strip()
    atom_lines = [ln.strip() for ln in lines[2 : 2 + n_atoms]]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"XYZ file {p} declares {n_atoms} atoms but contains {len(atom_lines)} rows.")

    symbols: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for idx, row in enumerate(atom_lines):
        fields = row.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ row {idx + 3} in {p}: {row!r}")
        sym = fields[0]
        try:
            xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
        except ValueError as exc:
            raise ValueError(f"Non-numeric coordinate on row {idx + 3} of {p}: {row!r}") from exc
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError(f"Non-finite coordinate on row {idx + 3} of {p}.")
        symbols.append(sym)
        coords[idx] = xyz
    return symbols, coords, comment


def write_xyz(
    path: str | Path,
    symbols: list[str],
    coords: np.ndarray,
    comment: str = "",
) -> Path:
    """Writes a single-frame XYZ file with full floating-point precision."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(coords, dtype=np.float64)
    if arr.shape != (len(symbols), 3):
        raise ValueError(f"Coordinate shape {arr.shape} does not match {len(symbols)} symbols.")
    lines = [str(len(symbols)), comment]
    for sym, xyz in zip(symbols, arr):
        lines.append(f"{sym:<2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def validate_electron_spin_consistency(symbols: list[str], charge: int, multiplicity: int) -> int:
    """Validates total electron count parity against spin multiplicity 2S + 1."""
    if multiplicity < 1:
        raise ValueError(f"Spin multiplicity must be >= 1, got {multiplicity}.")
    total_z = 0
    for sym in symbols:
        if sym not in ATOMIC_NUMBERS:
            raise ValueError(f"Unknown element symbol '{sym}' for electron parity check.")
        total_z += ATOMIC_NUMBERS[sym]
    n_electrons = total_z - int(charge)
    if n_electrons <= 0:
        raise ValueError(f"Non-positive electron count ({n_electrons}) for charge={charge}.")
    n_unpaired = int(multiplicity) - 1
    if (n_electrons - n_unpaired) % 2 != 0:
        raise ValueError(
            f"Electron/spin parity mismatch: {n_electrons} electrons cannot have multiplicity {multiplicity}."
        )
    return n_electrons


def validate_stationary_stoichiometry(
    states: dict[str, dict[str, Any]],
    expected_charge: int,
    expected_multiplicity: int = 1,
    expected_formula: str | None = None,
    require_same_atom_order: bool = True,
    min_interatomic_dist_ang: float = 0.40,
) -> dict[str, Any]:
    """
    Strictly validates atom count, element composition, atom ordering (optional),
    charge, spin multiplicity, electron parity, and nuclear clash absence across
    all stationary points (e.g. R, TS1, Int_tet, TS2, P).

    Each value in `states` must contain:
      - "symbols": list[str]
      - "coords": np.ndarray of shape (N, 3)
      - optional "charge": int (defaults to expected_charge if omitted, checked if present)
      - optional "multiplicity": int (defaults to expected_multiplicity if omitted, checked if present)
    """
    if not states:
        raise ValueError("No stationary states supplied for stoichiometry validation.")

    ref_name: str | None = None
    ref_symbols: list[str] | None = None
    ref_counts: Counter[str] | None = None
    summary: dict[str, Any] = {
        "valid": True,
        "states_checked": list(states.keys()),
        "charge": int(expected_charge),
        "multiplicity": int(expected_multiplicity),
        "per_state": {},
    }

    for state_name, entry in states.items():
        symbols = list(entry["symbols"])
        coords = np.asarray(entry["coords"], dtype=np.float64)
        st_charge = int(entry.get("charge", expected_charge))
        st_mult = int(entry.get("multiplicity", expected_multiplicity))

        if coords.ndim != 2 or coords.shape != (len(symbols), 3):
            raise ValueError(
                f"State '{state_name}': coords shape {coords.shape} does not match {len(symbols)} atoms."
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"State '{state_name}': non-finite coordinates detected.")

        if st_charge != int(expected_charge):
            raise ValueError(
                f"Charge mismatch in state '{state_name}': got Q={st_charge}, expected Q={expected_charge}."
            )
        if st_mult != int(expected_multiplicity):
            raise ValueError(
                f"Multiplicity mismatch in state '{state_name}': got M={st_mult}, expected M={expected_multiplicity}."
            )

        n_elec = validate_electron_spin_consistency(symbols, st_charge, st_mult)
        formula = formula_from_symbols(symbols, charge=st_charge)
        counts = Counter(symbols)

        if expected_formula is not None and formula != expected_formula:
            raise ValueError(
                f"Formula mismatch in state '{state_name}': got {formula}, expected {expected_formula}."
            )

        if ref_symbols is None:
            ref_name = state_name
            ref_symbols = symbols
            ref_counts = counts
        else:
            if len(symbols) != len(ref_symbols):
                raise ValueError(
                    f"Atom count mismatch: '{state_name}' has {len(symbols)} atoms vs '{ref_name}' ({len(ref_symbols)})."
                )
            if counts != ref_counts:
                raise ValueError(
                    f"Element composition mismatch: '{state_name}' ({dict(counts)}) vs '{ref_name}' ({dict(ref_counts)})."
                )
            if require_same_atom_order and symbols != ref_symbols:
                raise ValueError(
                    f"Atom ordering mismatch between '{state_name}' and '{ref_name}'. "
                    "Stationary points for CI-NEB/IRC must preserve consistent atom indices."
                )

        # Nuclear clash check
        min_dist = float("inf")
        clash_pair: tuple[int, int] | None = None
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < min_dist:
                    min_dist = d
                    clash_pair = (i, j)
        if min_dist < min_interatomic_dist_ang:
            raise ValueError(
                f"Nuclear clash in state '{state_name}': atoms {clash_pair} separated by {min_dist:.4f} A "
                f"(< {min_interatomic_dist_ang:.2f} A)."
            )

        summary["per_state"][state_name] = {
            "n_atoms": len(symbols),
            "formula": formula,
            "charge": st_charge,
            "multiplicity": st_mult,
            "n_electrons": n_elec,
            "min_interatomic_dist_ang": round(min_dist, 6),
        }

    assert ref_symbols is not None
    summary["n_atoms"] = len(ref_symbols)
    summary["formula"] = formula_from_symbols(ref_symbols, charge=int(expected_charge))
    summary["element_counts"] = dict(Counter(ref_symbols))
    return summary
