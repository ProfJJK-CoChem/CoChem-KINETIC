"""
CHEM 312 Spring 2027 All-6-Team Input Matrix & AIMNet2 Qualification Tests
--------------------------------------------------------------------------
Zero-mock test suite validating:
  1. `cochem-kinetic list-specs` registers all 6 CHEM 312 Spring 2027 teams
     (monomers, Week 10 reactions, Team 4/5 intermolecular steps, and Team 6 d4-EC).
  2. `cochem-kinetic validate-stoich` passes on every Team 1-6 monomer and Week 10 pair
     (exact atom count, Hill formula, charge, multiplicity, electron parity, and atom order).
  3. `benchmarks/chem312_teams_matrix/teams_qualification_manifest.json` SHA-256 integrity
     across all 29 staged XYZ files, finite AIMNet2 energies & converged forces (fmax < 0.02 eV/A),
     and honest VERIFIED (Teams 1 & 4) vs PENDING_STUDENT_RUN (Teams 2, 3, 5, 6) status.
  4. Collaborator institution lock: Tennessee Tech University ONLY (zero "Texas Tech").
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cochem_kinetic.chem312_specs import BUILTIN_SPECS, list_builtin_specs, read_xyz
from cochem_kinetic.cli import main as cli_main
from cochem_kinetic.rxn_profile import sha256_file

REPO_ROOT = Path(__file__).parent.parent
MATRIX_ROOT = REPO_ROOT / "benchmarks" / "chem312_teams_matrix"
MANIFEST_PATH = MATRIX_ROOT / "teams_qualification_manifest.json"


EXPECTED_TEAM_SPECS: list[tuple[str, str, int, int, str]] = [
    # (spec_name, relative_dir, n_atoms, charge, formula)
    ("team1_propargyl_formate_monomer", "team1_propargyl_formate/monomer", 10, 0, "C4H4O2"),
    ("team1_propargyl_formate", "team1_propargyl_formate/week10", 22, 0, "C5H10O6S"),
    ("team2_difluoroethyl_formate_monomer", "team2_difluoroethyl_formate/monomer", 11, 0, "C3H4F2O2"),
    ("team2_difluoroethyl_formate", "team2_difluoroethyl_formate/week10", 23, 0, "C4H10F2O6S"),
    ("team3_trifluoroethyl_formate_monomer", "team3_trifluoroethyl_formate/monomer", 11, 0, "C3H3F3O2"),
    ("team3_trifluoroethyl_formate", "team3_trifluoroethyl_formate/week10", 23, 0, "C4H9F3O6S"),
    ("team4_ec_monomer", "team4_ethylene_carbonate/monomer", 10, 0, "C3H4O3"),
    ("team4_ec_cyclization", "team4_ethylene_carbonate/week10", 15, -1, "C4H7O4-"),
    ("team4_diol_dmc_intermolecular", "team4_ethylene_carbonate/intermolecular_21atom", 21, -1, "C5H11O5-"),
    ("team5_pc_monomer", "team5_propylene_carbonate/monomer", 13, 0, "C4H6O3"),
    ("team5_pc_cyclization", "team5_propylene_carbonate/week10", 18, -1, "C5H9O4-"),
    ("team5_diol_dmc_intermolecular", "team5_propylene_carbonate/intermolecular_24atom", 24, -1, "C6H13O5-"),
    ("team6_emc_monomer", "team6_emc_d4ec/monomer", 15, 0, "C4H8O3"),
    ("team6_emc_transesterification", "team6_emc_d4ec/week10", 20, -1, "C5H11O4-"),
    ("team6_d4_ec_isotopologue", "team6_emc_d4ec/d4_ec_isotopologue", 10, 0, "C3D4O3"),
]


def test_list_specs_registers_all_six_chem312_teams(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `cochem-kinetic list-specs` prints every CHEM 312 Team 1-6 spec."""
    assert cli_main(["list-specs"]) == 0
    out = capsys.readouterr().out
    registered = set(list_builtin_specs())
    for spec_name, _rel_dir, n_atoms, charge, formula in EXPECTED_TEAM_SPECS:
        assert spec_name in registered
        assert spec_name in out
        spec = BUILTIN_SPECS[spec_name]
        assert (spec["n_atoms"], spec["charge"], spec["multiplicity"], spec["formula"]) == (
            n_atoms,
            charge,
            1,
            formula,
        )
        assert spec["institution"] == "Tennessee Tech University"


@pytest.mark.parametrize(
    ("spec_name", "rel_dir", "n_atoms", "charge", "formula"),
    EXPECTED_TEAM_SPECS,
)
def test_validate_stoich_passes_every_team_monomer_and_week10_pair(
    spec_name: str,
    rel_dir: str,
    n_atoms: int,
    charge: int,
    formula: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `cochem-kinetic validate-stoich` passes on every Team 1-6 monomer and Week 10 pair."""
    target_dir = MATRIX_ROOT / rel_dir
    assert target_dir.is_dir(), f"Missing staged directory: {target_dir}"
    rc = cli_main(["validate-stoich", "--input-dir", str(target_dir), "--spec", spec_name])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["n_atoms"] == n_atoms
    assert payload["charge"] == charge
    assert payload["multiplicity"] == 1
    assert payload["formula"] == formula

    # Confirm exact 1-to-1 atom ordering across R and P whenever both are staged
    if "R" in payload["per_state"] and "P" in payload["per_state"]:
        r_syms, _, _ = read_xyz(target_dir / "R.xyz")
        p_syms, _, _ = read_xyz(target_dir / "P.xyz")
        assert r_syms == p_syms


def test_teams_qualification_manifest_sha256_aimnet2_and_honest_status() -> None:
    """Verify SHA-256 integrity, finite AIMNet2 energies/forces (<0.02 eV/A), and VERIFIED vs PENDING_STUDENT_RUN."""
    assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["institution"] == "Tennessee Tech University"
    assert manifest["course"] == "CHEM 312 Spring 2027"
    assert set(manifest["elements_qualified"]) >= {"C", "H", "D", "O", "F", "S"}
    assert set(manifest["charges_qualified"]) >= {-1, 0}

    # Verify every staged XYZ file hash matches manifest
    files_sha256 = manifest["files_sha256"]
    assert len(files_sha256) == 29
    for rel_path, expected_digest in files_sha256.items():
        fpath = MATRIX_ROOT / rel_path
        assert fpath.is_file(), f"Missing manifest XYZ file: {fpath}"
        assert sha256_file(fpath) == expected_digest, f"SHA-256 mismatch for {rel_path}"

    # Verify honest DFT/TS/IRC status per team and converged AIMNet2 energies/forces
    expected_statuses = {
        "team1_propargyl_formate": "VERIFIED",
        "team2_difluoroethyl_formate": "PENDING_STUDENT_RUN",
        "team3_trifluoroethyl_formate": "PENDING_STUDENT_RUN",
        "team4_ethylene_carbonate": "VERIFIED",
        "team5_propylene_carbonate": "PENDING_STUDENT_RUN",
        "team6_emc_d4ec": "PENDING_STUDENT_RUN",
    }
    teams = manifest["teams"]
    assert set(teams.keys()) == set(expected_statuses.keys())

    for team_key, expected_status in expected_statuses.items():
        entry = teams[team_key]
        assert entry["dft_ts_irc_status"] == expected_status

        # Check monomer + Week 10 R & P AIMNet2 records
        records = [
            entry["monomer"]["aimnet2"],
            entry["week10"]["R"]["aimnet2"],
            entry["week10"]["P"]["aimnet2"],
        ]
        if "intermolecular_21atom" in entry:
            records.extend([
                entry["intermolecular_21atom"]["R"]["aimnet2"],
                entry["intermolecular_21atom"]["P"]["aimnet2"],
            ])
        if "intermolecular_24atom" in entry:
            records.extend([
                entry["intermolecular_24atom"]["R"]["aimnet2"],
                entry["intermolecular_24atom"]["P"]["aimnet2"],
            ])
        if "d4_ec_isotopologue" in entry:
            records.append(entry["d4_ec_isotopologue"]["aimnet2"])
            assert entry["d4_ec_isotopologue"]["born_oppenheimer_delta_e_ev"] == 0.0
            assert entry["d4_ec_isotopologue"]["deuterium_mass_amu"] > 2.0

        for rec in records:
            assert math.isfinite(rec["energy_ev"]) and rec["energy_ev"] < 0.0
            assert math.isfinite(rec["energy_ha"]) and rec["energy_ha"] < 0.0
            assert math.isfinite(rec["fmax_ev_ang"]) and 0.0 <= rec["fmax_ev_ang"] < 0.02
            assert math.isfinite(rec["frms_ev_ang"]) and 0.0 <= rec["frms_ev_ang"] < 0.02
            assert rec["converged"] is True
            assert rec["wall_seconds"] >= 0.0


def test_zero_texas_tech_references_in_specs_and_matrix() -> None:
    """Enforce Tennessee Tech University collaborator institution lock (zero 'Texas Tech')."""
    scan_paths = [
        REPO_ROOT / "cochem_kinetic" / "chem312_specs.py",
        REPO_ROOT / "cochem_kinetic" / "cli.py",
        MANIFEST_PATH,
        MATRIX_ROOT / "README.md",
    ]
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        assert "texas tech" not in text.lower(), f"Forbidden 'Texas Tech' reference found in {path}"
