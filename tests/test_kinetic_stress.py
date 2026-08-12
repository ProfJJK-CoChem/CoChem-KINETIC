import hashlib
from typing import Any, Dict, List, Optional
#!/usr/bin/env python3
"""
CoChem-KINETIC Empirical Stress Test Suite
------------------------------------------
Comprehensive adversarial and edge-case stress harness testing:
    1. PESStore creation, concurrent writes, large dataset append, corruption checks, HDF5 compression and Fletcher32 checksum validation.
2. mace_pre_opt.py optimization under noisy gradients (Float32 noise floor edge cases, fmax floor, sign-flip halting).
3. State chaining rules D1-D5 in dispatcher.py with edge-case stationary point gradients, frequency shifts, density differences, and ghost atom mismatches.
4. Bead count calculator for high/low temperatures (10 K, 300 K, 1000 K) and high/low vibrational frequencies.
5. Rotational constant warning & [E] tag formatting in aimd_hoover.py.
"""

import os
import shutil
import tempfile
import threading
from pathlib import Path
import numpy as np
import pytest
import h5py

from kinetic_core.cochem_pes_store import PESStore
from kinetic_core.mace_pre_opt import MACEPreOptimizer
from kinetic_core.dispatcher import KineticDispatcher, V4_WALLCLOCK_BUDGETS, TIER_HIERARCHY
from kinetic_core.aimd_hoover import calculate_required_beads, compute_rotational_constants_md, NoseHooverAIMDSampler


# ==============================================================================
# 1. PESStore Creation, Concurrency, Large Dataset, Compression & Fletcher32 Tests
# ==============================================================================

class TestPESStoreStress:
    """Stress tests for PESStore HDF5 interface (§8C)."""

    def test_pes_store_creation_and_attributes(self, tmp_path) -> None:
        h5_file = tmp_path / "pes_test_attr.h5"
        store = PESStore(h5_file, provenance_tag="[M]")
        
        with h5py.File(h5_file, "r") as f:
            for grp_path in ["pes/grid", "pes/fit", "pes/uncertainty"]:
                assert grp_path in f
                grp = f[grp_path]
                assert grp.attrs["qcschema_version"] == 2
                assert grp.attrs["provenance_tag"] == "[M]"

    def test_pes_store_readonly_mode_bug(self, tmp_path) -> None:
        """
        Tests opening PESStore in read-only mode ('r').
        Empirical finding: PESStore.__init__ attempts to set attributes on group init,
        which fails if the file is opened read-only.
        """
        h5_file = tmp_path / "pes_ro.h5"
        # First create the file
        store = PESStore(h5_file, mode="a")
        store.append_point(np.zeros((2, 3)), energy=-76.0)

        # Attempt to open read-only
        try:
            store_ro = PESStore(h5_file, mode="r")
            grid = store_ro.get_grid()
            assert grid["coordinates"].shape == (1, 2, 3)
        except KeyError as e:
            pytest.fail(f"PESStore failed to open in read-only mode 'r': KeyError ({e})")

    def test_pes_store_hdf5_compression_shuffle_fletcher32(self, tmp_path) -> None:
        h5_file = tmp_path / "pes_compression.h5"
        store = PESStore(h5_file)
        
        coords = np.random.randn(100, 3, 3)
        energies = np.random.randn(100)
        grads = np.random.randn(100, 3, 3)
        store.append_batch(coords, energies, gradient_batch=grads, group="grid")
        
        with h5py.File(h5_file, "r") as f:
            ds_c = f["/pes/grid/coordinates"]
            ds_e = f["/pes/grid/energies"]
            ds_g = f["/pes/grid/gradients"]
            
            # Verify 512-point chunking, gzip level 4, shuffle, and fletcher32
            assert ds_c.chunks[0] == 512
            assert ds_c.compression == "gzip"
            assert ds_c.compression_opts == 4
            assert ds_c.shuffle is True
            assert ds_c.fletcher32 is True
            
            assert ds_e.chunks[0] == 512
            assert ds_e.fletcher32 is True

    def test_pes_store_large_dataset_append(self, tmp_path) -> None:
        h5_file = tmp_path / "pes_large.h5"
        store = PESStore(h5_file)
        
        # Append 5,500 points in 11 batches of 500
        n_batches = 11
        batch_size = 500
        for b in range(n_batches):
            coords = np.random.randn(batch_size, 4, 3)
            energies = np.full(batch_size, -100.0 + b)
            store.append_batch(coords, energies, group="grid")
            
        grid = store.get_grid()
        assert grid["coordinates"].shape == (5500, 4, 3)
        assert grid["energies"].shape == (5500,)
        assert grid["energies"][-1] == -100.0 + 10

    def test_pes_store_single_point_and_uncertainty(self, tmp_path) -> None:
        h5_file = tmp_path / "pes_single.h5"
        store = PESStore(h5_file)
        
        coords = np.zeros((2, 3))
        store.append_point(coords, energy=-76.4, variance=0.001, group="uncertainty")
        
        unc = store.get_uncertainty()
        assert unc["coordinates"].shape == (1, 2, 3)
        assert unc["energies"].shape == (1,)
        assert unc["variance"].shape == (1,)
        assert np.isclose(unc["variance"][0], 0.001)

    def test_pes_store_fletcher32_checksum_validation_on_corruption(self, tmp_path) -> None:
        h5_file = tmp_path / "pes_corrupt.h5"
        store = PESStore(h5_file)
        
        # Write large batch to fill at least one chunk (chunk size = 512)
        coords = np.ones((600, 3, 3), dtype=np.float64) * 1.5
        energies = np.ones(600, dtype=np.float64) * -50.0
        store.append_batch(coords, energies, group="grid")
        
        # Corrupt data bytes directly in the HDF5 dataset raw storage area
        with open(h5_file, "r+b") as f:
            f.seek(2000)
            f.write(b"\xFF" * 500)

        # Attempting to read corrupted dataset chunk via h5py must trigger HDF5 corruption error (OSError or KeyError)
        with pytest.raises((OSError, KeyError)) as exc_info:
            with h5py.File(h5_file, "r") as h5_read:
                _ = h5_read["/pes/grid/coordinates"][:]
        assert exc_info.value is not None

    def test_pes_store_concurrent_thread_writes(self, tmp_path) -> None:
        """
        Tests multi-threaded concurrent write safety.
        Empirical finding: Race condition in _get_or_create_dataset when dataset does not exist yet.
        """
        h5_file = tmp_path / "pes_concurrent.h5"
        store = PESStore(h5_file)
        errors = []

        def worker_append(thread_id) -> Any:
            try:
                coords = np.full((10, 2, 3), float(thread_id))
                energies = np.full(10, float(thread_id))
                store.append_batch(coords, energies, group="grid")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_append, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            pytest.fail(f"PESStore concurrent write failed with {len(errors)} errors. First error: {errors[0]}")

        grid = store.get_grid()
        assert grid["coordinates"].shape == (100, 2, 3)
        assert grid["energies"].shape == (100,)


# ==============================================================================
# 2. MACE Pre-Optimizer Noisy Gradient, Noise Guard & Fallback Tests
# ==============================================================================

class TestMACEPreOptimizerStress:
    """Stress tests for MACE-OFF24m pre-optimizer and Float32 guards (§8A.2, §16.1)."""

    def test_float32_noise_guard_energy_floor(self) -> None:
        pre_opt = MACEPreOptimizer()
        
        # Energy change below 1e-5 -> True (stop optimization)
        stopped = pre_opt.apply_float32_noise_guard(
            prev_grad=np.array([0.01, 0.01]),
            curr_grad=np.array([0.01, 0.01]),
            prev_energy=-100.000000,
            curr_energy=-100.000005  # diff = 5e-6 < 1e-5
        )
        assert stopped is True

        # Energy change above 1e-5 -> False (continue)
        stopped = pre_opt.apply_float32_noise_guard(
            prev_grad=np.array([0.01, 0.01]),
            curr_grad=np.array([0.01, 0.01]),
            prev_energy=-100.00000,
            curr_energy=-100.00003  # diff = 3e-5 > 1e-5
        )
        assert stopped is False

    def test_float32_noise_guard_gradient_sign_flip(self) -> None:
        pre_opt = MACEPreOptimizer()
        
        # Low force regime (< 1e-3) with gradient sign flip (dot product < -0.2 * n1 * n2)
        prev_g = np.array([0.0005, 0.0000])
        curr_g = np.array([-0.0005, 0.0000])  # dot product = -2.5e-7, -0.2*n1*n2 = -5e-8 -> dot < threshold
        stopped = pre_opt.apply_float32_noise_guard(
            prev_grad=prev_g,
            curr_grad=curr_g,
            prev_energy=-100.0,
            curr_energy=-100.0001
        )
        assert stopped is True

    def test_float32_noise_guard_edge_cases(self) -> None:
        pre_opt = MACEPreOptimizer()
        
        # Zero gradients (norms = 0)
        prev_g = np.zeros((3, 3))
        curr_g = np.zeros((3, 3))
        stopped = pre_opt.apply_float32_noise_guard(prev_g, curr_g, prev_energy=None, curr_energy=-10.0)
        assert stopped is False

        # First step (prev_energy and prev_grad are None)
        stopped = pre_opt.apply_float32_noise_guard(None, curr_g, prev_energy=None, curr_energy=-10.0)
        assert stopped is False

    def test_fmax_floor_guard(self) -> None:
        pre_opt = MACEPreOptimizer()
        # Ensure fmax passed as 1e-6 is guarded to max(1e-6, 1e-3) = 1e-3
        syms = ["O", "H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]])
        
        rel_coords, e = pre_opt.pre_relax_geometry(syms, coords, max_steps=2, fmax=1e-6)
        assert rel_coords.shape == (3, 3)
        assert isinstance(e, float)

    def test_absence_of_pairwise_lennard_jones_potential_formulas(self) -> None:
        """
        Verifies zero pairwise Lennard-Jones potential calculation formulas exist in code per §8A.2.
        Ignores docstring text prohibiting LJ.
        """
        code_path = Path(__file__).parent.parent / "kinetic_core" / "mace_pre_opt.py"
        with open(code_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Filter out docstrings and comments
        code_lines = [line for line in lines if not line.strip().startswith("#") and '"""' not in line]
        code_text = "\n".join(code_lines)
        
        assert "sigma_default" not in code_text
        assert "epsilon_default" not in code_text
        assert "12 - 6" not in code_text
        assert "r_ij" not in code_text


# ==============================================================================
# 3. State Chaining Rules D1-D5 Stress Tests
# ==============================================================================

class TestStateChainingRulesStress:
    """Stress tests for state chaining rules D1-D5 in dispatcher.py (§8B)."""

    @pytest.fixture
    def dispatcher(self) -> Any:
        return KineticDispatcher()

    # --- Rule D1: Stationary Point Check ---
    def test_rule_d1_stationary_point(self, dispatcher) -> None:
        # Grad norm < 1e-4 -> Pass
        g_pass = np.array([[1e-5, 2e-5, -1e-5], [0.0, 3e-5, 1e-5]])
        assert dispatcher.check_d1_stationary_point(g_pass, tol=1e-4) is True

        # Grad norm > 1e-4 -> Fail
        g_fail = np.array([[1e-3, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert dispatcher.check_d1_stationary_point(g_fail, tol=1e-4) is False

        # Edge cases: empty gradient
        assert dispatcher.check_d1_stationary_point(np.array([])) is False
        assert dispatcher.check_d1_stationary_point(None) is False

        # 1D gradient array
        g_1d = np.array([1e-5, 2e-5, -1e-5])
        assert dispatcher.check_d1_stationary_point(g_1d, tol=1e-4) is True

    # --- Rule D2: Hessian Mode Match ---
    def test_rule_d2_hessian_mode_match(self, dispatcher) -> None:
        # TS mode (is_ts=True): requires exactly 1 imaginary freq
        freqs_prev_ts = np.array([-300.0, 150.0, 300.0])
        freqs_curr_ts = np.array([-320.0, 155.0, 310.0])
        assert dispatcher.check_d2_hessian_mode_match(freqs_prev_ts, freqs_curr_ts, is_ts=True, tol=50.0) is True

        # TS mode fail: 0 imaginary freqs
        freqs_no_imag = np.array([100.0, 200.0, 300.0])
        assert dispatcher.check_d2_hessian_mode_match(freqs_no_imag, freqs_curr_ts, is_ts=True) is False

        # Minimum mode (is_ts=False): requires 0 imaginary freqs
        freqs_min1 = np.array([100.0, 200.0, 300.0])
        freqs_min2 = np.array([120.0, 205.0, 298.0])
        assert dispatcher.check_d2_hessian_mode_match(freqs_min1, freqs_min2, is_ts=False, tol=50.0) is True

        # Frequency difference exceeding tolerance
        freqs_far = np.array([200.0, 200.0, 300.0])
        assert dispatcher.check_d2_hessian_mode_match(freqs_min1, freqs_far, is_ts=False, tol=50.0) is False

    def test_rule_d2_empty_frequencies_edge_case(self, dispatcher) -> None:
        """
        Edge case test: Empty frequency array should return False without crashing.
        Empirical finding: Currently crashes with IndexError on freqs_prev[0].
        """
        try:
            res = dispatcher.check_d2_hessian_mode_match([], [], is_ts=False)
            assert res is False
        except IndexError as e:
            pytest.fail(f"Rule D2 crashed with IndexError on empty frequency arrays: {e}")

    # --- Rule D3: SCF Basin Check ---
    def test_rule_d3_scf_basin(self, dispatcher) -> None:
        d1 = np.eye(4) * 0.5
        d2 = d1 + np.eye(4) * 1e-4  # norm diff = sqrt(4 * 1e-8) = 2e-4 < 1e-3
        assert dispatcher.check_d3_scf_basin(d1, d2, tol=1e-3) is True

        d3 = d1 + np.eye(4) * 0.1  # norm diff = 0.2 > 1e-3
        assert dispatcher.check_d3_scf_basin(d1, d3, tol=1e-3) is False

    def test_rule_d3_shape_mismatch_edge_case(self, dispatcher) -> None:
        """
        Edge case test: Density matrices with different dimensions (e.g. cross-tier basis upgrade).
        Empirical finding: Currently crashes with ValueError (cannot subtract 3x3 from 5x5).
        """
        d_small = np.eye(3)
        d_large = np.eye(5)
        try:
            res = dispatcher.check_d3_scf_basin(d_small, d_large)
            assert res is False
        except ValueError as e:
            pytest.fail(f"Rule D3 crashed with ValueError on density matrix shape mismatch across tiers: {e}")

    # --- Rule D4: Counterpoise Ghost Match ---
    def test_rule_d4_cp_ghost_match(self, dispatcher) -> None:
        ghosts1 = [1, 3, 5]
        ghosts2 = [5, 1, 3]  # Same indices, different order
        assert dispatcher.check_d4_cp_ghost_match(ghosts1, ghosts2) is True

        ghosts3 = [1, 3, 4]
        assert dispatcher.check_d4_cp_ghost_match(ghosts1, ghosts3) is False

    def test_rule_d4_none_ghost_atoms_edge_case(self, dispatcher) -> None:
        """Edge case test: None passed for ghost atom list."""
        try:
            res = dispatcher.check_d4_cp_ghost_match(None, [1, 2])
            assert res is False
        except TypeError as e:
            pytest.fail(f"Rule D4 crashed with TypeError when None passed for ghost atoms: {e}")

    # --- Rule D5: Safe Overwrite ---
    def test_rule_d5_safe_overwrite(self, dispatcher) -> None:
        # Higher or equal tier can overwrite
        meta_prev = {"wallclock_tier": "T1-30min"}
        meta_curr = {"wallclock_tier": "T2-1h"}
        assert dispatcher.check_d5_safe_overwrite(meta_prev, meta_curr) is True

        # Same tier can overwrite
        meta_same = {"wallclock_tier": "T1-30min"}
        assert dispatcher.check_d5_safe_overwrite(meta_prev, meta_same) is True

        # Lower tier CANNOT overwrite higher tier
        meta_lower = {"wallclock_tier": "T1-10s"}
        assert dispatcher.check_d5_safe_overwrite(meta_prev, meta_lower) is False

    def test_v4_wallclock_budgets_mapping(self, dispatcher) -> None:
        assert len(V4_WALLCLOCK_BUDGETS) == 10
        assert dispatcher.get_tier_budget("T1-10s") == 10.0
        assert dispatcher.get_tier_budget("T1-30min") == 1800.0
        assert dispatcher.get_tier_budget("T4-1mo") == 2592000.0
        with pytest.raises(ValueError):
            dispatcher.get_tier_budget("INVALID_TIER")


# ==============================================================================
# 4. Bead Count Calculator Stress Tests
# ==============================================================================

class TestBeadCountCalculatorStress:
    """Stress tests for path-integral bead count calculator (§5.6, §11.2)."""

    @pytest.mark.parametrize(
        "temp_k, w_max_cm1, expected_min_beads",
        [
            (10.0, 3000.0, 949),   # Very low temp -> high bead count (~950)
            (100.0, 3000.0, 95),   # Low temp -> ~95 beads
            (300.0, 3000.0, 32),   # Standard room temp -> ~32 beads
            (1000.0, 3000.0, 10),  # High temp -> ~10 beads
            (300.0, 200.0, 3),     # Low frequency at 300 K -> ~3 beads
            (1000.0, 200.0, 1),    # High temp, low freq -> 1 bead
        ],
    )
    def test_bead_count_temperature_and_frequency_sweeps(self, temp_k, w_max_cm1, expected_min_beads) -> None:
        beads = calculate_required_beads(temp_k, w_max_cm1)
        assert isinstance(beads, int)
        assert beads >= expected_min_beads

    def test_bead_count_zero_or_negative_temperature(self) -> None:
        with pytest.raises(ValueError):
            calculate_required_beads(0.0, 3000.0)
        with pytest.raises(ValueError):
            calculate_required_beads(-100.0, 3000.0)


# ==============================================================================
# 5. Rotational Constant & Classical MD Formatting Stress Tests
# ==============================================================================

class TestRotationalConstantMDFormattingStress:
    """Stress tests for rotational constant B0 prohibition guard (§11.1, §13.4)."""

    def test_classical_md_b0_prohibition(self) -> None:
        # Water molecule geometry
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]])
        masses = np.array([15.999, 1.008, 1.008])

        # Classical MD = True -> MUST output provenance [E], is_absolute_b0=False, and warning string
        res = compute_rotational_constants_md(coords, masses, is_classical_md=True)
        assert res["provenance_tag"] == "[E]"
        assert res["is_absolute_b0"] is False
        assert "WARNING: Classical MD cannot yield absolute B0" in res["warning"]
        assert len(res["rotational_constants_cm1"]) == 3

    def test_non_classical_md_b0_allowed(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]])
        masses = np.array([15.999, 1.008, 1.008])

        # Classical MD = False -> Output provenance [M], is_absolute_b0=True, no warning
        res = compute_rotational_constants_md(coords, masses, is_classical_md=False)
        assert res["provenance_tag"] == "[M]"
        assert res["is_absolute_b0"] is True
        assert res["warning"] is None

    def test_linear_molecule_rotational_constants(self) -> None:
        # CO2 linear geometry along Z axis
        coords = np.array([[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
        masses = np.array([15.999, 12.011, 15.999])

        res = compute_rotational_constants_md(coords, masses, is_classical_md=True)
        rot_c = res["rotational_constants_cm1"]
        # Linear molecule has first moment near zero -> rot_c[0] should be 0.0
        assert rot_c[0] == 0.0
        assert rot_c[1] > 0.0
        assert np.isclose(rot_c[1], rot_c[2], rtol=1e-3)

    def test_nose_hoover_aimd_transition_metal_dt_scaling(self) -> None:
        sampler = NoseHooverAIMDSampler(target_temp_k=300.0)
        
        # Non-transition metal species (H2O) -> default dt (1.0 fs)
        dt_h2o = sampler._determine_timestep(["O", "H", "H"])
        assert dt_h2o == 1.0

        # Transition metal species (Fe, Cu, etc.) -> scaled dt (0.5 fs)
        dt_fe = sampler._determine_timestep(["Fe", "O", "O"])
        assert dt_fe == 0.5
def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()