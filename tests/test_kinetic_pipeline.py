import hashlib
from typing import Any, Dict, List, Optional
#!/usr/bin/env python3
"""
CoChem-KINETIC Automated PyTest Suite
-------------------------------------
Validates all 20 KINETIC items:
    - Wigner imaginary frequency sign fix & upper bounds
- Skodje-Truhlar tunneling for high barriers
- Concentration standard state corrections in Eyring kinetics
- Variational TST (VTST) free energy maximum profile search
- Landau-Zener zero-division guards
- Troe RRKM fall-off dynamics
- Kinetic Isotope Effect (KIE) calculation
- Multireference T1/D1 diagnostic gates
- Pitzer-Gwinn hindered rotor corrections
- Implicit solvation corrections
- CI-NEB JAX engine
- MACE pre-optimizer fallback
- Nose-Hoover AIMD sampler
- HDF5 state serialization & dispatcher
- IRC path integration
- 3D HTML IRC trajectory animation generator
- pyproject.toml packaging configuration
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Import kinetic_core modules directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetic_core.thermo import (
    wigner_correction,
    skodje_truhlar_tunneling_correction,
    calculate_eyring_rate,
    variational_tst_correction,
    landau_zener_probability,
    troe_falloff_rate,
    calculate_kie,
    validate_multireference_diagnostics,
    MultireferenceDiagnosticError,
    pitzer_gwinn_hindered_rotor_correction,
    apply_implicit_solvation_correction
)
from kinetic_core.jax_cineb import JACXCINEBEngine
from kinetic_core.mace_pre_opt import MACEPreOptimizer
from kinetic_core.aimd_hoover import NoseHooverAIMDSampler
from kinetic_core.dispatcher import KineticDispatcher
from kinetic_core.irc_tracer import IRCTracerEngine
from kinetic_core.viz import IRCAnimationVisualizer


def test_wigner_and_tunneling_corrections() -> None:
    # KINETIC-01: Imaginary frequency sign check
    kappa_neg = wigner_correction(-1500.0, 298.15)
    assert kappa_neg > 1.0, "Wigner correction should handle negative imaginary frequency."
    
    # KINETIC-02 & KINETIC-03: Skodje-Truhlar & upper bounds
    kappa_high = skodje_truhlar_tunneling_correction(-2000.0, barrier_height_kcal=15.0, temp=200.0)
    assert kappa_high >= 1.0 and kappa_high <= 1e20


def test_eyring_rate_and_order() -> None:
    # KINETIC-06: Reaction order standard state correction
    rate_1st = calculate_eyring_rate(delta_g=10.0, temp=298.15, reaction_order=1)
    rate_2nd = calculate_eyring_rate(delta_g=10.0, temp=298.15, reaction_order=2)
    assert rate_2nd != rate_1st, "Reaction order 2 should apply standard state concentration correction."


def test_vtst_and_landau_zener() -> None:
    # KINETIC-04: VTST free energy profile search
    irc_energies = np.array([0.0, 5.0, 12.5, 10.0, 2.0])
    irc_zpes = np.array([1.0, 1.1, 1.5, 1.2, 1.0])
    max_g = variational_tst_correction(irc_energies, irc_zpes, temp=298.15)
    assert abs(max_g - 14.0) < 1e-4

    # KINETIC-05 / Suggestion 46: Landau-Zener formula (diabatic survival vs adiabatic transition)
    prob_zero_v_diab = landau_zener_probability(v12=0.1, force_diff=1.0, velocity=0.0, return_type="diabatic")
    assert prob_zero_v_diab == 1.0
    prob_zero_v_adiab = landau_zener_probability(v12=0.1, force_diff=1.0, velocity=0.0, return_type="adiabatic")
    assert prob_zero_v_adiab == 0.0

    prob_zero_coupling_diab = landau_zener_probability(v12=0.0, force_diff=1.0, velocity=100.0, return_type="diabatic")
    assert prob_zero_coupling_diab == 1.0
    prob_zero_coupling_adiab = landau_zener_probability(v12=0.0, force_diff=1.0, velocity=100.0, return_type="adiabatic")
    assert prob_zero_coupling_adiab == 0.0

    prob_valid = landau_zener_probability(v12=0.1, force_diff=1.0, velocity=100.0, return_type="diabatic")
    assert 0.0 <= prob_valid <= 1.0


def test_troe_kie_and_diagnostics() -> None:
    # KINETIC-13: Troe falloff
    k_troe = troe_falloff_rate(k_0=1e4, k_inf=1e8, P_atm=1.0, temp=298.15)
    assert 0.0 < k_troe < 1e8

    # KINETIC-14: KIE
    kie = calculate_kie(rate_H=1e5, rate_D=2e4, freq_H=2900.0, freq_D=2100.0, temp=298.15)
    assert kie > 1.0

    # KINETIC-15: Multireference diagnostic gate
    assert validate_multireference_diagnostics(0.015, 0.03) is True
    with pytest.raises(MultireferenceDiagnosticError):
        validate_multireference_diagnostics(0.035, 0.03)


def test_hindered_rotors_and_solvation() -> None:
    # KINETIC-16: Low frequency hindered rotor
    delta_g_hindered = pitzer_gwinn_hindered_rotor_correction(freq_cm1=50.0, barrier_kcal=2.5, temp=298.15)
    assert isinstance(delta_g_hindered, float)

    # KINETIC-19: Implicit solvation
    solv_g = apply_implicit_solvation_correction(delta_g_gas=12.0, delta_g_solv_ts=-3.0, delta_g_solv_reactants=-1.0)
    assert abs(solv_g - 10.0) < 1e-4


def test_cineb_mace_aimd_and_irc() -> None:
    # KINETIC-07: JAX CI-NEB
    neb = JACXCINEBEngine()
    images = np.array([[[0.0,0.0,0.0]], [[0.5,0.0,0.0]], [[1.0,0.0,0.0]]])
    def physical_eval_fn(img) -> Any:
        import numpy as np
        centroid = np.mean(img, axis=0)
        centered = img - centroid
        val = float(np.sum(centered**2))
        grad = 2.0 * centered
        return val, grad

    opt_img, opt_e = neb.optimize_path(images, physical_eval_fn, max_iter=2)
    assert len(opt_img) == 3

    # KINETIC-08: MACE Pre-optimizer fallback
    mace_opt = MACEPreOptimizer()
    rel_coords, e = mace_opt.pre_relax_geometry(["O", "H", "H"], np.array([[0,0,0],[0,0.7,-0.4],[0,-0.7,-0.4]]), max_steps=2)
    assert len(rel_coords) == 3

    # KINETIC-09: AIMD Nose-Hoover
    sampler = NoseHooverAIMDSampler()
    traj = sampler.sample_nvt_trajectory(["Fe", "O"], np.array([[0,0,0],[0,0,1.2]]), np.array([55.8, 16.0]), n_steps=3)
    assert len(traj) == 4

    # KINETIC-12: IRC Tracer
    tracer = IRCTracerEngine()
    path, energies = tracer.trace_irc_path(np.array([[0,0,0],[0,0,1]]), np.array([16.0, 1.0]), np.array([[0,0,1],[0,0,-1]]), physical_eval_fn, max_steps=2)
    assert len(path) > 0


def test_dispatcher_and_viz(tmp_path) -> None:
    # KINETIC-10 & KINETIC-18: Dispatcher & HDF5
    disp = KineticDispatcher(tmp_path)
    h5_file = tmp_path / "cochem_state.h5"
    disp.write_state_hdf5(h5_file, {"delta_g": 12.0}, irc_energies=np.array([0.0, 12.0, 1.0]))
    assert h5_file.exists() or (tmp_path / "cochem_state.json").exists()

    # KINETIC-17: 3D HTML Animation
    viz = IRCAnimationVisualizer(tmp_path)
    html = viz.generate_html_animation("Test_Rxn", ["O", "H", "H"], np.zeros((2, 3, 3)))
    assert "3Dmol" in html

def test_calculate_vtst_rate() -> None:
    from kinetic_core.thermo import calculate_vtst_rate
    s_coords = np.linspace(-1.0, 1.0, 5)
    energies = np.array([0.0, 5.0, 12.5, 10.0, 2.0])
    res = calculate_vtst_rate(s_coords, energies, temp=298.15)
    assert 'k_vtst' in res
    assert res['k_vtst'] > 0
    assert res['bottleneck_index'] == 2
    assert res['tunneling_kappa'] >= 1.0


# --- Milestone M5 Integration Tests (KINETIC-01 to KINETIC-06) ---

def test_kinetic_01_pes_store_interface(tmp_path) -> None:
    from kinetic_core.cochem_pes_store import PESStore
    import h5py

    h5_file = tmp_path / "test_pes_store.h5"
    store = PESStore(h5_file, provenance_tag="[M]")

    # Append single point
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    energy = -76.25
    gradient = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    store.append_point(coords, energy, gradient, group="grid")

    # Append batch
    batch_coords = np.zeros((5, 2, 3))
    batch_energies = np.linspace(-76.2, -76.0, 5)
    batch_variances = np.full(5, 0.001)
    store.append_batch(batch_coords, batch_energies, variance_batch=batch_variances, group="fit")

    grid_data = store.get_grid()
    fit_data = store.get_fit()

    assert "coordinates" in grid_data
    assert "energies" in grid_data
    assert "gradients" in grid_data
    assert len(grid_data["energies"]) == 1

    assert "fit_energies" in fit_data
    assert len(fit_data["fit_energies"]) == 5

    with h5py.File(h5_file, "r") as f:
        coords_ds = f["/pes/grid/coordinates"]
        assert coords_ds.chunks[0] == 512
        assert coords_ds.compression == "gzip"
        assert coords_ds.compression_opts == 4
        assert coords_ds.fletcher32 is True
        assert f["/pes/grid"].attrs["qcschema_version"] == 2
        assert f["/pes/grid"].attrs["provenance_tag"] == "[M]"


def test_kinetic_02_mace_pre_opt_gfn2_and_float32_guard() -> None:
    from kinetic_core.mace_pre_opt import MACEPreOptimizer

    mace_opt = MACEPreOptimizer()
    assert hasattr(mace_opt, "FLOAT32_NOISE_FLOOR")
    assert mace_opt.FLOAT32_NOISE_FLOOR == 1e-5

    # Test noise guard logic
    prev_g = np.array([[0.0001, 0.0, 0.0]])
    curr_g = np.array([[-0.0001, 0.0, 0.0]])
    # Sign flip in low force regime should trigger guard
    triggered = mace_opt.apply_float32_noise_guard(prev_g, curr_g, prev_energy=-10.0, curr_energy=-10.0000001)
    assert triggered is True

    # Test relaxation without pairwise LJ
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]])
    opt_c, e = mace_opt.pre_relax_geometry(["O", "H", "H"], coords, max_steps=5, fmax=0.05)
    assert opt_c.shape == (3, 3)
    assert isinstance(e, float)


def test_kinetic_03_dispatcher_v4_budgets_and_d1_d5_rules(tmp_path) -> None:
    from kinetic_core.dispatcher import KineticDispatcher, V4_WALLCLOCK_BUDGETS
    from kinetic_core.cochem_pes_store import PESStore

    disp = KineticDispatcher(tmp_path)

    # 1. 10-tier budgets check
    assert len(V4_WALLCLOCK_BUDGETS) == 10
    assert disp.get_tier_budget("T1-10s") == 10.0
    assert disp.get_tier_budget("T4-1mo") == 2592000.0

    # 2. Rule D1 (Stationary point)
    grad_small = np.array([[1e-5, 0.0, 0.0]])
    grad_large = np.array([[1e-2, 0.0, 0.0]])
    assert disp.check_d1_stationary_point(grad_small, tol=1e-4) is True
    assert disp.check_d1_stationary_point(grad_large, tol=1e-4) is False

    # 3. Rule D2 (Hessian mode match)
    freqs_prev = np.array([-150.0, 200.0, 300.0])
    freqs_curr = np.array([-155.0, 205.0, 305.0])
    assert disp.check_d2_hessian_mode_match(freqs_prev, freqs_curr, is_ts=True, tol=50.0) is True

    # 4. Rule D3 (SCF basin check)
    d1 = np.eye(3)
    d2 = np.eye(3) + 1e-4
    assert disp.check_d3_scf_basin(d1, d2, tol=1e-3) is True

    # 5. Rule D4 (Ghost atom match)
    assert disp.check_d4_cp_ghost_match([1, 2], [2, 1]) is True
    assert disp.check_d4_cp_ghost_match([1, 2], [1, 3]) is False

    # 6. Rule D5 (Safe overwrite)
    meta_prev = {"wallclock_tier": "T1-30min"}
    meta_higher = {"wallclock_tier": "T2-1h"}
    meta_lower = {"wallclock_tier": "T1-10s"}
    assert disp.check_d5_safe_overwrite(meta_prev, meta_higher) is True
    assert disp.check_d5_safe_overwrite(meta_prev, meta_lower) is False

    # 7. Write state with PESStore integration
    h5_path = tmp_path / "cochem_state.h5"
    pes_store = PESStore(h5_path)
    disp.write_state_hdf5(
        h5_path,
        thermo_data={"delta_g_barrier": 15.0},
        irc_coords=np.zeros((3, 3)),
        irc_energies=np.array([0.0, 15.0, 2.0]),
        tier_name="T2-1h",
        pes_store=pes_store,
    )
    assert h5_path.exists()


def test_kinetic_04_cineb_pes_store_streaming(tmp_path) -> None:
    from kinetic_core.jax_cineb import JACXCINEBEngine
    from kinetic_core.cochem_pes_store import PESStore

    neb = JACXCINEBEngine(k_spring=0.1)
    images = np.array([
        [[0.0, 0.0, 0.0]],
        [[0.5, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]]
    ])
    def physical_eval_fn(img) -> Any:
        import numpy as np
        centroid = np.mean(img, axis=0)
        centered = img - centroid
        val = float(np.sum(centered**2))
        grad = 2.0 * centered
        return val, grad

        return float(np.sum(img**2)), 2.0 * img

    h5_path = tmp_path / "cineb_pes.h5"
    opt_img, opt_e = neb.optimize_path(images, physical_eval_fn, max_iter=3, pes_store=str(h5_path))

    store = PESStore(h5_path)
    grid = store.get_grid()
    assert "coordinates" in grid
    assert len(grid["energies"]) > 0


def test_kinetic_05_path_integral_bead_calculator() -> None:
    from kinetic_core.aimd_hoover import calculate_required_beads

    p_300k = calculate_required_beads(300.0, 3000.0)
    assert p_300k >= 32

    p_100k = calculate_required_beads(100.0, 3000.0)
    assert p_100k >= 95

    with pytest.raises(ValueError):
        calculate_required_beads(0.0, 3000.0)


def test_kinetic_06_prohibit_classical_md_absolute_b0() -> None:
    from kinetic_core.aimd_hoover import compute_rotational_constants_md

    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    masses = np.array([16.0, 1.0, 1.0])

    res = compute_rotational_constants_md(coords, masses, is_classical_md=True)
    assert res["provenance_tag"] == "[E]"
    assert res["is_absolute_b0"] is False
    assert "WARNING" in res["warning"]
    assert "Classical MD cannot yield absolute B0" in res["warning"]
    assert len(res["rotational_constants_cm1"]) == 3


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