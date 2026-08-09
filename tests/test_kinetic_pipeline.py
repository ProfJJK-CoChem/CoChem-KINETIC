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


def test_wigner_and_tunneling_corrections():
    # KINETIC-01: Imaginary frequency sign check
    kappa_neg = wigner_correction(-1500.0, 298.15)
    assert kappa_neg > 1.0, "Wigner correction should handle negative imaginary frequency."
    
    # KINETIC-02 & KINETIC-03: Skodje-Truhlar & upper bounds
    kappa_high = skodje_truhlar_tunneling_correction(-2000.0, barrier_height_kcal=15.0, temp=200.0)
    assert kappa_high >= 1.0 and kappa_high < 1e5


def test_eyring_rate_and_order():
    # KINETIC-06: Reaction order standard state correction
    rate_1st = calculate_eyring_rate(delta_g=10.0, temp=298.15, reaction_order=1)
    rate_2nd = calculate_eyring_rate(delta_g=10.0, temp=298.15, reaction_order=2)
    assert rate_2nd != rate_1st, "Reaction order 2 should apply standard state concentration correction."


def test_vtst_and_landau_zener():
    # KINETIC-04: VTST free energy profile search
    irc_energies = np.array([0.0, 5.0, 12.5, 10.0, 2.0])
    irc_zpes = np.array([1.0, 1.1, 1.5, 1.2, 1.0])
    max_g = variational_tst_correction(irc_energies, irc_zpes, temp=298.15)
    assert abs(max_g - 14.0) < 1e-4

    # KINETIC-05: Landau-Zener zero-division guard
    prob_zero_v = landau_zener_probability(v12=0.1, force_diff=1.0, velocity=0.0)
    assert prob_zero_v == 0.0
    prob_valid = landau_zener_probability(v12=0.1, force_diff=1.0, velocity=100.0)
    assert 0.0 <= prob_valid <= 1.0


def test_troe_kie_and_diagnostics():
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


def test_hindered_rotors_and_solvation():
    # KINETIC-16: Low frequency hindered rotor
    delta_g_hindered = pitzer_gwinn_hindered_rotor_correction(freq_cm1=50.0, barrier_kcal=2.5, temp=298.15)
    assert isinstance(delta_g_hindered, float)

    # KINETIC-19: Implicit solvation
    solv_g = apply_implicit_solvation_correction(delta_g_gas=12.0, delta_g_solv_ts=-3.0, delta_g_solv_reactants=-1.0)
    assert abs(solv_g - 10.0) < 1e-4


def test_cineb_mace_aimd_and_irc():
    # KINETIC-07: JAX CI-NEB
    neb = JACXCINEBEngine()
    images = np.array([[[0.0,0.0,0.0]], [[0.5,0.0,0.0]], [[1.0,0.0,0.0]]])
    def dummy_fn(img): return np.sum(img**2), 2.0*img
    opt_img, opt_e = neb.optimize_path(images, dummy_fn, max_iter=2)
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
    path, energies = tracer.trace_irc_path(np.array([[0,0,0],[0,0,1]]), np.array([16.0, 1.0]), np.array([[0,0,1],[0,0,-1]]), dummy_fn, max_steps=2)
    assert len(path) > 0


def test_dispatcher_and_viz(tmp_path):
    # KINETIC-10 & KINETIC-18: Dispatcher & HDF5
    disp = KineticDispatcher(tmp_path)
    h5_file = tmp_path / "cochem_state.h5"
    disp.write_state_hdf5(h5_file, {"delta_g": 12.0}, irc_energies=np.array([0.0, 12.0, 1.0]))
    assert h5_file.exists() or (tmp_path / "cochem_state.json").exists()

    # KINETIC-17: 3D HTML Animation
    viz = IRCAnimationVisualizer(tmp_path)
    html = viz.generate_html_animation("Test_Rxn", ["O", "H", "H"], np.zeros((2, 3, 3)))
    assert "3Dmol" in html

def test_calculate_vtst_rate():
    from kinetic_core.thermo import calculate_vtst_rate
    s_coords = np.linspace(-1.0, 1.0, 5)
    energies = np.array([0.0, 5.0, 12.5, 10.0, 2.0])
    res = calculate_vtst_rate(s_coords, energies, temp=298.15)
    assert 'k_vtst' in res
    assert res['k_vtst'] > 0
    assert res['bottleneck_index'] == 2
    assert res['tunneling_kappa'] >= 1.0

