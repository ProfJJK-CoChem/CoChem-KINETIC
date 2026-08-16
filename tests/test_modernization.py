import pytest
import math
from kinetic_core.thermo import calculate_wigner_correction
from cochem_kinetic.math.tunneling import calculate_eckart_tunneling
from cochem_kinetic.math.thermodynamics import grimes_qrrho_entropy

def test_grimes_qrrho_interpolation():
    # 2. Grimme's qRRHO Interpolation
    freqs = [50.0, 150.0, 500.0, 3000.0]
    
    # At 298.15 K
    S_298 = grimes_qrrho_entropy(freqs, temp=298.15, cutoff_freq=100.0)
    assert S_298 > 0
    assert isinstance(S_298, float)
    
    # Make sure low frequencies contribute more due to free rotor approx
    # Free rotor has higher entropy at very low freq than pure harmonic oscillator
    freqs_low = [10.0, 150.0, 500.0, 3000.0]
    S_298_low = grimes_qrrho_entropy(freqs_low, temp=298.15, cutoff_freq=100.0)
    assert S_298_low > S_298

@pytest.mark.parametrize("V_f, V_r, expected_kappa, expected_tag", [
    # 3. Negative Barrier Height VTST Gate
    (-1.0, 10.0, 1.0, "[E]"),
    (10.0, -2.0, 1.0, "[E]"),
    (-5.0, -5.0, 1.0, "[E]"),
])
def test_negative_barrier_height_vtst_gate(caplog, V_f, V_r, expected_kappa, expected_tag):
    result = calculate_eckart_tunneling(imag_freq=500.0, temp=298.15, V_f=V_f, V_r=V_r)
    assert result["kappa"] == expected_kappa
    assert result["provenance_tag"] == expected_tag
    # 6. Enforce strict [E] provenance logging
    assert "[E] Fallback triggered: Negative barrier height detected. Using Classical VTST Transition (kappa=1.0)." in caplog.text

def test_multireference_eckart_binding_gate(caplog):
    # 5. Multi-reference Eckart Binding Gate
    # imag_freq > 3000 and V_f < 1.0
    result = calculate_eckart_tunneling(imag_freq=3500.0, temp=298.15, V_f=0.5, V_r=10.0)
    assert result["provenance_tag"] == "[E]"
    assert "[E] Fallback triggered: Multi-reference failure: extremely high freq for low barrier. Using Wigner surrogate." in caplog.text
    assert result["kappa"] > 1.0 # Should be wigner surrogate

@pytest.mark.parametrize("imag_freq, temp, V_f, V_r", [
    # 4. Eckart Reaction Asymmetry
    (500.0, 298.15, 10.0, 10.0),    # Symmetric
    (500.0, 298.15, 10.0, 20.0),    # Asymmetric (exothermic)
    (500.0, 298.15, 20.0, 10.0),    # Asymmetric (endothermic)
    (100.0, 100.0,  5.0,  15.0),    # Low T, asymmetric
])
def test_eckart_reaction_asymmetry(imag_freq, temp, V_f, V_r):
    result = calculate_eckart_tunneling(imag_freq=imag_freq, temp=temp, V_f=V_f, V_r=V_r)
    kappa = result["kappa"]
    assert kappa >= 1.0
    assert result["provenance_tag"] == "[M]"

def test_eckart_cryogenic_wigner_fallback(caplog):
    # Test wigner fallback to eckart at cryogenic temperatures
    kappa = calculate_wigner_correction(imag_freq=500.0, temp=30.0)
    assert kappa >= 1.0
    # Wait, wigner correction fallback to eckart does not log to caplog unless I use ProvenanceLogger
    # Let's check caplog
    assert "Cryogenic temperature" in caplog.text
