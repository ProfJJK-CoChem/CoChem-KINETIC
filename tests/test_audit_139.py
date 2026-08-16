import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kinetic_core.thermo import calculate_eyring_rate

def test_barrierless_recombination():
    print("Running Document 139 Audit: Barrierless Radical Recombination")
    
    # Thermodynamic data for Methyl radical recombination where apparent Ea evaluates to -2.0 kcal/mol.
    # In thermo.py, calculate_eyring_rate uses delta_g (which relates to Ea).
    # We will pass delta_g = -2.0 kcal/mol, reaction_order = 2.
    
    ea_apparent = -2.0
    temp = 298.15
    
    rate = calculate_eyring_rate(delta_g=ea_apparent, temp=temp, reaction_order=2)
    
    print(f"Input Apparent Ea: {ea_apparent} kcal/mol")
    print(f"Calculated Rate Constant: {rate} L/(mol*s) / cm^3 mol^-1 s^-1")
    
    # Assert the system dynamically overrides TST and outputs the correct PST plateau value
    # Expected PST rate is approx 3e13 cm^3 mol^-1 s^-1
    expected_rate = 3e13
    
    assert abs(rate - expected_rate) <= 1e12, "FAIL: KINETIC did not apply the Phase Space Theory centrifugal bottleneck approximation."
    print("PASS: System dynamically applied PST.")

if __name__ == "__main__":
    test_barrierless_recombination()
