import warnings
import math

class NonAdiabaticOverrideWarning(Warning):
    raise NotImplementedError("Implementation pending")
class NonAdiabaticReaction:
    def __init__(self, reaction_type: str, kappa_NA: float):
        self.reaction_type = reaction_type
        self.kappa_NA = kappa_NA
        
        # Determine the spin-orbit coupling matrix element (H_12) based on
        # typical light atom (C, H, O) limits. Typical SOC values ~ 1-50 cm^-1.
        # We model this rigorously via the Landau-Zener probability:
        # P_LZ = 1 - exp(-2 * pi * H_12^2 / (hbar * v * |delta F|))
        
        if "Singlet -> Triplet" in self.reaction_type or "Triplet -> Singlet" in self.reaction_type:
            # Check for unphysical transmission for spin-forbidden crossing
            if self.kappa_NA >= 1e-3:
                # Use a physically justifiable bound based on LZ theory
                # with representative parameters for light atoms.
                
                hbar = 1.0545718e-34 # J s
                h_12_cm1 = 5.0 # realistic H_12 for typical light atoms is lower, ~ 5 cm^-1
                h_12_joules = h_12_cm1 * 1.986e-23
                
                # Representative crossing velocity ~ 10^3 m/s
                v_m_s = 1000.0
                
                # Representative force difference ~ 1 eV/Angstrom
                delta_f_ev_ang = 1.0
                delta_f_n = delta_f_ev_ang * 1.60218e-19 * 1e10
                
                # Calculate LZ probability
                exponent = (2.0 * math.pi * (h_12_joules**2)) / (hbar * v_m_s * delta_f_n)
                max_kappa_lz = 1.0 - math.exp(-exponent)
                
                warnings.warn(f"Spin-forbidden transmission coefficient {self.kappa_NA} is unphysically high. Overriding based on Landau-Zener limit to {max_kappa_lz}.", NonAdiabaticOverrideWarning)
                self.kappa_NA = max_kappa_lz
