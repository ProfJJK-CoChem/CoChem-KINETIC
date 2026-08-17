import warnings
import math

class NonAdiabaticOverrideWarning(Warning):
    pass
class NonAdiabaticReaction:
    def __init__(self, reaction_type: str, kappa_NA: float) -> None:
        self.reaction_type: str = reaction_type
        self.kappa_NA: float = kappa_NA
        
        # Determine the spin-orbit coupling matrix element (H_12) based on
        # typical light atom (C, H, O) limits. Typical SOC values ~ 1-50 cm^-1.
        # We model this rigorously via the Landau-Zener probability:
        # P_LZ = 1 - exp(-2 * pi * H_12^2 / (hbar * v * |delta F|))
        
        if "Singlet -> Triplet" in self.reaction_type or "Triplet -> Singlet" in self.reaction_type:
            # Check for unphysical transmission for spin-forbidden crossing
            if self.kappa_NA >= 1e-3:
                # Use a physically justifiable bound based on LZ theory
                # with representative parameters for light atoms.
                
                hbar: float = 1.0545718e-34 # J s
                h_12_cm1: float = 5.0 # realistic H_12 for typical light atoms is lower, ~ 5 cm^-1
                h_12_joules: float = h_12_cm1 * 1.986e-23
                
                # Representative crossing velocity ~ 10^3 m/s
                v_m_s: float = 1000.0
                
                # Representative force difference ~ 1 eV/Angstrom
                delta_f_ev_ang: float = 1.0
                delta_f_n: float = delta_f_ev_ang * 1.60218e-19 * 1e10
                
                try:
                    # Calculate LZ probability
                    exponent: float = (2.0 * math.pi * (h_12_joules**2)) / (hbar * v_m_s * delta_f_n)
                    max_kappa_lz: float = 1.0 - math.exp(-exponent)
                except OverflowError:
                    max_kappa_lz = 1.0
                except ZeroDivisionError:
                    max_kappa_lz = 0.0
                
                warnings.warn(f"Spin-forbidden transmission coefficient {self.kappa_NA} is unphysically high. Overriding based on Landau-Zener limit to {max_kappa_lz}.", NonAdiabaticOverrideWarning)
                self.kappa_NA = max_kappa_lz
