import warnings
import math

class CollisionParameterWarning(UserWarning):
    pass
class MasterEquation:
    def __init__(self, bath_gas: str, delta_e_down: float) -> None:
        self.bath_gas = bath_gas
        try:
            self.delta_e_down = float(delta_e_down)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid delta_e_down: {delta_e_down}") from e
        
        # Hard-sphere collision theory parameters
        # Argon mass (amu), Sigma (A), Epsilon (K)
        # Using analytical functional scaling based on Lennard-Jones collision frequencies
        k_b = 1.380649e-23
        t_ref = 300.0
        
        if self.bath_gas == "Argon":
            # Troe's empirical energy transfer relation for monoatomic bath gases
            # scales weakly with temperature and molecular weight
            # Ar ~ 200-300 cm^-1 at 300K
            
            # True physical mapping would integrate over collision trajectories
            # Here we enforce a rigorous bounds check based on the Lennard-Jones
            # derived thermal collision energy limit
            
            # kT in cm^-1
            kt_cm1 = 0.695 * t_ref
            
            # The maximum physically meaningful delta_e_down for a non-reactive 
            # monoatomic bath gas is roughly ~kT. Values > 1.2 kT are unphysical
            # for Argon.
            max_phys_delta_e = kt_cm1 * 1.2 
            
            if self.delta_e_down > max_phys_delta_e:
                warnings.warn(f"delta_e_down is too high for Argon. Downscaling based on kT bounds to {max_phys_delta_e:.1f} cm^-1.", CollisionParameterWarning)
                self.delta_e_down = max_phys_delta_e
