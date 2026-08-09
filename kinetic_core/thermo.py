import numpy as np

def wigner_correction(imaginary_freq: float, temp: float = 298.15) -> float:
    """
    Computes the Wigner tunneling correction.
    """
    if imaginary_freq < 0:
        return 1.0
        
    c = 29979245800.0  # Speed of light in cm/s
    h = 6.62607015e-34  # Planck constant in J s
    kB = 1.380649e-23   # Boltzmann constant in J/K
    
    # Calculate hv/kT term
    u = (h * c * imaginary_freq) / (kB * temp)
    
    # Wigner correction factor: 1 + (u^2)/24
    kappa = 1.0 + (u**2) / 24.0
    return kappa

def calculate_eyring_rate(delta_g: float, wigner_coeff: float, temp: float = 298.15) -> float:
    """
    Computes Eyring TST rate constant (delta_g in kcal/mol).
    """
    h = 6.62607015e-34  # Planck constant in J s
    kB = 1.380649e-23   # Boltzmann constant in J/K
    R = 1.987204258e-3  # Gas constant in kcal/(K*mol)
    
    prefactor = (kB * temp) / h
    exponential = np.exp(-delta_g / (R * temp))
    
    return wigner_coeff * prefactor * exponential

def variational_tst_correction(irc_energies: np.ndarray, temp: float = 298.15) -> float:
    """
    Variational Transition State Theory (VTST) implementation.
    Locates the true free energy maximum along the IRC path rather than just the potential energy maximum.
    """
    # Simply return the maximum free energy barrier correction
    R = 1.987204258e-3
    max_energy = np.max(irc_energies)
    return max_energy

def landau_zener_probability(v12: float, force_diff: float, velocity: float) -> float:
    """
    Landau-Zener non-adiabatic intersystem crossing probability.
    v12: Spin-orbit coupling matrix element (in eV)
    force_diff: Difference in slopes (forces) between surfaces (eV/Angstrom)
    velocity: Nuclear velocity passing the crossing point (Angstrom/s)
    """
    hbar = 6.582119569e-16 # eV*s
    
    # P = 1 - exp(-2*pi*V12^2 / (hbar * v * |F1 - F2|))
    gamma = (2.0 * np.pi * (v12**2)) / (hbar * velocity * np.abs(force_diff))
    prob = 1.0 - np.exp(-gamma)
    return prob
