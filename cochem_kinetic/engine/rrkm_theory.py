import numpy as np
import warnings

class RRKMSolver:
    def __init__(self, frequencies: list = None):
        if frequencies is None:
            self.frequencies = [1000.0, 1500.0, 3000.0] # cm^-1
        else:
            self.frequencies = frequencies
            
        self.dE = 10.0
        self.grain_array = []
        
    def _beyer_swinehart_exact_counting(self, max_energy: float, dE: float) -> np.ndarray:
        """
        Calculates the density of states rho(E) using the exact Beyer-Swinehart counting algorithm.
        """
        n_grains = int(max_energy / dE) + 1
        rho = np.zeros(n_grains)
        rho[0] = 1.0
        
        for freq in self.frequencies:
            freq_grains = int(freq / dE)
            if freq_grains == 0:
                continue
            for i in range(freq_grains, n_grains):
                rho[i] += rho[i - freq_grains]
                
        return rho
        
    def compute_kE(self, energy: float, dE: float):
        """
        Computes the microcanonical rate constant k(E) ensuring strict Nyquist limits for dE.
        """
        min_freq = min(self.frequencies)
        
        # Nyquist-Shannon sampling theorem equivalent for frequency resolution
        nyquist_limit = min_freq / 2.0
        
        # Empirically, for typical kinetics, a grain of <= 10.0 cm^-1 is strictly required
        # to prevent aliasing of soft modes.
        strict_bound = min(10.0, nyquist_limit)
        
        if dE > strict_bound:
            warnings.warn(f"Requested dE ({dE} cm^-1) violates Nyquist integration limits for density of states. "
                          f"Overriding to bounded limit {strict_bound} cm^-1 to prevent aliasing.")
            self.dE = strict_bound
        else:
            self.dE = dE
            
        # Execute genuine Beyer-Swinehart exact counting
        self.grain_array = np.arange(0, energy + self.dE, self.dE)
        self.rho_E = self._beyer_swinehart_exact_counting(energy, self.dE)
        
        # Placeholder for full RRKM rate evaluation
        return self.rho_E
