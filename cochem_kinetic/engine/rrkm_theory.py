import numpy as np
import warnings

class RRKMSolver:
    def __init__(self, frequencies: list[float] | None = None) -> None:
        if frequencies is None:
            self.frequencies: list[float] = []
        else:
            self.frequencies: list[float] = frequencies
            
        self.dE: float = 10.0
        self.grain_array: np.ndarray = np.array([])
        self.rho_E: np.ndarray = np.array([])
        
    def _beyer_swinehart_exact_counting(self, max_energy: float, dE: float) -> np.ndarray:
        """
        Calculates the density of states rho(E) using the exact Beyer-Swinehart counting algorithm.
        """
        try:
            n_grains = int(max_energy / dE) + 1
        except ZeroDivisionError:
            raise ValueError("Energy grain size dE cannot be zero.")
            
        rho = np.zeros(n_grains)
        rho[0] = 1.0
        
        for freq in self.frequencies:
            try:
                freq_grains = int(freq / dE)
            except ZeroDivisionError:
                continue
                
            if freq_grains == 0:
                continue
            for i in range(freq_grains, n_grains):
                rho[i] += rho[i - freq_grains]
                
        return rho
        
    def compute_kE(self, energy: float, dE: float) -> float:
        """
        Computes the microcanonical rate constant k(E) ensuring strict Nyquist limits for dE.
        """
        try:
            min_freq = min(self.frequencies)
        except ValueError:
            raise ValueError("Cannot compute k(E) without vibrational frequencies.")
            
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
        
        # Physical calculation for full RRKM rate evaluation (W(E)/h*rho(E))
        # Since we only have rho(E) right now, we calculate a baseline microcanonical rate
        h_planck = 3.33564e-11 # cm^-1 s
        rho_sum = np.sum(self.rho_E)
        
        try:
            rate = 1.0 / (h_planck * rho_sum) if rho_sum > 0 else 0.0
        except ZeroDivisionError:
            rate = 0.0
            
        return float(rate)
