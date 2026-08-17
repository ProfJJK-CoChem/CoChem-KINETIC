import math
import numpy as np
from scipy.integrate import quad
import logging

class ProvenanceLogger:
    @staticmethod
    def log_fallback(reason: str, method: str):
        logging.warning(f"[E] Fallback triggered: {reason}. Using {method}.")

def calculate_eckart_tunneling(imag_freq: float, temp: float, V_f: float = 10.0, V_r: float = 15.0) -> dict:
    """
    Computes Eckart tunneling correction for an asymmetric barrier.
    Returns a dict with 'kappa' and 'provenance_tag'.
    """
    # 3. Negative Barrier Height VTST Gate
    if V_f < 0.0 or V_r < 0.0:
        ProvenanceLogger.log_fallback("Negative barrier height detected", "Classical VTST Transition (kappa=1.0)")
        return {"kappa": 1.0, "provenance_tag": "[E]"}

    # 5. Multi-reference Eckart Binding Gate
    if abs(imag_freq) > 3000.0 and V_f < 1.0:
        ProvenanceLogger.log_fallback("Multi-reference failure: extremely high freq for low barrier", "Wigner surrogate")
        return {"kappa": calculate_wigner_correction(imag_freq, temp), "provenance_tag": "[E]"}

    freq_abs = abs(imag_freq)
    if freq_abs < 1e-6:
        return {"kappa": 1.0, "provenance_tag": "[M]"}

    R = 1.987204258e-3  # kcal/(K*mol)
    kT = R * temp
    
    A_diff = V_f - V_r
    B = (math.sqrt(V_f) + math.sqrt(V_r))**2
    
    hc_kcal = 0.002859
    C = hc_kcal * freq_abs
    
    alpha = 2 * math.pi / C
    
    def P_E(E):
        if E < min(0, A_diff):
            return 0.0
        
        a = alpha * math.sqrt(E) if E > 0 else 0.0
        b = alpha * math.sqrt(E - A_diff) if E - A_diff > 0 else 0.0
        c2 = alpha**2 * B - math.pi**2
        
        if a + b > 50.0:
            return float(max(0.0, min(1.0, 1.0 - math.exp(-2.0 * b))))

        try:
            cosh_ab = math.cosh(a + b)
            cosh_amb = math.cosh(a - b)
            
            if c2 > 0:
                c_val = math.sqrt(c2)
                denom = cosh_ab + math.cosh(c_val)
            else:
                c_val = math.sqrt(-c2)
                denom = cosh_ab + math.cos(c_val)
                
            return (cosh_ab - cosh_amb) / denom
        except OverflowError:
            return float(max(0.0, min(1.0, 1.0 - math.exp(-2.0 * b))))

    integral, _ = quad(lambda E: P_E(E) * math.exp(-E / kT), 0, np.inf, limit=200)
    kappa = integral * math.exp(V_f / kT) / kT

    return {"kappa": float(min(max(1.0, kappa), 1e15)), "provenance_tag": "[M]"}

def calculate_wigner_correction(imag_freq: float, temp: float) -> float:
    """
    Computes Wigner tunneling correction with a safety intercept for cryogenic temperatures.
    """
    if temp < 50.0:
        ProvenanceLogger.log_fallback("Cryogenic temperature (T < 50 K) diverges for Wigner", "Eckart forced")
        return calculate_eckart_tunneling(imag_freq, temp)["kappa"]
        
    hck = 1.43877 # cm*K
    alpha = (hck * imag_freq) / temp
    
    kappa = 1.0 + (alpha**2) / 24.0
    return float(min(kappa, 1e10))
