import logging
import math
import warnings

import numpy as np
from scipy.integrate import quad

logger = logging.getLogger(__name__)


class ProvenanceLogger:
    @staticmethod
    def log_fallback(reason: str, method: str) -> str:
        message = f"[E] Fallback triggered: {reason}. Using {method}."
        logger.warning(message)
        return message


def calculate_eckart_tunneling(
    imag_freq: float,
    temp: float,
    V_f: float,
    V_r: float,
) -> dict[str, float | str | None]:
    """
    Computes Eckart tunneling correction for an asymmetric barrier.
    Returns a dict with 'kappa', 'provenance_tag', and 'warning' (None on
    nominal [M] evaluation, or an explanatory warning string on [E] fallback).
    """
    # 3. Negative Barrier Height VTST Gate
    if V_f < 0.0 or V_r < 0.0:
        warning_msg = ProvenanceLogger.log_fallback(
            "Negative barrier height detected",
            "Classical VTST Transition (kappa=1.0)",
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        return {"kappa": 1.0, "provenance_tag": "[E]", "warning": warning_msg}

    # 5. Multi-reference Eckart Binding Gate
    if abs(imag_freq) > 3000.0 and V_f < 1.0:
        warning_msg = ProvenanceLogger.log_fallback(
            "Multi-reference failure: extremely high freq for low barrier",
            "Wigner surrogate",
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        return {
            "kappa": calculate_wigner_correction(imag_freq, temp),
            "provenance_tag": "[E]",
            "warning": warning_msg,
        }

    freq_abs = abs(imag_freq)
    if freq_abs < 1e-6:
        return {"kappa": 1.0, "provenance_tag": "[M]", "warning": None}

    R = 1.987204258e-3  # kcal/(K*mol)
    kT = R * temp

    A_diff = V_f - V_r
    B = (math.sqrt(V_f) + math.sqrt(V_r)) ** 2

    hc_kcal = 0.002859
    C = hc_kcal * freq_abs

    alpha = 2 * math.pi / C

    def P_E(E: float) -> float:
        if E < min(0.0, A_diff):
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

    try:
        integral, _ = quad(lambda E: P_E(E) * math.exp(-E / kT), 0, np.inf, limit=200)
        if not math.isfinite(integral):
            raise ValueError(f"Non-finite quadrature integral: {integral}")
        kappa = integral * math.exp(V_f / kT) / kT
        if not math.isfinite(kappa):
            raise ValueError(f"Non-finite Eckart tunneling factor: {kappa}")
    except Exception as e:
        warning_msg = ProvenanceLogger.log_fallback(
            f"Quad integration failed: {e}",
            "kappa=1.0",
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        return {"kappa": 1.0, "provenance_tag": "[E]", "warning": warning_msg}

    return {
        "kappa": float(min(max(1.0, kappa), 1e15)),
        "provenance_tag": "[M]",
        "warning": None,
    }


def calculate_wigner_correction(imag_freq: float, temp: float) -> float:
    """
    Computes Wigner tunneling correction with a safety intercept for cryogenic temperatures.
    Emits both a UserWarning and a logger warning when T < 50 K triggers kappa=1.0 fallback.
    """
    if temp < 50.0:
        warning_msg = ProvenanceLogger.log_fallback(
            f"Cryogenic temperature (T={temp:.2f} K < 50 K) diverges for Wigner",
            "No Tunneling (kappa=1.0)",
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        return 1.0

    hck = 1.43877  # cm*K
    alpha = (hck * imag_freq) / temp

    kappa = 1.0 + (alpha**2) / 24.0
    return float(min(kappa, 1e10))
