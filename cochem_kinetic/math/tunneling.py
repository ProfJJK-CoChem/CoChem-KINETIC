"""Exact 1D Eckart and Wigner quantum tunneling corrections with provenance gates."""

from __future__ import annotations

import logging
import math
import warnings

from scipy.integrate import quad

logger = logging.getLogger(__name__)

# Exact CODATA 2018 physical constants
PLANCK_H: float = 6.62607015e-34  # J * s
SPEED_OF_LIGHT_CM_S: float = 2.99792458e10  # cm / s
KB_J_K: float = 1.380649e-23  # J / K
AVOGADRO: float = 6.02214076e23  # mol^-1
R_KCAL_MOL_K: float = (KB_J_K * AVOGADRO) / 4184.0  # kcal / (mol * K)
HC_KCAL_PER_CM1: float = (
    PLANCK_H * SPEED_OF_LIGHT_CM_S * AVOGADRO
) / 4184.0  # kcal / (mol * cm^-1)
H_C_OVER_KB_CM_K: float = (PLANCK_H * SPEED_OF_LIGHT_CM_S) / KB_J_K  # cm * K


class ProvenanceLogger:
    @staticmethod
    def log_fallback(reason: str, method: str) -> str:
        message = f"[E] Fallback triggered: {reason}. Using {method}."
        logger.warning(message)
        return message


def _log_eckart_transmission_probability(
    energy_kcal: float,
    v_f: float,
    v_r: float,
    alpha1: float,
    alpha2: float,
) -> float:
    """
    Evaluate ``ln(P(E))`` for the exact 1D unsymmetrical Eckart (1930) /
    Johnston & Heicklen (J. Phys. Chem. 1962, 66, 532) potential barrier.

    With dimensionless parameters:
        alpha_1 = 2 * pi * V_f / (h * c * |nu_imag|)
        alpha_2 = 2 * pi * V_r / (h * c * |nu_imag|)
        eta = E / V_f
        2*pi*a = 2 * sqrt(alpha_1 * eta) / (alpha_1^(-1/2) + alpha_2^(-1/2))
        2*pi*b = 2 * sqrt(alpha_1 * eta - alpha_1 + alpha_2)
                 / (alpha_1^(-1/2) + alpha_2^(-1/2))
        (2*pi*d)^2 = 4 * alpha_1 * alpha_2 - pi^2

    The exact transmission probability is:
        P(E) = (cosh(2*pi*(a+b)) - cosh(2*pi*(a-b)))
               / (cosh(2*pi*(a+b)) + D)
    where D = cosh(2*pi*d) when 4*alpha_1*alpha_2 >= pi^2 and
    D = cos(pi * sqrt(1 - 4*alpha_1*alpha_2 / pi^2)) otherwise.

    Using the exact identity
        cosh(x + y) - cosh(x - y) = 2 * sinh(x) * sinh(y)
                                  = 0.5 * exp(x + y) * (1 - exp(-2x)) * (1 - exp(-2y))
    evaluated in log-space with ``expm1`` guarantees ``0 <= P(E) <= 1`` across
    all energy and barrier regimes without overflow or catastrophic cancellation.
    """
    e_min = max(0.0, v_f - v_r)
    if energy_kcal <= e_min:
        return -math.inf

    inv_sum = (1.0 / math.sqrt(alpha1)) + (1.0 / math.sqrt(alpha2))
    eta = energy_kcal / v_f
    arg_b = alpha1 * eta - alpha1 + alpha2
    if arg_b <= 0.0:
        return -math.inf

    x = (2.0 * math.sqrt(alpha1 * eta)) / inv_sum  # 2 * pi * a
    y = (2.0 * math.sqrt(arg_b)) / inv_sum  # 2 * pi * b
    s = x + y  # 2 * pi * (a + b)

    t1 = -math.expm1(-2.0 * x)
    t2 = -math.expm1(-2.0 * y)
    if t1 <= 0.0 or t2 <= 0.0:
        return -math.inf

    c2 = 4.0 * alpha1 * alpha2 - (math.pi**2)
    if c2 >= 0.0:
        z = math.sqrt(c2)  # 2 * pi * d
        m = max(s, z)
        log_num = s + math.log(t1) + math.log(t2)
        den_scaled = math.exp(s - m) * (1.0 + math.exp(-2.0 * s)) + math.exp(z - m) * (
            1.0 + math.exp(-2.0 * z)
        )
        log_den = m + math.log(den_scaled)
        return min(0.0, log_num - log_den)

    z = math.sqrt(-c2)  # pi * sqrt(1 - 4*alpha_1*alpha_2 / pi^2)
    den_scaled = (1.0 + math.exp(-2.0 * s)) + 2.0 * math.cos(z) * math.exp(-s)
    if den_scaled <= 0.0:
        return -math.inf
    return min(0.0, math.log(t1) + math.log(t2) - math.log(den_scaled))


def eckart_transmission_probability(
    energy_kcal: float,
    imag_freq: float,
    V_f: float,  # noqa: N803
    V_r: float,  # noqa: N803
) -> float:
    """
    Compute the exact 1D unsymmetrical Eckart barrier transmission probability
    ``P(E)`` in ``[0.0, 1.0]`` for energy ``E`` (kcal/mol above reactant zero).
    """
    if V_f < 0.0 or V_r < 0.0:
        raise ValueError("Barrier heights V_f and V_r must be non-negative.")
    freq_abs = abs(imag_freq)
    if freq_abs < 1e-6 or V_f == 0.0 or V_r == 0.0:
        return 1.0 if energy_kcal >= max(0.0, V_f, V_f - V_r) else 0.0

    hv_kcal = HC_KCAL_PER_CM1 * freq_abs
    alpha1 = (2.0 * math.pi * V_f) / hv_kcal
    alpha2 = (2.0 * math.pi * V_r) / hv_kcal
    log_p = _log_eckart_transmission_probability(energy_kcal, V_f, V_r, alpha1, alpha2)
    if not math.isfinite(log_p):
        return 0.0
    if log_p < -745.0:
        return 0.0
    return float(min(1.0, max(0.0, math.exp(log_p))))


def calculate_eckart_tunneling(
    imag_freq: float,
    temp: float,
    V_f: float,  # noqa: N803
    V_r: float,  # noqa: N803
) -> dict[str, float | str | None]:
    """
    Compute the thermally averaged 1D unsymmetrical Eckart (1930) /
    Johnston & Heicklen (1962) quantum tunneling transmission factor ``kappa(T)``.

    Returns a dict with ``'kappa'``, ``'provenance_tag'``, and ``'warning'``
    (``None`` on nominal ``[M]`` evaluation, or an explanatory warning string
    on ``[E]`` fallback).
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
    if freq_abs < 1e-6 or V_f == 0.0 or V_r == 0.0:
        return {"kappa": 1.0, "provenance_tag": "[M]", "warning": None}

    hv_kcal = HC_KCAL_PER_CM1 * freq_abs
    alpha1 = (2.0 * math.pi * V_f) / hv_kcal
    alpha2 = (2.0 * math.pi * V_r) / hv_kcal
    e_min = max(0.0, V_f - V_r)

    try:
        if not math.isfinite(temp) or temp < 0.0:
            raise ValueError(f"Temperature must be finite and > 0 K, got {temp!r}")
        k_t = R_KCAL_MOL_K * temp
        # Dimensionless energy shifted to the barrier top: eps = (E - V_f) / (k_B * T).
        # kappa(T) = integral_{eps_min}^{inf} P(V_f + eps * k_B * T) * exp(-eps) d(eps).
        # Division by k_t raises ZeroDivisionError for temp == 0.0.
        eps_min = (e_min - V_f) / k_t

        def _integrand(eps: float) -> float:
            energy_kcal = V_f + eps * k_t
            log_p = _log_eckart_transmission_probability(
                energy_kcal, V_f, V_r, alpha1, alpha2
            )
            if not math.isfinite(log_p):
                return 0.0
            log_val = log_p - eps
            if log_val < -745.0:
                return 0.0
            return math.exp(log_val)

        # Locate the sub-barrier integrand peak so adaptive quadrature never misses
        # a narrow deep-tunneling maximum at large V_f / (k_B * T).
        sub_pts: list[float] = [eps_min, 0.0]
        if eps_min < -10.0:
            n_grid = 256
            best_eps = 0.0
            best_log_val = -math.inf
            for idx in range(1, n_grid):
                eps_g = eps_min + (0.0 - eps_min) * (idx / n_grid)
                lp = _log_eckart_transmission_probability(
                    V_f + eps_g * k_t, V_f, V_r, alpha1, alpha2
                )
                lv = lp - eps_g
                if lv > best_log_val:
                    best_log_val = lv
                    best_eps = eps_g
            sub_pts = sorted(
                {
                    eps_min,
                    max(eps_min, best_eps - 10.0),
                    max(eps_min, best_eps - 2.0),
                    best_eps,
                    min(0.0, best_eps + 2.0),
                    min(0.0, best_eps + 10.0),
                    0.0,
                }
            )

        kappa = 0.0
        for lo, hi in zip(sub_pts[:-1], sub_pts[1:]):
            if hi > lo:
                seg_val, _ = quad(
                    _integrand, lo, hi, limit=200, epsabs=0.0, epsrel=1e-10
                )
                kappa += seg_val

        for lo, hi in ((0.0, 10.0), (10.0, 60.0), (60.0, math.inf)):
            seg_val, _ = quad(_integrand, lo, hi, limit=200, epsabs=0.0, epsrel=1e-10)
            kappa += seg_val

        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError(f"Non-finite or non-positive Eckart factor: {kappa}")
    except Exception as e:
        warning_msg = ProvenanceLogger.log_fallback(
            f"Quad integration failed: {e}",
            "kappa=1.0",
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        return {"kappa": 1.0, "provenance_tag": "[E]", "warning": warning_msg}

    return {
        "kappa": float(kappa),
        "provenance_tag": "[M]",
        "warning": None,
    }


def calculate_wigner_correction(imag_freq: float, temp: float) -> float:
    """
    Compute Wigner tunneling correction with a safety intercept for cryogenic
    temperatures. Emits both a UserWarning and a logger warning when
    ``temp < 50 K`` triggers the ``kappa = 1.0`` fallback.
    """
    if temp < 50.0:
        warning_msg = ProvenanceLogger.log_fallback(
            f"Cryogenic temperature (T={temp:.2f} K < 50 K) diverges for Wigner",
            "No Tunneling (kappa=1.0)",
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        return 1.0

    alpha = (H_C_OVER_KB_CM_K * abs(imag_freq)) / temp
    kappa = 1.0 + (alpha**2) / 24.0
    return float(min(kappa, 1e10))
