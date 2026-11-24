"""
Zero-mock unit tests for explicit fallback warnings and provenance metadata in
SmoluchowskiDiffusionEngine and Eckart/Wigner tunneling routines.
"""

from __future__ import annotations

import logging
import math
import warnings

import pytest

from cochem_kinetic.literature.diffusion_mapping import (
    DEFAULT_GENERIC_VISCOSITY_PA_S,
    SmoluchowskiDiffusionEngine,
)
from cochem_kinetic.math.tunneling import (
    calculate_eckart_tunneling,
    calculate_wigner_correction,
)


def _analytical_smoluchowski_molar(
    radius_a_angstrom: float,
    radius_b_angstrom: float,
    temp: float,
    viscosity_pa_s: float,
) -> float:
    kb = 1.380649e-23
    n_a = 6.02214076e23
    r_a = radius_a_angstrom * 1e-10
    r_b = radius_b_angstrom * 1e-10
    d_a = (kb * temp) / (6.0 * math.pi * viscosity_pa_s * r_a)
    d_b = (kb * temp) / (6.0 * math.pi * viscosity_pa_s * r_b)
    return 4.0 * math.pi * (d_a + d_b) * (r_a + r_b) * n_a * 1e3


def test_unknown_solvent_emits_user_warning_log_and_uses_generic_viscosity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = SmoluchowskiDiffusionEngine()
    with caplog.at_level(logging.WARNING):
        with pytest.warns(UserWarning, match=r"Unknown solvent 'toluene'.*1\.0e-03 Pa\*s"):
            k_d = engine.compute_diffusion_limit(
                radius_a_angstrom=2.0,
                radius_b_angstrom=2.0,
                temp=298.15,
                solvent="toluene",
            )

    expected = _analytical_smoluchowski_molar(
        2.0, 2.0, 298.15, DEFAULT_GENERIC_VISCOSITY_PA_S
    )
    assert isinstance(k_d, float)
    assert math.isclose(k_d, expected, rel_tol=1e-12)
    assert engine.last_provenance_tag == "[E]"
    assert engine.last_fallback_warning is not None
    assert "toluene" in engine.last_fallback_warning
    assert "1.0e-03 Pa*s" in engine.last_fallback_warning
    assert engine.last_viscosity_metadata["fallback_used"] is True
    assert engine.last_viscosity_metadata["viscosity_pa_s"] == DEFAULT_GENERIC_VISCOSITY_PA_S
    assert "Unknown solvent 'toluene'" in caplog.text


def test_diffusion_provenance_helper_and_process_association_rate() -> None:
    engine = SmoluchowskiDiffusionEngine()
    with pytest.warns(UserWarning, match=r"Unknown solvent 'chloroform'"):
        prov = engine.compute_diffusion_limit_with_provenance(
            radius_a_angstrom=2.5,
            radius_b_angstrom=3.5,
            temp=298.15,
            solvent="chloroform",
        )

    expected = _analytical_smoluchowski_molar(
        2.5, 3.5, 298.15, DEFAULT_GENERIC_VISCOSITY_PA_S
    )
    assert math.isclose(prov["k_diffusion_limit"], expected, rel_tol=1e-12)
    assert prov["fallback_used"] is True
    assert prov["provenance_tag"] == "[E]"
    assert prov["viscosity_pa_s"] == DEFAULT_GENERIC_VISCOSITY_PA_S
    assert isinstance(prov["warning"], str) and "chloroform" in prov["warning"]

    state_dict: dict[str, object] = {}
    with pytest.warns(UserWarning, match=r"Unknown solvent 'hexane'"):
        capped = engine.process_association_rate(
            raw_rate=1e15,
            radius_a=2.0,
            radius_b=2.0,
            state_dict=state_dict,
            temp=298.15,
            solvent="hexane",
        )

    assert state_dict["tag"] == "DiffusionLimited"
    assert state_dict["provenance_tag"] == "[E]"
    assert isinstance(state_dict["warning"], str) and "hexane" in state_dict["warning"]
    assert math.isclose(
        capped,
        _analytical_smoluchowski_molar(2.0, 2.0, 298.15, DEFAULT_GENERIC_VISCOSITY_PA_S),
        rel_tol=1e-12,
    )


def test_known_water_solvent_emits_no_fallback_warning() -> None:
    engine = SmoluchowskiDiffusionEngine()

    # First trigger a fallback to verify subsequent valid calls reset metadata cleanly
    with pytest.warns(UserWarning, match=r"Unknown solvent 'benzene'"):
        engine.compute_diffusion_limit(2.0, 2.0, solvent="benzene")
    assert engine.last_fallback_warning is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k_water = engine.compute_diffusion_limit(
            radius_a_angstrom=2.0,
            radius_b_angstrom=3.0,
            temp=298.15,
            solvent="water",
        )
        prov_water = engine.compute_diffusion_limit_with_provenance(
            radius_a_angstrom=2.0,
            radius_b_angstrom=3.0,
            temp=298.15,
            solvent="WATER",
        )

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []
    expected = _analytical_smoluchowski_molar(2.0, 3.0, 298.15, 8.9e-4)
    assert math.isclose(k_water, expected, rel_tol=1e-12)
    assert math.isclose(prov_water["k_diffusion_limit"], expected, rel_tol=1e-12)
    assert prov_water["fallback_used"] is False
    assert prov_water["provenance_tag"] == "[M]"
    assert prov_water["warning"] is None
    assert engine.last_fallback_warning is None
    assert engine.last_provenance_tag == "[M]"


@pytest.mark.parametrize("cryo_temp", [10.0, 30.0, 49.9])
def test_cryogenic_wigner_emits_user_warning_and_returns_unity(
    caplog: pytest.LogCaptureFixture,
    cryo_temp: float,
) -> None:
    with caplog.at_level(logging.WARNING):
        with pytest.warns(UserWarning, match=r"Cryogenic temperature.*< 50 K"):
            kappa = calculate_wigner_correction(imag_freq=500.0, temp=cryo_temp)

    assert kappa == 1.0
    assert "[E] Fallback triggered: Cryogenic temperature" in caplog.text


def test_eckart_quad_failure_emits_user_warning_and_warning_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Passing temp=0.0 genuinely causes ZeroDivisionError inside the quad integrand
    # without using any mocks or monkeypatching.
    with caplog.at_level(logging.WARNING):
        with pytest.warns(UserWarning, match=r"Quad integration failed"):
            result = calculate_eckart_tunneling(
                imag_freq=500.0,
                temp=0.0,
                V_f=10.0,
                V_r=10.0,
            )

    assert result["kappa"] == 1.0
    assert result["provenance_tag"] == "[E]"
    assert isinstance(result["warning"], str)
    assert "Quad integration failed" in result["warning"]
    assert "[E] Fallback triggered: Quad integration failed" in caplog.text


def test_eckart_vtst_and_multireference_gates_emit_user_warning_and_warning_field() -> None:
    with pytest.warns(UserWarning, match=r"Negative barrier height detected"):
        neg_res = calculate_eckart_tunneling(
            imag_freq=500.0, temp=298.15, V_f=-1.0, V_r=10.0
        )
    assert neg_res["kappa"] == 1.0
    assert neg_res["provenance_tag"] == "[E]"
    assert isinstance(neg_res["warning"], str)
    assert "Negative barrier height" in neg_res["warning"]

    with pytest.warns(UserWarning, match=r"Multi-reference failure"):
        mr_res = calculate_eckart_tunneling(
            imag_freq=3500.0, temp=298.15, V_f=0.5, V_r=10.0
        )
    assert float(mr_res["kappa"]) > 1.0  # type: ignore[arg-type]
    assert mr_res["provenance_tag"] == "[E]"
    assert isinstance(mr_res["warning"], str)
    assert "Wigner surrogate" in mr_res["warning"]


def test_normal_eckart_and_wigner_regimes_emit_no_fallback_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eckart_asym = calculate_eckart_tunneling(
            imag_freq=500.0, temp=298.15, V_f=10.0, V_r=15.0
        )
        eckart_zero = calculate_eckart_tunneling(
            imag_freq=0.0, temp=298.15, V_f=10.0, V_r=10.0
        )
        wigner_298 = calculate_wigner_correction(imag_freq=500.0, temp=298.15)
        wigner_boundary = calculate_wigner_correction(imag_freq=500.0, temp=50.0)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []

    assert eckart_asym["provenance_tag"] == "[M]"
    assert eckart_asym["warning"] is None
    assert float(eckart_asym["kappa"]) >= 1.0  # type: ignore[arg-type]

    assert eckart_zero["provenance_tag"] == "[M]"
    assert eckart_zero["warning"] is None
    assert eckart_zero["kappa"] == 1.0

    assert wigner_298 > 1.0
    assert wigner_boundary > 1.0
