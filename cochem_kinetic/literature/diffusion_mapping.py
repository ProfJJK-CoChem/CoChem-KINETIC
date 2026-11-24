import logging
import math
import typing
import warnings

logger = logging.getLogger(__name__)

DEFAULT_GENERIC_VISCOSITY_PA_S: float = 1e-3


class SmoluchowskiDiffusionEngine:
    def __init__(self) -> None:
        self.known_viscosities: dict[str, float] = {
            "water": 8.9e-4,
        }
        self.last_fallback_warning: str | None = None
        self.last_provenance_tag: str = "[M]"
        self.last_viscosity_metadata: dict[str, typing.Any] = {}

    def _get_solvent_viscosity(
        self,
        solvent: str,
        *,
        _stacklevel: int = 2,
    ) -> float:
        """
        Retrieves known solvent viscosity in Pa*s (kg/(m*s)).

        Falls back to 1e-3 Pa*s for unknown solvents while emitting both a
        UserWarning and a logger warning, and recording structured fallback
        metadata on the engine instance.
        """
        try:
            solvent_key = solvent.lower()
            eta = self.known_viscosities[solvent_key]
        except (KeyError, AttributeError):
            fallback_eta = DEFAULT_GENERIC_VISCOSITY_PA_S
            known_list = ", ".join(sorted(self.known_viscosities))
            warning_msg = (
                f"[E] Unknown solvent {solvent!r} in SmoluchowskiDiffusionEngine; "
                f"falling back to generic viscosity {fallback_eta:.1e} Pa*s "
                f"(known solvents: {known_list})."
            )
            self.last_fallback_warning = warning_msg
            self.last_provenance_tag = "[E]"
            self.last_viscosity_metadata = {
                "solvent": solvent,
                "viscosity_pa_s": fallback_eta,
                "fallback_used": True,
                "provenance_tag": "[E]",
                "warning": warning_msg,
            }
            logger.warning(warning_msg)
            warnings.warn(warning_msg, UserWarning, stacklevel=_stacklevel)
            return fallback_eta

        self.last_fallback_warning = None
        self.last_provenance_tag = "[M]"
        self.last_viscosity_metadata = {
            "solvent": solvent_key,
            "viscosity_pa_s": eta,
            "fallback_used": False,
            "provenance_tag": "[M]",
            "warning": None,
        }
        return eta

    def _compute_diffusion_limit_value(
        self,
        radius_a_angstrom: float,
        radius_b_angstrom: float,
        temp: float = 298.15,
        solvent: str = "water",
        *,
        _stacklevel: int = 3,
    ) -> float:
        kB = 1.380649e-23
        N_A = 6.02214076e23

        eta = self._get_solvent_viscosity(solvent, _stacklevel=_stacklevel)

        r_a = radius_a_angstrom * 1e-10
        r_b = radius_b_angstrom * 1e-10

        try:
            # Stokes-Einstein
            d_a = (kB * temp) / (6 * math.pi * eta * r_a)
            d_b = (kB * temp) / (6 * math.pi * eta * r_b)
        except ZeroDivisionError:
            raise ValueError("Radius or viscosity cannot be zero.")

        d_ab = d_a + d_b
        r_ab = r_a + r_b

        # Smoluchowski rate in m^3 / (molecule s)
        k_smoluchowski = 4 * math.pi * d_ab * r_ab

        # Convert to M^-1 s^-1 (L / mol s)
        k_smol_molar = k_smoluchowski * N_A * 1e3

        return k_smol_molar

    def compute_diffusion_limit(
        self,
        radius_a_angstrom: float,
        radius_b_angstrom: float,
        temp: float = 298.15,
        solvent: str = "water",
    ) -> float:
        """
        Computes the theoretical diffusion limit using the Smoluchowski equation:
        k_D = 4 * pi * D_AB * R_AB * N_A
        where D_AB = D_A + D_B (diffusion coefficients) and R_AB = R_A + R_B (encounter radius).
        Using Stokes-Einstein for D: D = k_B T / (6 pi eta r)
        Returns k_D in M^-1 s^-1.
        """
        return self._compute_diffusion_limit_value(
            radius_a_angstrom=radius_a_angstrom,
            radius_b_angstrom=radius_b_angstrom,
            temp=temp,
            solvent=solvent,
            _stacklevel=4,
        )

    def compute_diffusion_limit_with_provenance(
        self,
        radius_a_angstrom: float,
        radius_b_angstrom: float,
        temp: float = 298.15,
        solvent: str = "water",
    ) -> dict[str, typing.Any]:
        """
        Computes the Smoluchowski diffusion limit and returns structured
        provenance metadata including any solvent viscosity fallback warning.
        """
        k_smol_molar = self._compute_diffusion_limit_value(
            radius_a_angstrom=radius_a_angstrom,
            radius_b_angstrom=radius_b_angstrom,
            temp=temp,
            solvent=solvent,
            _stacklevel=4,
        )
        return {
            "k_diffusion_limit": k_smol_molar,
            "solvent": self.last_viscosity_metadata.get("solvent", solvent),
            "viscosity_pa_s": self.last_viscosity_metadata.get(
                "viscosity_pa_s",
                DEFAULT_GENERIC_VISCOSITY_PA_S,
            ),
            "fallback_used": bool(
                self.last_viscosity_metadata.get("fallback_used", False)
            ),
            "provenance_tag": self.last_provenance_tag,
            "warning": self.last_fallback_warning,
        }

    def process_association_rate(
        self,
        raw_rate: float,
        radius_a: float,
        radius_b: float,
        state_dict: dict[str, typing.Any],
        temp: float = 298.15,
        solvent: str = "water",
    ) -> float:
        """
        Applies diffusion bounding. If the theoretical raw rate exceeds the Smoluchowski limit,
        caps it and tags the metadata. Records viscosity fallback metadata in state_dict when
        a generic viscosity fallback is triggered.
        """
        diffusion_limit = self._compute_diffusion_limit_value(
            radius_a_angstrom=radius_a,
            radius_b_angstrom=radius_b,
            temp=temp,
            solvent=solvent,
            _stacklevel=4,
        )

        if self.last_fallback_warning is not None:
            state_dict["provenance_tag"] = self.last_provenance_tag
            state_dict["warning"] = self.last_fallback_warning
            state_dict["viscosity_fallback"] = dict(self.last_viscosity_metadata)

        if raw_rate > diffusion_limit:
            state_dict["tag"] = "DiffusionLimited"
            return diffusion_limit

        return raw_rate
