import warnings

class CollisionCrossSectionWarning(Warning):
    pass

BATH_GAS_PARAMETERS: dict[str, tuple[float, float]] = {
    "Helium": (2.5, 10.0),
    "Argon": (3.4, 114.0),
    "Xenon": (4.0, 230.0),
    "Nitrogen": (3.7, 71.4),
}

def _get_empirical_limits(gas_name: str) -> tuple[float, float]:
    """
    Retrieves empirical Lennard-Jones parameters from the local database.
    Returns (sigma in Angstroms, eps/k_B in Kelvin).
    """
    try:
        return BATH_GAS_PARAMETERS[gas_name]
    except KeyError as e:
        raise ValueError(f"Unknown bath gas: {gas_name}") from e

def configure_bath_gas(gas_name: str, sigma: float, eps: float) -> tuple[float, float]:
    """
    Validates provided bath gas Lennard-Jones parameters against empirical bounds.
    If a massive mismatch is detected, issues a warning and overrides with true empirical parameters.
    """
    try:
        true_sigma, true_eps = _get_empirical_limits(gas_name)
    except ValueError:
        return sigma, eps
        
    # Check for physically impossible deviations (e.g., > 20% mismatch in sigma or huge eps deviation)
    # Helium sigma ~2.5, Xenon ~4.0. If someone inputs Xenon with Helium parameters, it's caught.
    sigma_mismatch = abs(sigma - true_sigma) / true_sigma
    eps_mismatch = abs(eps - true_eps) / true_eps
    
    if sigma_mismatch > 0.2 or eps_mismatch > 0.5:
        warning_msg = (
            f"Massive parameter mismatch for {gas_name}. "
            f"Provided: sigma={sigma}, eps={eps}. "
            f"Empirical limits: sigma={true_sigma}, eps={true_eps}. "
            "Overriding with true empirical parameters."
        )
        warnings.warn(warning_msg, CollisionCrossSectionWarning)
        return true_sigma, true_eps
            
    return sigma, eps
