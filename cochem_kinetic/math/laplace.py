import numpy as np
from scipy.optimize import nnls
import warnings

def _naive_inverse_laplace(T_array: np.ndarray, k_T_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Naive implementation of Inverse Laplace Transform using basic matrix inversion.
    Prone to severe oscillations and negative values.
    """
    # Assuming k(T) = \int k(E) exp(-E/RT) dE
    # We discretize E_array based on T_array spread
    R = 8.314 # J/mol-K
    E_array = np.linspace(0, 100000, len(T_array)) # J/mol
    
    # Kernel matrix K_ij = exp(-E_j / R T_i)
    K = np.exp(-E_array[np.newaxis, :] / (R * T_array[:, np.newaxis]))
    
    # Naive inversion: k_E = K^{-1} k_T
    # Using pseudoinverse for stability but still unregularized
    try:
        K_inv = np.linalg.pinv(K)
        k_E = K_inv @ k_T_array
    except np.linalg.LinAlgError as e:
        warnings.warn(f"Pseudoinverse computation failed: {e}")
        k_E = np.full(len(T_array), -1.0)
    
    return E_array, k_E

def _tikhonov_nnls_inverse_laplace(T_array: np.ndarray, k_T_array: np.ndarray, alpha: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Regularized implementation of Inverse Laplace Transform using Tikhonov regularization 
    and Non-Negative Least Squares (NNLS).
    """
    R = 8.314 # J/mol-K
    E_array = np.linspace(0, 100000, len(T_array))
    
    K = np.exp(-E_array[np.newaxis, :] / (R * T_array[:, np.newaxis]))
    
    # Tikhonov regularization: min ||K x - y||^2 + alpha ||L x||^2
    # We append alpha * I to K, and zeros to y
    L = np.eye(len(E_array))
    K_aug = np.vstack((K, alpha * L))
    y_aug = np.concatenate((k_T_array, np.zeros(len(E_array))))
    
    # Solve with NNLS to strictly enforce non-negativity
    try:
        k_E, _ = nnls(K_aug, y_aug)
    except RuntimeError as e:
        raise RuntimeError(f"NNLS failed to converge: {e}") from e
    
    return E_array, k_E

def inverse_laplace_transform(T_array: np.ndarray, k_T_array: np.ndarray, regularization: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the microcanonical rate constant k(E) from the thermal rate constant k(T)
    using the Inverse Laplace Transform.
    """
    if regularization is None:
        E_array, k_E = _naive_inverse_laplace(T_array, k_T_array)
        
        if np.any(k_E < 0):
            warnings.warn("Negative values detected in k(E) array (Gibbs phenomenon). Automatically restarting with Tikhonov/NNLS regularization.")
            E_array, k_E = _tikhonov_nnls_inverse_laplace(T_array, k_T_array)
            
        return E_array, k_E
    elif regularization.lower() in ['tikhonov', 'nnls']:
        return _tikhonov_nnls_inverse_laplace(T_array, k_T_array)
    else:
        raise ValueError(f"Unknown regularization method: {regularization}")
