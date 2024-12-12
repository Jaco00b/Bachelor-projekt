import numpy as np
from typing import Tuple, Any
from scipy.linalg import qr, cholesky, svd

def randomized_nystrom_approximation(A: Any, sketch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements Algorithm 1: Randomized Nyström Approximation
    
    Args:
        A: Symmetric PSD matrix A ∈ R^{nxn}
        sketch_size: rank l for approximation
    
    Returns:
        U: Orthogonal factor of the Nyström approximation
        Lambda: Diagonal matrix (eigenvalues) such that A_nys = UΛU^T
    """
    n = A.shape[0]
    
    # Step 1: Generate Gaussian test matrix
    Omega = np.random.randn(n, sketch_size)
    
    # Step 2: Thin QR decomposition
    Omega, _ = qr(Omega, mode='economic')
    
    # Step 3: Multiply by A to get ℓ matvecs
    Y = A @ Omega
    
    # Step 4: Compute stability shift
    # eps(norm(Y, 'fro')) - using Frobenius norm
    frobenius_norm = np.linalg.norm(Y, 'fro')
    nu = np.finfo(Y.dtype).eps * frobenius_norm
    
    # Step 5: Apply stability shift
    Y_nu = Y + nu * Omega
    
    # Step 6: Cholesky decomposition of Ω^T Y_ν
    try:
        C = cholesky(Omega.T @ Y_nu, lower=True)
    except np.linalg.LinAlgError:
        # If Cholesky fails, add small diagonal perturbation
        perturbed = Omega.T @ Y_nu + 1e-12 * np.eye(sketch_size)
        C = cholesky(perturbed, lower=True)
    
    # Step 7: Compute B = Y_ν / C
    B = Y_nu @ np.linalg.inv(C)  # Alternative to division notation in algorithm
    
    # Step 8: Thin SVD
    U, Sigma, _ = svd(B, full_matrices=False)
    
    # Step 9: Remove shift and compute eigenvalues
    # λ = max{0, Σ² - νI}
    Lambda = np.maximum(0, Sigma**2 - nu)
    
    return U, Lambda