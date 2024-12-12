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
    Omega = np.linalg.qr(Omega, mode='reduced')[0]
    
    # Step 3: Multiply by A to get ℓ matvecs
    Y = A @ Omega
    
    # Step 4: Compute stability shift
    nu = np.finfo(float).eps * np.linalg.norm(Y, 'fro')
    
    # Step 5: Apply stability shift
    Y_nu = Y + nu * Omega
    
    # Step 6: Cholesky decomposition of Ω^T Y_ν
    C = np.linalg.cholesky(Omega.T @ Y_nu)
    
    # Step 7: Compute B = Y_ν / C
    B = Y_nu @ np.linalg.inv(C.T) 
    
    # Step 8: Thin SVD
    U, Sigma, _ = np.linalg.svd(B, full_matrices=False)
    
    # Step 9: Remove shift and compute eigenvalues
    # λ = max{0, Σ² - νI}
    Lambda = np.maximum(0, Sigma**2-nu)
    
    return U, Lambda