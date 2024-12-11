import numpy as np
from typing import Tuple, Any
from RandomizedNystrom import randomized_nystrom_approximation
from RPCholesky import rpcholesky

def pcg_solver(A: Any, b: np.ndarray, x0: np.ndarray, mu: float, 
               sketch_size: int, precond_type: str,
               eta: float = 1e-6, max_iter: int = 100000) -> Tuple[np.ndarray, int, float]:
    """
    Implements PCG with either Nyström or RPCholesky preconditioner for solving regularized linear systems
    
    Args:
        A: PSD matrix
        b: right-hand side vector
        x0: initial guess
        mu: regularization parameter
        sketch_size: size of the sketch (l)
        precond_type: type of preconditioner ('nystrom' or 'rpcholesky')
        eta: solution tolerance
        max_iter: maximum number of iterations
    
    Returns:
        x: Approximate solution to the regularized system
        num_iters: Number of iterations performed
        rel_error: Final relative error
    """
    # Step 1: Compute approximation based on preconditioner type
    if precond_type.lower() == 'nystrom':
        U, Lambda = randomized_nystrom_approximation(A, sketch_size)
    elif precond_type.lower() == 'rpcholesky':
        nystrom_approximation = rpcholesky(A, sketch_size)
        F = nystrom_approximation.get_left_factor()
        # Convert FF^T to USU^T form using SVD
        U, S, _ = np.linalg.svd(F, full_matrices=False)
        Lambda = S**2  # Square singular values to get eigenvalues
    else:
        raise ValueError("precond_type must be either 'nystrom' or 'rpcholesky'")
    
    def apply_preconditioner(r: np.ndarray) -> np.ndarray:
        """
        Applies the preconditioner P^{-1} to vector r using equation (15)
        """
        # P^{-1}r = r + U(Lambda + μI)^{-1}U^T r
        Ut_r = U.T @ r
        Lambda_inv = 1.0 / (Lambda + mu)
        return r + U @ (Lambda_inv * Ut_r)
    
    # Initialize
    r0 = b - ((A + mu * np.eye(A.shape[0])) @ x0)
    z0 = apply_preconditioner(r0)
    p0 = z0.copy()
    
    x = x0.copy()
    r = r0.copy()
    z = z0.copy()
    p = p0.copy()
    
    # Main PCG loop
    num_iters = 0
    for i in range(max_iter):
        num_iters = i + 1
        
        # Check convergence
        rel_error = np.linalg.norm(r) / np.linalg.norm(b)
        if rel_error <= eta:
            break
            
        v = (A + mu * np.eye(A.shape[0])) @ p
        alpha = (r.T @ z) / (p.T @ v)
        x = x + alpha * p
        r = r - alpha * v
        z = apply_preconditioner(r)
        beta = (r.T @ z) / (r0.T @ z0)
        
        r0 = r.copy()
        p = z + beta * p
        z0 = z.copy()
    
    return x, num_iters, rel_error

