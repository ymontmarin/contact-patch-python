def enhanced_fista(A, b, mu, n, AAT_inv):
    # Pre-computation
    lambda_max = max_eigenvalue(A @ A.T)
    
    # Initialize
    X_k_minus_1 = X_k = np.zeros(3*n)
    t_k = 1.0
    
    for k in range(max_iterations):
        # 1. Momentum extrapolation
        if k > 0:
            t_k_plus_1 = (1 + sqrt(1 + 4*t_k²)) / 2
            theta = (t_k - 1) / t_k_plus_1
            Y_k = X_k + theta * (X_k - X_k_minus_1)
        else:
            Y_k = X_k
        
        # 2. Gradient computation (with preconditioning option)
        residual = A @ Y_k - b
        if use_preconditioning:
            grad = A.T @ (AAT_inv @ residual)
            alpha_init = 0.99
        else:
            grad = A.T @ residual  
            alpha_init = 1.0 / lambda_max
        
        # 3. Line search (optional)
        if use_line_search:
            alpha, X_k_plus_1 = armijo_line_search(Y_k, grad, alpha_init, mu, n)
        else:
            X_k_plus_1 = project_ice_cream_blocks(Y_k - alpha_init * grad, mu, n)
        
        # 4. Adaptive restart (optional)
        if k > 0 and should_restart(X_k, X_k_minus_1, Y_k):
            t_k = 1.0  # Reset momentum
        
        # 5. Convergence check
        if converged(X_k, X_k_plus_1):
            return X_k_plus_1, A @ X_k_plus_1
        
        X_k_minus_1, X_k, t_k = X_k, X_k_plus_1, t_k_plus_1

def specialized_admm(A, b, mu, n, rho=1.0):
    # Pre-computation  
    ATA = A.T @ A  # (3n × 3n)
    ATA_plus_rhoI = ATA + rho * np.eye(3*n)
    
    if use_prefactorization:
        L = cholesky_decompose(ATA_plus_rhoI)  # One-time O(n³)
    
    ATb = A.T @ b
    
    # Initialize
    X = Y = lambda_dual = np.zeros(3*n)
    
    for k in range(max_iterations):
        # 1. X-update: Solve linear system
        rhs = ATb + rho * Y - lambda_dual
        
        if use_prefactorization:
            X = cholesky_solve(L, rhs)  # O(n²) per iteration
        else:
            X = solve_linear_system(ATA_plus_rhoI, rhs)  # O(n³) per iteration
        
        # 2. Y-update: Parallel SOC projections  
        temp = X + lambda_dual / rho
        for i in range(n):
            Y[3*i:3*i+3] = project_ice_cream_cone(temp[3*i:3*i+3], mu)
        
        # 3. Dual update
        lambda_dual = lambda_dual + rho * (X - Y)
        
        # 4. Convergence check
        primal_residual = norm(X - Y)
        dual_residual = rho * norm(Y - Y_old)  
        if primal_residual < tol and dual_residual < tol:
            return X, A @ X

def project_ice_cream_cone(u, mu):
    """Project u = (x,y,z) onto {(x,y,z) : ||(x,y)||₂ ≤ μz}"""
    x, y, z = u[0], u[1], u[2]
    norm_xy = sqrt(x² + y²)
    
    if norm_xy <= mu * z:  # Inside cone
        return u
    elif norm_xy <= -mu * z:  # In negative cone  
        return np.zeros(3)
    else:  # Project to boundary
        alpha = (mu * z + norm_xy) / (mu² + 1)
        return np.array([
            alpha * mu * x / norm_xy,
            alpha * mu * y / norm_xy,
            alpha
        ])

def check_convergence(X_old, X_new, A, b, tol=1e-8):
    # Step convergence
    step_conv = norm(X_new - X_old) < tol * (1 + norm(X_old))
    
    # Objective convergence  
    f_old = 0.5 * norm(A @ X_old - b)**2
    f_new = 0.5 * norm(A @ X_new - b)**2
    obj_conv = abs(f_new - f_old) < tol * (1 + abs(f_old))
    
    # Residual convergence (since you want AX)
    res_conv = norm(A @ X_new - b) < sqrt(tol)
    
    return step_conv and obj_conv
