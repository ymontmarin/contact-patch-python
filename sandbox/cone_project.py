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
