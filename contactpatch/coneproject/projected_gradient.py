import numpy as np


class ProjectedGradient():
    """
    Projected Gradient solver with optional FISTA acceleration,
    preconditioning, and adaptive restart.
    """
    def __init__(
            self,
            patch,
            max_iterations=1000,
            accel=True,
            precond=True,
            adaptive_restart=True,
            armijo=True,
            armijo_iter=20,
            armijo_sigma=.1,
            armijo_beta=.5,
            armijo_force_restart=.8,
            rel_crit=1e-6,
            abs_crit=1e-8,
            rel_obj_crit=1e-6,
            abs_obj_crit=1e-9,
            optim_crit=1e-9,
            alpha_cond=.99,
            alpha_free=.99,
            verbose=False,
        ):
        self.patch = patch

        self.max_iterations = max_iterations
        self.precond = precond
        self.adaptive_restart = adaptive_restart
        self.accel = accel
        self.verbose = verbose

        self.armijo = armijo
        self.armijo_iter = armijo_iter
        self.armijo_beta = armijo_beta
        self.armijo_sigma = armijo_sigma
        self.armijo_force_restart = armijo_force_restart

        self.abs_crit = abs_crit
        self.abs_obj_crit = abs_obj_crit
        self.rel_crit = rel_crit
        self.optim_crit = optim_crit
        self.rel_obj_crit = rel_obj_crit

        # Find the initial step depending on the mode
        if self.precond:
            # Full step as precond
            self.alpha = alpha_cond
        else:
            # Optimal step for projected gradient
            self.alpha = alpha_free / self.patch.L

    def solve(self, l, x_0=None, y_0=None, t_0=None, history=None, **kwargs):
        """
        Solve: min_{x in C} (1/2)||Ax - l||^2
        
        Args:
            l: Target vector
            x_0: Initial point (optional, defaults to zero)
        
        Returns:
            x: Solution
            converged: True if converged, False if max_iter reached
        """
        # Initialize the variables
        x_k = x_0.copy() if x_0 is not None else np.zeros(self.patch.hidden_shape)
        g_k = np.zeros(self.patch.hidden_shape)
        dx_k = np.zeros(self.patch.hidden_shape)
        x_kp1 = np.zeros(self.patch.hidden_shape)

        residual = np.zeros(self.patch.size)

        t_k = 1. if t_0 is None else t_0
        if self.accel:
            # Momentum point variable
            y_k = x_k.copy() if y_0 is None else y_0.copy()
            if self.adaptive_restart:
                dx_km1 = np.zeros(self.patch.hidden_shape)
        else:
            # Only an alias
            y_k = x_k
        if self.armijo:
            dx_trial = np.zeros(self.patch.hidden_shape)
            residual_trial = np.zeros(self.patch.size)

        obj_km1 = None

        if self.verbose:
            self._print_header()

        for k in range(self.max_iterations):
            # Residual
            # r = Ayk - b
            self.patch.apply_A(y_k, _out=residual)
            residual -= l

            obj_k = 0.5 * np.dot(residual, residual)

            # Gradient is:
            # g = A^T r
            if self.precond:
                # Hessian is A^TA so conditionned gradient is:
                #   (A^TA)^+ g = A^+ (A^+)^T A^T r
                #              = A^T (AA^T)^-1 (A^T (AA^T)^-1)^T A^T r
                #              = A^T (AA^T)^-1 (AA^T)^-1 AA^T r
                #              = A^T (AA^T)^-1 r
                # Use AAT_inv as diagonal for conditioning
                self.patch.apply_AAT_inv_(residual)
            self.patch.apply_AT(residual, _out=g_k)

            # Take the step
            # xk+1_tmp = xk - alpha g
            # xk+1 = PI_K(xk+1_tmp)
            alpha = self.alpha
            force_restart = False
            arm_i = 0
            if self.armijo:
                success = False
                for arm_i in range(self.armijo_iter):
                    # Trial point
                    x_kp1[...] = -alpha * g_k
                    x_kp1 += y_k
                    self.patch.project_hidden_cone_(x_kp1)
                    
                    # Direction
                    np.subtract(x_kp1, y_k, out=dx_trial)
                    
                    # New residual and objective
                    self.patch.apply_A(x_kp1, _out=residual_trial)
                    residual_trial -= l
                    obj_trial = 0.5 * np.dot(residual_trial, residual_trial)
                    
                    # Armijo condition
                    if self.accel:
                        armijo_crit = obj_trial <= obj_k + np.dot(g_k.flatten(), dx_trial.flatten()) + (1/(2*alpha)) * np.dot(dx_trial.flatten(), dx_trial.flatten())
                    else:
                        armijo_crit = obj_trial <= obj_k + self.armijo_sigma * np.dot(g_k.flatten(), dx_trial.flatten())
                    if armijo_crit:
                        success = True
                        break
                    alpha *= self.armijo_beta

                if not success:
                    force_restart = True
                    if self.verbose:
                        print(f"  Warning: Armijo line search failed at iteration {k}")
                elif arm_i > self.armijo_force_restart * self.armijo_iter:
                    force_restart = True
            else:
                x_kp1[...] = -alpha * g_k
                x_kp1 += y_k
                self.patch.project_hidden_cone_(x_kp1)

            # Compute the stats
            np.subtract(x_kp1, x_k, out=dx_k)
            dx_k_norm = np.linalg.norm(dx_k)  # It correspond to \|P(x_k + a grad) - x_k\| 
            x_k_norm = np.linalg.norm(x_k)

            abs_change = dx_k_norm
            rel_change =  abs_change / (x_k_norm + 1e-10)
            rel_obj_change = np.abs(obj_k - obj_km1) / (max(obj_k, obj_km1) + 1e-10) if obj_km1 is not None else 1.
            optim_crit_value = dx_k_norm

            # Check convergence
            change = (abs_change < self.abs_crit or rel_change < self.rel_crit) and rel_obj_change < self.rel_obj_crit
            obj_val = obj_k < self.abs_obj_crit
            optim_test = optim_crit_value < self.optim_crit

            # Log stat
            if self.verbose:
                self._print_iteration(k, obj_k, dx_k_norm, rel_change, optim_crit_value, t_k, alpha, arm_i)
            if history is not None:
                history.append((k, obj_k, dx_k_norm, rel_change, optim_crit_value, t_k, alpha, arm_i))

            final_crit = change or obj_val or optim_test
            # final_crit = obj_val or optim_test
            if final_crit:
                if self.verbose:
                    print(f"\nConverged in {k+1} iterations! Change:{change} | Obj val:{obj_val} | Optim crit:{optim_test}")
                return x_kp1, True, {'x_0': x_kp1, 'y_0': y_k, 't_0': t_k}

            # Prepare variable for next iterate
            if self.accel:
                if k > 0 and (force_restart or (self.adaptive_restart and (np.dot(dx_k.flatten(), dx_km1.flatten()) < 0 or obj_k > obj_km1))):
                    # Restart moment
                    t_kp1 = 1.
                    y_k[...] = x_kp1
                else:
                    # Update momentum point
                    t_kp1 = (1 + np.sqrt(1 + 4*t_k**2)) / 2
                    y_k[...] = dx_k
                    y_k *= (t_k - 1) / t_kp1
                    y_k += x_kp1
                # New variables
                x_k, x_kp1 = x_kp1, x_k
                t_k = t_kp1
                if self.adaptive_restart:
                    dx_k, dx_km1 = dx_km1, dx_k
            else:
                # New variables and update y ref
                x_k, x_kp1 = x_kp1, x_k
                y_k = x_k
            obj_km1 = obj_k

        if self.verbose:
            print(f"\nMaximum iterations ({self.max_iterations}) reached")
        return x_k, False, {'x_0': x_k, 'y_0': y_k, 't_0': t_k}

    def _print_header(self):
        """Print header for verbose output"""
        mode = []
        if self.accel:
            mode.append("FISTA")
        else:
            mode.append("PG")
        if self.precond:
            mode.append(f"Precond(a={self.alpha})")
        else:
            mode.append(f"No-Precond(a={self.alpha})")
        if self.adaptive_restart:
            mode.append("Restart")
        if self.armijo:
            mode.append(f"Armijo(N={self.armijo_iter}, s={self.armijo_sigma}, b={self.armijo_beta})")

        print("=" * 70)
        print(f"Projected Gradient Solver: {' + '.join(mode)}")
        print(f"Max iterations: {self.max_iterations}, R-Tolerance: {self.rel_crit:.1e}")
        print(f"Objective R-Tolerance: {self.rel_obj_crit:.1e}")

        print("=" * 70)
        print(f"{'Iter':>6} {'Objective':>12} {'||dx||':>12} {'Rel Change':>12} {'Optim Crit':>12} {'t_k':>8} {'a':>8} {'a_iter':>6}")
        print("-" * 70)

    def _print_iteration(self, k, obj, dx_norm, rel_change, optim_crit_value, t_k, alpha, arm_i):
        """Print iteration information"""
        # Print every iteration for first 10, then every 10, then every 100
        if k < 10 or k % 10 == 0:
            line = f"{k:6d} {obj:12.6e} {dx_norm:12.6e} {rel_change:12.6e} {optim_crit_value:12.6e}"
            if self.accel:
                line += f" {t_k:8.3f}"
            else:
                line += f"    N/A"
            if self.armijo:
                line += f" {alpha:8.3f} {arm_i:6d} "
            else:
                line += f"    N/A    N/A"
            print(line)
