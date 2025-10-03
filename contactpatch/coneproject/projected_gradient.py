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
            adaptive_restart=False,
            armijo=False,
            rel_crit=1e-6,
            verbose=False):
        self.patch = patch
        self.max_iterations = max_iterations
        self.precond = precond
        self.rel_crit = rel_crit
        self.adaptive_restart = adaptive_restart
        self.accel = accel
        self.armijo = armijo
        self.verbose = verbose

        # Find the initial step depending on the mode
        if self.precond:
            # Full step as precond
            self.alpha_init = 0.99
        else:
            # Optimal step for projected gradient
            self.alpha_init = 1.0 / self.patch.L

    def solve(self, l, x_0=None):
        """
        Solve: min_{x in C} (1/2)||Ax - l||^2
        
        Args:
            l: Target vector
            x_0: Initial point (optional, defaults to zero)
        
        Returns:
            x: Solution
            converged: True if converged, False if max_iter reached
        """
        # Initialize te variables
        x_k = x_0.copy() if x_0 is not None else np.zeros(self.patch.hidden_shape)
        if self.accel:
            # Momentum point variable
            y_k = x_k.copy()
            if self.adaptive_restart:
                dx_k = np.zeros(self.patch.hidden_shape)
        else:
            # Only an alias
            y_k = x_k
        dx_kp1 = np.zeros(self.patch.hidden_shape)
        x_kp1 = np.zeros(self.patch.hidden_shape)
        residual = np.zeros(self.patch.size)
        t_k = 1.

        if self.verbose:
            self._print_header()

        for k in range(self.max_iterations):
            # r = Ayk - b
            self.patch.apply_A(y_k, _out=residual)
            residual -= l
            if self.precond:
                # Use AAT_inv as diagonal for conditioning
                self.patch.apply_AAT_inv_(residual)

            # g = A^TD r
            # xk+1_tmp = xk - alpha g
            # xk+1 = PI_K(xk+1_tmp)
            self.patch.apply_AT(residual, _out=x_kp1)
            x_kp1 *= -self.alpha_init
            x_kp1 += y_k
            self.patch.project_hidden_cone_(x_kp1)

            np.subtract(x_kp1, x_k, out=dx_kp1)
            dx_norm = np.linalg.norm(dx_kp1)
            x_norm = np.linalg.norm(x_k)
            rel_change = dx_norm / (x_norm + 1e-10)

            if self.verbose:
                obj = 0.5 * np.linalg.norm(residual)**2
                self._print_iteration(k, obj, dx_norm, rel_change, t_k)

            # Check convergence
            if rel_change < self.rel_crit:
                if self.verbose:
                    print(f"\nConverged in {k+1} iterations!")
                return x_kp1, True

            # Prepare variable for next iterate
            if self.accel:
                if k > 0 and self.adaptive_restart and np.dot(dx_kp1, dx_k) < 0:
                    # Restart moment
                    t_kp1 = 1.
                    y_k[...] = x_kp1
                else:
                    # Update momentum point
                    t_kp1 = (1 + np.sqrt(1 + 4*t_k**2)) / 2
                    y_k[...] = dx_kp1
                    y_k *= (t_k - 1) / t_kp1
                    y_k += x_kp1
                # New variables
                x_k, x_kp1 = x_kp1, x_k
                t_k = t_kp1
                if self.adaptive_restart:
                    dx_k, dx_kp1 = dx_kp1, dx_k
            else:
                # New variables and update y ref
                x_k, x_kp1 = x_kp1, x_k
                y_k = x_k
        if self.verbose:
            print(f"\nMaximum iterations ({self.max_iterations}) reached")
        return x_k, False

    def _print_header(self):
        """Print header for verbose output"""
        mode = []
        if self.accel:
            mode.append("FISTA")
        else:
            mode.append("PG")
        if self.precond:
            mode.append("Precond")
        if self.adaptive_restart:
            mode.append("Restart")
        
        print("=" * 70)
        print(f"Projected Gradient Solver: {' + '.join(mode)}")
        print(f"Max iterations: {self.max_iterations}, Tolerance: {self.rel_crit:.1e}")
        print("=" * 70)
        print(f"{'Iter':>6} {'Objective':>12} {'||dx||':>12} {'Rel Change':>12} {'t_k':>8}")
        print("-" * 70)

    def _print_iteration(self, k, obj, dx_norm, rel_change, t_k):
        """Print iteration information"""
        # Print every iteration for first 10, then every 10, then every 100
        if k < 10 or k % 10 == 0:
            if self.accel:
                print(f"{k:6d} {obj:12.6e} {dx_norm:12.6e} {rel_change:12.6e} {t_k:8.3f}")
            else:
                print(f"{k:6d} {obj:12.6e} {dx_norm:12.6e} {rel_change:12.6e}    N/A")
