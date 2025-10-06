import numpy as np


class ADMM():
    """
    ADMM solver with pure AL alternate option
    """
    def __init__(
            self,
            patch,
            max_iterations=1000,
            crit=1e-6,
            rho_init=1e-3,
            verbose=False):
        self.patch = patch
        self.max_iterations = max_iterations
        self.crit = crit
        self.rho_init = rho_init
        self.verbose = verbose

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
        y_km1 = x_0.copy() if x_0 is not None else np.zeros(self.patch.hidden_shape)
        y_k = np.zeros(self.patch.hidden_shape)
        x_k = np.zeros(self.patch.hidden_shape)
        l_k = np.zeros(self.patch.hidden_shape)
        diff = np.zeros(self.patch.hidden_shape)

        # TODO Set the rho with spectral
        rho = self.rho_init

        if self.verbose:
            self._print_header()

        for k in range(self.max_iterations):
            # 1. X-update: Solve linear system (ATA + rhoI) x_k = A^Tl + rho y_km1 - l_km1
            self.patch.apply_AT(l, _out=x_k)
            x_k += rho * y_km1
            x_k -= l_k
            self.patch.apply_ATA_reg_inv_(x_k, rho=rho)

            # 2. Y-update:
            # y_k = Proj(x_k + l_km1 / rho)
            np.add(x_k, l_k / rho, out=y_k)
            self.patch.project_hidden_cone_(y_k)
            
            # 3. Dual update
            # l_k = lkm1 + rho (x_k - y_k)
            np.substract(x_k, y_k, out=diff)
            l_k += rho * diff
            
            # 4. Convergence check
            primal_residual = np.linalg.norm(diff)
            dual_residual = rho * np.linalg.norm(y_k - y_km1)
            if self.verbose:
                self._print_iteration(k, primal_residual, dual_residual)

            if primal_residual < self.crit and dual_residual < self.crit:
                return x_k, True

            # 5. rho update
            # TODO

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
