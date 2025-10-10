import numpy as np


class ADMM():
    """
    ADMM solver with pure AL alternate option
    """
    def __init__(
            self,
            patch,
            max_iterations=1000,
            rel_crit=1e-4,
            abs_crit=1e-6,
            rel_obj_crit=1e-5,
            alpha=1.,
            rho_init=1e-3,
            rho_power=.2,
            rho_power_factor=.05,
            rho_lin_factor=2.,
            rho_update_ratio=10.,
            rho_update_rule='constant',
            dual_momentum=0.,
            verbose=False
        ):

        self.patch = patch
        self.max_iterations = max_iterations
        self.rel_crit = rel_crit
        self.abs_crit = abs_crit
        self.verbose = verbose
        self.alpha = alpha
        self.rel_obj_crit = rel_obj_crit

        # Set rho stuff
        self.rho_update_ratio = rho_update_ratio
        self.rho_update_rule = rho_update_rule

        cond = patch.L / patch.l

        if self.rho_update_rule in {"constant", "linear"}:
            self.rho_init = rho_init
        elif self.rho_update_rule in {"spectral"}:
            self.rho_init = np.sqrt(patch.L * patch.l) * cond**rho_power

        if self.rho_update_rule in {"linear"}:
            self.rho_factor = rho_lin_factor
        elif self.rho_update_rule in {"spectral"}:
            self.rho_factor = cond**rho_power_factor

        # Momentum stuff
        self.dual_momentum = dual_momentum

        # Rho momentum ?
        # Anderson acceleration ?
 
    def solve(self, l, x_0=None, history=None):
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
        y_km1 = x_0.copy() if x_0 is not None else np.zeros(self.patch.hidden_shape)
        y_k = np.zeros(self.patch.hidden_shape)
        x_k = np.zeros(self.patch.hidden_shape)
        l_k = np.zeros(self.patch.hidden_shape)
        if self.dual_momentum > 0.:
            l_kp1 = np.zeros(self.patch.hidden_shape)
        else:
            l_kp1 = l_k
        diff = np.zeros(self.patch.hidden_shape)

        obj_residual = np.zeros(self.patch.size)

        rho = self.rho_init

        obj_km1 = None

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
            y_k[...] = self.alpha * x_k
            y_k += (1 - self.alpha) * y_km1
            y_k += l_k / rho
            self.patch.project_hidden_cone_(y_k)
            
            # 3. Dual update
            # l_k = lkm1 + rho (x_k - y_k)
            np.subtract(x_k, y_k, out=diff)

            if self.dual_momentum > 0. and k > 0:
                if k == 1:
                    l_kp1[...] = l_k
                else:
                    # l_km1-> l_kp1
                    l_kp1 *= - self.dual_momentum
                    l_kp1 += (1. + self.dual_momentum) * l_k                    
            l_kp1 += rho * diff

            # 4. Convergence check
            primal_residual = np.linalg.norm(diff)
            dual_residual = rho * np.linalg.norm(y_k - y_km1)

            # Obj CHeck
            self.patch.apply_A(y_k, _out=obj_residual)
            obj_residual -= l
            obj_k = 0.5 * np.dot(obj_residual, obj_residual)

            rel_primal_residual = primal_residual / (max(np.linalg.norm(x_k), np.linalg.norm(y_k)) + 1e-10)
            rel_dual_residual = dual_residual / (np.linalg.norm(l_k) + 1e-10)
            rel_obj_change = np.abs(obj_k - obj_km1) / (max(obj_k, obj_km1) + 1e-10) if obj_km1 is not None else 1.

            # Log
            if self.verbose:
                self._print_iteration(k, obj_k, primal_residual, dual_residual, rel_primal_residual, rel_dual_residual, rho)
            if history is not None:
                history.append((k, obj_k, primal_residual, dual_residual, rel_primal_residual, rel_dual_residual, rho))

            if (primal_residual < self.abs_crit and dual_residual < self.abs_crit) or (rel_primal_residual < self.rel_crit and rel_dual_residual < self.rel_crit) or (rel_obj_change < self.rel_obj_crit):
                return y_k, True

            # 5. rho update
            if self.rho_update_rule in {"spectral", "linear"}:
                if primal_residual > self.rho_update_ratio * dual_residual:
                    rho *= self.rho_factor
                elif dual_residual > self.rho_update_ratio * primal_residual:
                    rho /= self.rho_factor
                rho = np.clip(rho, 1e-8, 1e8)

            # Prepare next iteration
            obj_km1 = obj_k
            y_km1, y_k = y_k, y_km1
            l_kp1, l_k = l_k, l_kp1

        if self.verbose:
            print(f"\nMaximum iterations ({self.max_iterations}) reached")
        return y_k, False

    def _print_header(self):
        """Print header for verbose output"""        
        print("=" * 70)
        print(f"ADMM Solver with {self.rho_update_rule} update")
        print(f"Max iterations: {self.max_iterations}, Tolerance: {self.rel_crit:.1e}")
        print("=" * 70)
        print(f"{'Iter':>6} {'Objective':>12} {'primal':>12} {'dual':>12} {'rel primal':>12} {'rel dual':>12} {'rho':>8}")
        print("-" * 70)

    def _print_iteration(self, k, obj, primal, dual, rel_primal, rel_dual, rho):
        """Print iteration information"""
        # Print every iteration for first 10, then every 10, then every 100
        if k < 10 or k % 10 == 0:
            print(f"{k:6d} {obj:12.6e} {primal:12.6e} {dual:12.6e} {rel_primal:12.6e} {rel_dual:12.6e} {rho:8.3f}")
