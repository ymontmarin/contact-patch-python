import numpy as np


class ADMM:
    """
    ADMM solver with pure AL alternate option
    """

    def __init__(
        self,
        patch,
        max_iterations=2000,
        rel_crit=1e-5,
        abs_crit=1e-6,
        abs_obj_crit=1e-12,
        min_residual_threshold=1e-8,
        rho_clip=1e6,
        prox=1e-6,
        alpha=1.0,
        rho_init=1e-1,
        rho_power=0.3,
        rho_power_factor=0.15,
        rho_lin_factor=2.0,
        rho_update_ratio=5.0,
        rho_update_cooldown=1.0,
        rho_adaptive_fraction=0.3,
        rho_update_rule="osqp",
        dual_momentum=0.3,
        verbose=False,
    ):
        self.patch = patch

        self.max_iterations = max_iterations
        self.rho_clip = rho_clip
        self.min_residual_threshold = min_residual_threshold
        self.verbose = verbose
        self.alpha = alpha
        self.dual_momentum = dual_momentum
        self.prox = prox

        self.rel_crit = rel_crit
        self.abs_crit = abs_crit
        self.abs_obj_crit = abs_obj_crit

        # Set rho stuff
        self.rho_adaptive_fraction = rho_adaptive_fraction
        self.rho_update_ratio = rho_update_ratio
        self.rho_update_rule = rho_update_rule
        self.rho_update_cooldown = rho_update_cooldown
        cond = patch.L / patch.l

        if self.rho_update_rule in {"constant", "linear"}:
            self.rho_init = rho_init
        elif self.rho_update_rule in {"spectral"}:
            self.rho_init = np.sqrt(patch.L * patch.l) * cond**rho_power
        elif self.rho_update_rule in {"osqp"}:
            self.rho_init = np.sqrt(patch.L / patch.l)

        if self.rho_update_rule in {"linear"}:
            self.rho_factor = rho_lin_factor
        elif self.rho_update_rule in {"spectral"}:
            self.rho_factor = cond**rho_power_factor

    def solve(self, l, x_0=None, l_0=None, rho_0=None, history=None, **kwargs):
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
        lhs = np.zeros(self.patch.hidden_shape)
        l_k = np.zeros(self.patch.hidden_shape)

        nu = self.prox
        beta = self.dual_momentum
        rho = self.rho_init if rho_0 is None else rho_0
        sqrt_m = np.sqrt(self.patch.m)

        # Initialize l_0
        if l_0 is not None:
            l_k[...] = l_0
        elif x_0 is not None:
            # Initialize with gradient: λ_0 = A^T(Ax_0 - l)
            residual_0 = np.zeros(self.patch.n)
            self.patch.apply_A(x_0, _out=residual_0)
            residual_0 -= l
            l_k = np.zeros(self.patch.hidden_shape)
            self.patch.apply_AT(residual_0, _out=l_k)

        if self.dual_momentum > 0.0:
            l_kp1 = l_k.copy()
        else:
            l_kp1 = l_k

        diff = np.zeros(self.patch.hidden_shape)
        obj_residual = np.zeros(self.patch.n)

        # Check if x_0 is feasible to fix alpha
        if x_0 is not None:
            x_k[...] = x_0
            self.patch.project_hidden_cone(x_0, _out=y_k)
            feasibility_error = np.linalg.norm(x_k - y_k)
            if feasibility_error < 1e-8:
                # x_0 is feasible, use conservative alpha
                alpha = 1.0 if self.alpha > 1.0 else self.alpha
                if self.verbose:
                    print("Initial point is feasible, using alpha=1.0")
            else:
                alpha = self.alpha
        else:
            alpha = self.alpha

        rho_update_delay = 0

        if self.verbose:
            self._print_header()
        for k in range(self.max_iterations):
            # 1. X-update (with proximal operator):
            # Solve linear system (ATA + (rho + nu)I) x_kp1 = A^Tl + rho y_k + nu x_k - l_k
            self.patch.apply_AT(l, _out=lhs)
            lhs += nu * x_k
            lhs += rho * y_km1
            lhs -= l_k
            self.patch.apply_ATA_reg_inv(lhs, _out=x_k, rho=rho + nu)

            # 2. Y-update (with over rolation):
            # y_kp1 = Proj( a x_k + (1-a) y_k + l_k / rho)
            np.multiply(x_k, alpha, out=y_k)
            y_k += (1 - alpha) * y_km1
            y_k += l_k / rho
            self.patch.project_hidden_cone_(y_k)

            # 3. Dual update (with momentum)
            # l_kp1 = (1+beta)lk - beta lkm1 + rho (x_kp1 - y_kp1)
            #       = lk + beta (lk - lkm1) + rho (x_kp1 - y_kp1)
            np.subtract(x_k, y_k, out=diff)

            if beta > 0.0 and k > 0:
                # l_km1-> l_kp1
                l_kp1 *= -beta
                l_kp1 += (1.0 + beta) * l_k
            l_kp1 += rho * diff

            # 4. Convergence check
            primal_residual = np.linalg.norm(diff)
            np.subtract(y_k, y_km1, out=diff)
            dual_residual = rho * np.linalg.norm(diff)

            eps_primal = sqrt_m * self.abs_crit + self.rel_crit * max(
                np.sqrt(np.sum(x_k * x_k)), np.sqrt(np.sum(y_k * y_k))
            )
            eps_dual = sqrt_m * self.abs_crit + self.rel_crit * np.sqrt(
                np.sum(l_k * l_k)
            )

            self.patch.apply_A(y_k, _out=obj_residual)
            obj_residual -= l
            obj_k = 0.5 * np.sum(obj_residual * obj_residual)

            admm_converged = primal_residual < eps_primal and dual_residual < eps_dual
            obj_near_0 = obj_k < self.abs_obj_crit

            # Log
            if self.verbose:
                self._print_iteration(k, obj_k, primal_residual, dual_residual, rho)
            if history is not None:
                history.append((k, obj_k, primal_residual, dual_residual, rho))

            # Exit cond
            if admm_converged or obj_near_0:
                return y_k, True, {"x_0": y_k, "l_0": l_kp1, "rho_0": rho}

            rho_update_delay += 1
            # 5. rho update
            if rho_update_delay > self.rho_update_cooldown:
                change = False
                residuals_active = (
                    primal_residual > self.min_residual_threshold
                    or dual_residual > self.min_residual_threshold
                )
                if self.rho_update_rule in {"spectral", "linear"} and residuals_active:
                    if primal_residual > self.rho_update_ratio * dual_residual:
                        rho *= self.rho_factor
                        change = True
                    elif dual_residual > self.rho_update_ratio * primal_residual:
                        rho /= self.rho_factor
                        change = True
                elif self.rho_update_rule in {"osqp"}:
                    ratio = primal_residual / (dual_residual + 1e-10)
                    # Adaptive scaling
                    if (
                        ratio > self.rho_update_ratio
                        or ratio < 1 / self.rho_update_ratio
                    ):
                        rho_new = rho * np.sqrt(ratio)
                        rho = (
                            rho * (1 - self.rho_adaptive_fraction)
                            + rho_new * self.rho_adaptive_fraction
                        )
                        change = True
                if change:
                    rho = np.clip(rho, 1 / self.rho_clip, self.rho_clip)
                    rho_update_delay = 0

            # Swap variable for next iteration
            obj_km1 = obj_k
            y_km1, y_k = y_k, y_km1
            l_kp1, l_k = l_k, l_kp1

        if self.verbose:
            print(f"\nMaximum iterations ({self.max_iterations}) reached")
        return y_km1, False, {"x_0": y_km1, "l_0": l_k, "rho_0": rho}

    def _print_header(self):
        """Print header for verbose output"""
        print("=" * 70)
        print(f"ADMM Solver with {self.rho_update_rule} update")
        print(f"Max iterations: {self.max_iterations}, Tolerance: {self.rel_crit:.1e}")
        print("=" * 70)
        print(f"{'Iter':>6} {'Objective':>12} {'primal':>12} {'dual':>12} {'rho':>8}")
        print("-" * 70)

    def _print_iteration(self, k, obj, primal, dual, rho):
        """Print iteration information"""
        # Print every iteration for first 10, then every 10, then every 100
        if k < 10 or k % 10 == 0:
            print(f"{k:6d} {obj:12.6e} {primal:12.6e} {dual:12.6e} {rho:8.3f}")
