import numpy as np


class ProjectedGradient():
    """
    TODO:
    Add Line Search
    Add adaptive restart
    Init mechanism of options
    """
    def __init__(self, patch, max_iterations=1000, accel=True, precond=True, relcrit=1e-6):
        self.patch = patch
        self.max_iterations = max_iterations
        self.precond = precond
        self.relcrit = relcrit
        self.accel = accel

        # Find the initial step depending on the mode
        if self.precond:
            # Full step as precond
            self.alpha_init = 0.99
        else:
            # Optimal step for projected gradient
            self.alpha_init = 1.0 / self.patch.L

    def solve(self, l, x_0=None):
        # Initialize te variables
        x_k = x_0.copy() if x_0 is not None else np.zeros(self.patch.hidden_shape)
        if self.accel:
            # Momentum point variable
            y_k = x_k.copy()
        else:
            # Only an alias
            y_k = x_k
        x_kp1 = np.zeros(self.patch.hidden_shape)
        residual = np.zeros(self.patch.size)
        t_k = 1.

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
            x_kp1 += x_k
            self.patch.project_hidden_cone_(x_kp1)

            # Check convergence
            if np.linalg.norm(x_kp1 - x_k) / (np.linalg.norm(x_k) + 1e-10) < self.relcrit:
                return x_kp1, True

            # Prepare variable for next iterate
            if self.accel:
                # Update momentum point
                t_kp1 = (1 + np.sqrt(1 + 4*t_k**2)) / 2
                np.subtract(x_kp1, x_k, out=y_k)
                y_k *= (t_k - 1) / t_kp1
                y_k += x_kp1
                # New variables
                x_k, x_kp1 = x_kp1, x_k
                t_k = t_kp1
            else:
                # New variables and update y alias
                x_k, x_kp1 = x_kp1, x_k
                y_k = x_k

        return x_k, False
