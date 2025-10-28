import numpy as np
import matplotlib.pyplot as plt

from contactpatch.patches import PolygonContactPatch


def plot_convergence_history(
    history, title="Solver Convergence History", solver_type="PGD"
):
    """
    Plot convergence history from PGD or ADMM solver

    Args:
        history: List of tuples from solver
                 PGD: (iter, obj, norm_grad, step_size, [restart_flag])
                 ADMM: (iter, obj, primal_residual, dual_residual, rho)
        title: Plot title
        solver_type: 'PGD' or 'ADMM'
    """
    if len(history) == 0:
        print("No history to plot!")
        return

    history = np.array(history)

    if solver_type == "PGD":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Objective
        ax = axes[0, 0]
        ax.semilogy(history[:, 0], history[:, 1], "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.set_title("Objective Value")
        ax.grid(True, alpha=0.3)

        # Dx norm
        ax = axes[0, 1]
        ax.semilogy(history[:, 0], history[:, 2], "r-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Dx Norm")
        ax.set_title("Dx Norm (Optimality)")
        ax.grid(True, alpha=0.3)

        # Rel change
        ax = axes[1, 0]
        ax.semilogy(history[:, 0], history[:, 3], "g-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative change")
        ax.set_title("Relative change")
        ax.grid(True, alpha=0.3)

        # Optim crit
        ax = axes[1, 1]
        ax.semilogy(history[:, 0], history[:, 4], "g-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative change")
        ax.set_title("Step Size (α)")
        ax.grid(True, alpha=0.3)

    elif solver_type == "ADMM":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Objective
        ax = axes[0, 0]
        ax.semilogy(history[:, 0], history[:, 1], "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.set_title("Objective Value")
        ax.grid(True, alpha=0.3)

        # Primal and Dual residuals
        ax = axes[0, 1]
        ax.semilogy(history[:, 0], history[:, 2], "r-", linewidth=2, label="Primal")
        ax.semilogy(history[:, 0], history[:, 3], "g-", linewidth=2, label="Dual")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.set_title("ADMM Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rho evolution
        ax = axes[1, 0]
        ax.semilogy(history[:, 0], history[:, 4], "purple", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ρ (penalty parameter)")
        ax.set_title("Penalty Parameter Evolution")
        ax.grid(True, alpha=0.3)

        # Residual ratio
        ax = axes[1, 1]
        ratio = history[:, 2] / (history[:, 3] + 1e-10)
        ax.semilogy(history[:, 0], ratio, "orange", linewidth=2)
        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Balanced")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Primal / Dual Ratio")
        ax.set_title("Residual Balance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==================================================================
# FAILURE EXAMPLE #1
# PGD - PGD_baseline
# n_sample=10, poly_id=6
# ==================================================================

vis_1 = np.array(
    [
        [np.float64(3.0877401669292053), np.float64(1.8694635258853325)],
        [np.float64(1.3431635973893488), np.float64(2.42466851903562)],
        [np.float64(-0.32982280302442385), np.float64(2.262111858702421)],
        [np.float64(-1.0774178126674248), np.float64(2.1319829169950557)],
        [np.float64(-1.42293971447423), np.float64(1.9976628239025853)],
        [np.float64(-2.612829701215653), np.float64(1.1425727938946786)],
        [np.float64(-2.2354909021534124), np.float64(-2.50463512290216)],
        [np.float64(-0.8795174358077388), np.float64(-3.569324465844434)],
        [np.float64(0.1263305721037957), np.float64(-3.6459442251684875)],
        [np.float64(4.000784032920533), np.float64(-2.1085586245006116)],
    ]
)

test_point_1 = np.array(
    [
        np.float64(0.13313787264842705),
        np.float64(-1.8970327053422764),
        np.float64(1.5298530918784843),
        np.float64(-0.7523979653927262),
        np.float64(0.7800117450661668),
        np.float64(4.470171678520062),
    ]
)

# Test this case:
mu = 2.0
poly_1 = PolygonContactPatch(
    vis=vis_1,
    mu=mu,
    ker_precompute=False,
    warmstart_strat=None,
    solver_tyep="PGD",
    solver_kwargs={"max_iterations": 5000, "verbose": True},
)

# Try to project (should fail or take many iterations):
history_1 = []
result_1 = poly_1.project_cone(test_point_1, history=history_1)
print(result_1)
plot_convergence_history(
    history_1, title="Solver Convergence History", solver_type="PGD"
)

print(f"Converged: {len(history_1) < 5000}, Iterations: {len(history_1)}")
