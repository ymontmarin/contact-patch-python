import numpy as np
import pandas as pd
import time
from itertools import product
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import warnings
from joblib import Parallel, delayed

from contactpatch.patches import PolygonContactPatch


MU = 1.0
KER_PRECOMPUTE = False
MAX_TIME = 0.05
MAX_ITER = 20000


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run"""

    solver_type: str
    config_name: str
    n_vertices: int
    problem_id: int
    test_point_id: int
    mu: float
    iterations: int
    time_seconds: float
    converged: bool
    final_objective: float
    hp_dict: Dict[str, Any] = None

    def to_dict(self):
        """Convert to flat dictionary"""
        d = asdict(self)
        hp = d.pop("hp_dict", {})
        for key, value in hp.items():
            d[f"hp_{key}"] = value
        return d


class OptimizationBenchmark:
    """Enhanced benchmark suite with comprehensive failure analysis"""

    # Fixed convergence criteria for fair comparison
    STANDARD_PGD_CONVERGENCE = {
        "rel_crit": 1e-6,
        "abs_crit": 1e-8,
        "rel_obj_crit": 1e-7,
        "abs_obj_crit": 1e-12,
        "optim_crit": 1e-13,
    }

    STANDARD_ADMM_CONVERGENCE = {
        "rel_crit": 1e-5,
        "abs_crit": 1e-6,
        "abs_obj_crit": 1e-12,
    }

    def __init__(
        self,
        n_vertice_min: int = 3,
        n_vertice_max: int = 25,
        mu_min: float = 0.01,
        mu_max: float = 2.0,
        n_problems: int = 30,
        n_test_points: int = 20,
        ker_precompute: bool = KER_PRECOMPUTE,
        max_time: float = MAX_TIME,
        max_iter: int = MAX_ITER,
    ):
        self.n_vertice_min = n_vertice_min
        self.n_vertice_max = n_vertice_max

        self.mu_min = mu_min
        self.mu_max = mu_max

        self.n_problems = n_problems
        self.n_test_points = n_test_points

        self.max_time = max_time
        self.max_iter = max_iter
        self.ker_precompute = ker_precompute

        self.generate_test_problems()

        self.results = []

    def generate_test_problems(self):
        """Generate test polygons and points"""

        print("Generating test points...")
        self.test_points = []
        for point_id in range(self.n_test_points):
            fl = np.random.randn(6)
            scale = np.random.uniform(0.1, 10.0)
            fl = fl / (np.linalg.norm(fl) + 1e-10) * scale
            self.test_points.append(fl)

        print("Generating test problems...")
        n_vertices_list = np.random.randint(
            self.n_vertice_min, self.n_vertice_max, size=self.n_problems
        )
        mu_list = np.random.uniform(self.mu_min, self.mu_max, size=self.n_problems)
        self.test_problems = []
        for problem_id, (n_vertices, mu) in enumerate(zip(n_vertices_list, mu_list)):
            factor = 10
            for i in range(5):
                try:
                    vis = PolygonContactPatch.generate_polygon_vis(
                        N_sample=factor * n_vertices, aimed_n=n_vertices
                    )
                    break
                except Exception:
                    factor *= 10
            self.test_problems.append(
                {
                    "problem_id": problem_id,
                    "n_vertices": n_vertices,
                    "mu": mu,
                    "vis": vis,
                }
            )

        print(f"✓ Generated {len(self.test_points)} test points")
        print(f"✓ Generated {len(self.test_problems)} test problems")
        print(f"  Total test cases: {len(self.test_problems) * self.n_test_points}")
        return self

    def get_pgd_configs(self, mode: str = "default"):
        """Get PGD configurations - HYPERPARAMETERS ONLY"""
        configs = {}

        if mode == "baseline":
            configs["PGD_baseline"] = {
                "accel": False,
                "precond": True,
                "adaptive_restart": False,
                "armijo": False,
                "alpha": 0.99,
            }
            configs["PGD_nocond"] = {
                "accel": False,
                "precond": False,
                "adaptive_restart": False,
                "armijo": False,
                "alpha": 0.99,
            }
            configs["PGD_fista"] = {
                "accel": True,
                "precond": True,
                "adaptive_restart": False,
                "armijo": False,
                "alpha": 0.99,
            }
            configs["PGD_fista_restart"] = {
                "accel": True,
                "precond": True,
                "adaptive_restart": True,
                "armijo": False,
                "alpha": 0.99,
            }
            configs["PGD_armijo"] = {
                "accel": False,
                "precond": True,
                "adaptive_restart": False,
                "armijo": True,
                "armijo_iter": 20,
                "armijo_sigma": 0.1,
                "armijo_beta": 0.5,
                "alpha": 1.0,
            }
            configs["PGD_fista_restart_armijo"] = {
                "accel": True,
                "precond": True,
                "adaptive_restart": True,
                "armijo": True,
                "armijo_iter": 20,
                "armijo_sigma": 0.1,
                "armijo_beta": 0.5,
                "alpha": 1.0,
            }

        elif mode == "alpha_sweep":
            for alpha in [0.5, 0.7, 0.85, 0.95, 0.99, 1.0]:
                configs[f"PGD_fista_restart_alpha_{alpha}"] = {
                    "accel": True,
                    "precond": True,
                    "adaptive_restart": True,
                    "armijo": False,
                    "alpha": alpha,
                }

        elif mode == "armijo_sigma_sweep":
            for sigma in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
                configs[f"PGD_armijo_sigma_{sigma}"] = {
                    "accel": False,
                    "precond": True,
                    "adaptive_restart": False,
                    "armijo": True,
                    "armijo_iter": 20,
                    "armijo_sigma": sigma,
                    "armijo_beta": 0.5,
                    "alpha": 1.0,
                }

        elif mode == "armijo_beta_sweep":
            for beta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                configs[f"PGD_armijo_beta_{beta}"] = {
                    "accel": False,
                    "precond": True,
                    "adaptive_restart": False,
                    "armijo": True,
                    "armijo_iter": 20,
                    "armijo_sigma": 0.1,
                    "armijo_beta": beta,
                    "alpha": 1.0,
                }

        elif mode == "feature_combinations":
            for accel, restart, armijo, precond in product(*[[True, False]] * 4):
                if restart and not accel:
                    continue
                name_parts = ["PGD"]
                if accel:
                    name_parts.append("fista")
                if restart:
                    name_parts.append("restart")
                if armijo:
                    name_parts.append("armijo")
                if precond:
                    name_parts.append("precond")
                config_name = "_".join(name_parts)

                config = {
                    "accel": accel,
                    "precond": precond,
                    "adaptive_restart": restart if accel else False,
                    "armijo": armijo,
                    "alpha": 0.999,
                }
                if armijo:
                    config.update(
                        {
                            "armijo_iter": 20,
                            "armijo_sigma": 0.1,
                            "armijo_beta": 0.5,
                        }
                    )
                configs[config_name] = config

        elif mode == "maxi_GS":
            for alpha in [0.5, 0.8, 0.99, 1.0]:
                for accel, restart, armijo, precond in product(*[[True, False]] * 4):
                    if restart and not accel:
                        continue
                    name_parts = ["PGD"]
                    if accel:
                        name_parts.append("fista")
                    if restart:
                        name_parts.append("restart")
                    if armijo:
                        name_parts.append("armijo")
                    if precond:
                        name_parts.append("precond")
                    name_parts.append(f"alpha_{alpha}")

                    config_name = "_".join(name_parts)
                    config = {
                        "accel": accel,
                        "precond": precond,
                        "adaptive_restart": restart if accel else False,
                        "armijo": armijo,
                        "alpha": alpha,
                    }
                    if armijo:
                        for a_iter, a_sigma, a_beta, a_fr in product(
                            [5, 10, 20],
                            [0.01, 0.1, 0.5],
                            [0.3, 0.5, 0.7],
                            [0.5, 0.9],
                        ):
                            config_spe = config.copy()
                            config_spe.update(
                                {
                                    "armijo_iter": a_iter,
                                    "armijo_sigma": a_sigma,
                                    "armijo_beta": a_beta,
                                    "armijo_force_restart": a_fr,
                                }
                            )
                            config_name_spe = "_".join(
                                [
                                    config_name,
                                    f"aiter_{a_iter}",
                                    f"asigma_{a_sigma}",
                                    f"abeta_{a_beta}",
                                    f"afr_{a_fr}",
                                ]
                            )
                            configs[config_name_spe] = config_spe
                    else:
                        configs[config_name] = config

        elif mode == "GS_best":
            for alpha in [0.9, 0.95, 0.99, 1.0, 1.01, 1.1]:
                name_parts = ["PGD_fista_restart_precond"]
                name_parts.append(f"alpha_{alpha}")
                config_name = "_".join(name_parts)
                config = {
                    "accel": True,
                    "precond": True,
                    "adaptive_restart": False,
                    "armijo": False,
                    "alpha": alpha,
                }
                configs[config_name] = config

        for name, config in configs.items():
            config.update(self.STANDARD_PGD_CONVERGENCE)
            config.update(
                {
                    "max_iterations": self.max_iter,
                    "verbose": False,
                }
            )

        return configs

    def get_admm_configs(self, mode: str = "default"):
        """Get ADMM configurations - HYPERPARAMETERS ONLY"""
        configs = {}

        if mode == "baseline":
            configs["ADMM_constant"] = {
                "rho_update_rule": "constant",
                "alpha": 1.0,
                "dual_momentum": 0.0,
                "rho_init": 0.1,
            }
            configs["ADMM_linear"] = {
                "rho_update_rule": "linear",
                "alpha": 1.0,
                "dual_momentum": 0.0,
                "rho_init": 0.1,
                "rho_lin_factor": 2.0,
            }
            configs["ADMM_spectral"] = {
                "rho_update_rule": "spectral",
                "alpha": 1.0,
                "dual_momentum": 0.0,
                "rho_power": 0.3,
                "rho_power_factor": 0.15,
            }
            configs["ADMM_osqp"] = {
                "rho_update_rule": "osqp",
                "alpha": 1.0,
                "dual_momentum": 0.0,
                "rho_init": 0.1,
                "rho_adaptive_fraction": 0.4,
            }

        elif mode == "rho_init_sweep":
            for rho_init in [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]:
                configs[f"ADMM_rho_init_{rho_init}"] = {
                    "rho_update_rule": "linear",
                    "alpha": 1.0,
                    "dual_momentum": 0.0,
                    "rho_init": rho_init,
                    "rho_lin_factor": 2.0,
                }

        elif mode == "alpha_sweep":
            for alpha in [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9]:
                configs[f"ADMM_alpha_{alpha}"] = {
                    "rho_update_rule": "linear",
                    "alpha": alpha,
                    "dual_momentum": 0.0,
                    "rho_init": 0.1,
                    "rho_lin_factor": 2.0,
                }

        elif mode == "momentum_sweep":
            for beta in [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]:
                configs[f"ADMM_momentum_{beta}"] = {
                    "rho_update_rule": "linear",
                    "alpha": 1.0,
                    "dual_momentum": beta,
                    "rho_init": 0.1,
                    "rho_lin_factor": 2.0,
                }

        elif mode == "rho_factor_sweep":
            for factor in [1.5, 2.0, 2.5, 3.0, 4.0]:
                configs[f"ADMM_rho_factor_{factor}"] = {
                    "rho_update_rule": "linear",
                    "alpha": 1.0,
                    "dual_momentum": 0.0,
                    "rho_init": 0.1,
                    "rho_lin_factor": factor,
                }

        elif mode == "osqp_fraction_sweep":
            for fraction in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                configs[f"ADMM_osqp_frac_{fraction}"] = {
                    "rho_update_rule": "osqp",
                    "alpha": 1.0,
                    "dual_momentum": 0.0,
                    "rho_init": 0.1,
                    "rho_adaptive_fraction": fraction,
                }

        elif mode == "combined_best":
            for alpha, momentum in product([1.0, 1.3, 1.5], [0.0, 0.3, 0.5]):
                configs[f"ADMM_alpha_{alpha}_mom_{momentum}"] = {
                    "rho_update_rule": "linear",
                    "alpha": alpha,
                    "dual_momentum": momentum,
                    "rho_init": 0.1,
                    "rho_lin_factor": 2.0,
                }

        elif mode == "maxi_GS":
            for alpha, dual_momentum, prox, rho_update_rule in product(
                [1.0, 1.3, 1.6, 1.9],
                [0.0, 0.2, 0.4],
                [0.0, 1e-6, 1e-3],
                ["constant", "linear", "osqp", "spectral"],
            ):
                name_parts = ["ADMM"]
                name_parts.append(f"{rho_update_rule}")
                name_parts.append(f"alpha_{alpha}")
                name_parts.append(f"betra_{dual_momentum}")
                name_parts.append(f"prox_{prox}")

                config_name = "_".join(name_parts)
                config = {
                    "rho_update_rule": rho_update_rule,
                    "alpha": alpha,
                    "dual_momentum": dual_momentum,
                    "prox": prox,
                }

                if rho_update_rule == "constant":
                    for rho_init in [1e-1, 1, 10]:
                        config_spe = config.copy()
                        config_spe.update(
                            {
                                "rho_init": rho_init,
                            }
                        )
                        config_name_spe = "_".join(
                            [
                                config_name,
                                f"ri_{rho_init}",
                            ]
                        )
                        configs[config_name_spe] = config_spe
                else:
                    for rho_update_ratio, rho_update_cooldown in product(
                        [5, 10], [1, 5]
                    ):
                        config_spe = config.copy()
                        config_spe.update(
                            {
                                "rho_update_ratio": rho_update_ratio,
                                "rho_update_cooldown": rho_update_cooldown,
                            }
                        )
                        config_name_spe = "_".join(
                            [
                                config_name,
                                f"rur_{rho_update_ratio}",
                                f"ruc_{rho_update_cooldown}",
                            ]
                        )
                        if rho_update_rule == "linear":
                            for rho_init, rho_lin_factor in product(
                                [1e-1, 1, 10], [2, 5, 10]
                            ):
                                config_spe_spe = config_spe.copy()
                                config_spe_spe.update(
                                    {
                                        "rho_init": rho_init,
                                        "rho_lin_factor": rho_lin_factor,
                                    }
                                )
                                config_name_spe_spe = "_".join(
                                    [
                                        config_name_spe,
                                        f"ri_{rho_init}",
                                        f"rlf_{rho_lin_factor}",
                                    ]
                                )
                                configs[config_name_spe_spe] = config_spe_spe
                        elif rho_update_rule == "osqp":
                            for rho_adaptive_fraction in [0.3, 0.5, 0.8]:
                                config_spe_spe = config_spe.copy()
                                config_spe_spe.update(
                                    {
                                        "rho_adaptive_fraction": rho_adaptive_fraction,
                                    }
                                )
                                config_name_spe_spe = "_".join(
                                    [
                                        config_name_spe,
                                        f"raf_{rho_adaptive_fraction}",
                                    ]
                                )
                                configs[config_name_spe_spe] = config_spe_spe
                        elif rho_update_rule == "spectral":
                            for rho_power, rho_power_factor in product(
                                [0.1, 0.3, 0.5], [0.05, 0.15, 0.3]
                            ):
                                config_spe_spe = config_spe.copy()
                                config_spe_spe.update(
                                    {
                                        "rho_power": rho_power,
                                        "rho_power_factor": rho_power_factor,
                                    }
                                )
                                config_name_spe_spe = "_".join(
                                    [
                                        config_name_spe,
                                        f"rp_{rho_power}",
                                        f"rpf_{rho_power_factor}",
                                    ]
                                )
                                configs[config_name_spe_spe] = config_spe_spe

        elif mode == "GS_best":
            for alpha, dual_momentum, rho_adaptive_fraction in product(
                [1.0, 1.1, 1.2], [0.0, 0.2, 0.4], [0.2, 0.3, 0.4]
            ):
                name_parts = ["ADMM"]
                name_parts.append("osqp")
                name_parts.append(f"alpha_{alpha}")
                name_parts.append(f"beta_{dual_momentum}")
                name_parts.append(f"gamma_{rho_adaptive_fraction}")

                config_name = "_".join(name_parts)
                config = {
                    "rho_update_rule": "osqp",
                    "alpha": alpha,
                    "dual_momentum": dual_momentum,
                    "prox": 1e-6,
                    "rho_update_ratio": 5,
                    "rho_update_cooldown": 1,
                    "rho_adaptive_fraction": rho_adaptive_fraction,
                }
                configs[config_name] = config

        for name, config in configs.items():
            config.update(self.STANDARD_ADMM_CONVERGENCE)
            config.update(
                {
                    "max_iterations": 2000,
                    "min_residual_threshold": 1e-8,
                    "rho_clip": 1e6,
                    "prox": 1e-6,
                    "rho_update_ratio": 10.0,
                    "rho_update_cooldown": 5,
                    "verbose": False,
                }
            )

        return configs

    def run_single_benchmark(
        self,
        solver_type: str,
        config_name: str,
        config: Dict[str, Any],
        n_vertices: int,
        problem_id: int,
        test_point_id: int,
        test_point: np.ndarray,
        mu: float,
        vis: np.ndarray,
    ) -> BenchmarkResult:
        """Run a single benchmark test with error handling"""
        max_it = config["max_iterations"]
        try:
            poly = PolygonContactPatch(
                vis=vis,
                mu=mu,
                ker_precompute=self.ker_precompute,
                warmstart_strat="prev_linadjust",
                solver_tyep=solver_type,
                solver_kwargs=config,
            )

            history = []
            start_time = time.perf_counter()
            projected = poly.project_cone(test_point, strict=False, history=history)
            converged = len(history) < max_it
            elapsed = time.perf_counter() - start_time if converged else self.max_time

            iterations = len(history)
            final_obj = 0.0 if not history else float(history[-1][1])
        except Exception as e:
            warnings.warn(f"Error in {solver_type}/{config_name}: {str(e)[:100]}")
            elapsed = self.max_time
            iterations = max_it
            converged = False
            final_obj = float("inf")

        return BenchmarkResult(
            solver_type=solver_type,
            config_name=config_name,
            n_vertices=n_vertices,
            problem_id=problem_id,
            test_point_id=test_point_id,
            mu=mu,
            iterations=iterations,
            time_seconds=elapsed,
            converged=converged,
            final_objective=final_obj if np.isfinite(final_obj) else np.nan,
            hp_dict=config.copy(),
        )

    def run_benchmark_suite(
        self,
        pgd_modes: List[str] = ["baseline"],
        admm_modes: List[str] = ["baseline"],
        n_jobs: int = -1,
        verbose: int = 10,
    ):
        """Run benchmark with specified configuration modes using parallel execution

        Args:
            pgd_modes: List of PGD configuration modes to test
            admm_modes: List of ADMM configuration modes to test
            n_jobs: Number of parallel jobs (-1 uses all cores, 1 disables parallelism)
            verbose: Verbosity level for joblib progress (0=silent, 10=progress bar)
        """

        all_configs = {}

        for mode in pgd_modes:
            configs = self.get_pgd_configs(mode)
            for name, config in configs.items():
                all_configs[("PGD", name)] = config

        for mode in admm_modes:
            configs = self.get_admm_configs(mode)
            for name, config in configs.items():
                all_configs[("ADMM", name)] = config

        total_runs = len(all_configs) * self.n_problems * self.n_test_points

        print(f"\n{'=' * 70}")
        print("HYPERPARAMETER BENCHMARK SUITE")
        print(f"{'=' * 70}")
        print(
            f"Fixed PGD convergence: rel={self.STANDARD_PGD_CONVERGENCE['rel_crit']}, "
            f"abs={self.STANDARD_PGD_CONVERGENCE['abs_crit']}"
        )
        print(
            f"Fixed ADMM convergence: rel={self.STANDARD_ADMM_CONVERGENCE['rel_crit']}, "
            f"abs={self.STANDARD_ADMM_CONVERGENCE['abs_crit']}"
        )
        print(f"\nTotal configurations: {len(all_configs)}")
        print(f"Total test problems: {len(self.test_problems)}")
        print(f"Test points per problem: {self.n_test_points}")
        print(f"Total runs: {total_runs}")
        print(f"Parallel jobs: {n_jobs if n_jobs != -1 else 'all cores'}")
        print(f"{'=' * 70}\n")

        # Build list of all benchmark tasks
        benchmark_tasks = []
        for (solver_type, config_name), hyperparams in all_configs.items():
            for problem in self.test_problems:
                for point_id, test_point in enumerate(self.test_points):
                    benchmark_tasks.append(
                        {
                            "solver_type": solver_type,
                            "config_name": config_name,
                            "config": hyperparams,
                            "n_vertices": problem["n_vertices"],
                            "problem_id": problem["problem_id"],
                            "test_point_id": point_id,
                            "test_point": test_point,
                            "mu": problem["mu"],
                            "vis": problem["vis"],
                        }
                    )

        # Run benchmarks in parallel
        print(f"Running {len(benchmark_tasks)} benchmarks in parallel...")
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.run_single_benchmark)(**task) for task in benchmark_tasks
        )

        # Store results
        self.results.extend(results)

        print(f"\n{'=' * 70}")
        print(f"✓ Benchmark complete! {len(self.results)} results collected.")
        print(f"{'=' * 70}\n")

        return self

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = [result.to_dict() for result in self.results]
        df = pd.DataFrame(data)

        if "time_seconds" in df.columns:
            df["time_ms"] = df["time_seconds"] * 1000

        return df

    def analyze_solution_quality(self, df: pd.DataFrame = None):
        """Analyze actual solution quality achieved by each solver"""
        if df is None:
            df = self.get_results_dataframe()

        print("\n" + "=" * 70)
        print("SOLUTION QUALITY ANALYSIS")
        print("=" * 70)
        print("\nThis checks if both solvers achieve similar final accuracy,")
        print("validating that the convergence criteria comparison is fair.")
        print("-" * 70)

        for solver in df["solver_type"].unique():
            solver_df = df[df["solver_type"] == solver]
            converged = solver_df[solver_df["converged"]]

            if len(converged) == 0:
                print(f"\n{solver}: No converged cases!")
                continue

            print(f"\n{solver} (converged cases only, n={len(converged)}):")
            print("  Final Objective Statistics:")
            print(f"    Median:         {converged['final_objective'].median():.3e}")
            print(f"    Mean:           {converged['final_objective'].mean():.3e}")
            print(f"    Std:            {converged['final_objective'].std():.3e}")
            print(f"    Min:            {converged['final_objective'].min():.3e}")
            print(f"    Max:            {converged['final_objective'].max():.3e}")
            print(
                f"    95th percentile: {converged['final_objective'].quantile(0.95):.3e}"
            )

        # Compare solvers
        if len(df["solver_type"].unique()) > 1:
            print("\n" + "-" * 70)
            print("FAIRNESS CHECK:")
            medians = {}
            for solver in df["solver_type"].unique():
                converged = df[(df["solver_type"] == solver) & (df["converged"])]
                if len(converged) > 0:
                    medians[solver] = converged["final_objective"].median()

            if len(medians) == 2:
                solvers = list(medians.keys())
                ratio = medians[solvers[0]] / medians[solvers[1]]
                print(
                    f"  Median objective ratio ({solvers[0]}/{solvers[1]}): {ratio:.2f}"
                )

                if 0.1 < ratio < 10:
                    print("  ✓ Solvers achieve similar accuracy (ratio within 10×)")
                    print("  ✓ Convergence criteria comparison is FAIR")
                else:
                    print("  ⚠ Solvers achieve different accuracy (ratio > 10×)")
                    print("  ⚠ Consider adjusting convergence criteria")

        print("=" * 70)

    def diagnose_failures(self, df: pd.DataFrame = None):
        """Comprehensive failure diagnosis with detailed analysis"""
        if df is None:
            df = self.get_results_dataframe()

        print("\n" + "=" * 70)
        print("FAILURE DIAGNOSIS")
        print("=" * 70)

        failed = df[~df["converged"]]
        success = df[df["converged"]]

        total_failures = len(failed)
        total_runs = len(df)
        failure_rate = 100 * total_failures / total_runs

        print("\nOVERALL FAILURE STATISTICS:")
        print(f"  Total runs:     {total_runs}")
        print(f"  Failures:       {total_failures}")
        print(f"  Failure rate:   {failure_rate:.2f}%")
        print(f"  Success rate:   {100 - failure_rate:.2f}%")

        if total_failures == 0:
            print("\n✓ No failures detected!")
            return None

        # 1. By solver type
        print("\n" + "-" * 70)
        print("FAILURES BY SOLVER TYPE:")
        for solver in df["solver_type"].unique():
            solver_failed = len(failed[failed["solver_type"] == solver])
            solver_total = len(df[df["solver_type"] == solver])
            solver_rate = 100 * solver_failed / solver_total
            print(
                f"  {solver:8s}: {solver_failed:4d} / {solver_total:4d} ({solver_rate:5.2f}%)"
            )

        # Check if failure rate is similar across solvers
        solver_rates = []
        for solver in df["solver_type"].unique():
            solver_failed = len(failed[failed["solver_type"] == solver])
            solver_total = len(df[df["solver_type"] == solver])
            solver_rates.append(100 * solver_failed / solver_total)

        if len(solver_rates) > 1:
            rate_std = np.std(solver_rates)
            if rate_std < 2.0:
                print(
                    f"\n  → Failure rates are similar across solvers (std={rate_std:.2f}%)"
                )
                print("  → Failures are likely PROBLEM-DEPENDENT, not solver-dependent")
            else:
                print(f"\n  → Failure rates vary across solvers (std={rate_std:.2f}%)")
                print("  → Some solvers may be more robust than others")

        # 2. By configuration
        print("\n" + "-" * 70)
        print("FAILURES BY CONFIGURATION (worst 5):")
        config_failures = failed.groupby("config_name").size()
        config_totals = df.groupby("config_name").size()
        config_rates = (config_failures / config_totals * 100).sort_values(
            ascending=False
        )

        for config, rate in config_rates.head(min(5, len(config_failures))).items():
            count = config_failures[config]
            total = config_totals[config]
            print(f"  {config:40s}: {count:3d} / {total:3d} ({rate:5.2f}%)")

        # 3. By problem size
        print("\n" + "-" * 70)
        print("FAILURES BY PROBLEM SIZE:")
        for n in sorted(df["n_vertices"].unique()):
            n_failed = len(failed[failed["n_vertices"] == n])
            n_total = len(df[df["n_vertices"] == n])
            n_rate = 100 * n_failed / n_total
            print(f"  n_vertices={n:2d}: {n_failed:4d} / {n_total:4d} ({n_rate:5.2f}%)")

        # 4. By specific polygon
        print("\n" + "-" * 70)
        print("MOST PROBLEMATIC POLYGONS (top 10):")
        poly_failures = failed.groupby(["n_vertices", "problem_id"]).size()
        poly_totals = df.groupby(["n_vertices", "problem_id"]).size()
        poly_rates = (poly_failures / poly_totals * 100).sort_values(ascending=False)

        for (n, pb_id), rate in poly_rates.head(min(5, len(poly_failures))).items():
            count = poly_failures[(n, pb_id)]
            total = poly_totals[(n, pb_id)]
            print(
                f"  n_vertices={n:2d}, problem_id={pb_id:2d}: {count:3d} / {total:3d} ({rate:5.2f}%)"
            )

        # Check if failures are concentrated
        n_problematic_polygons = (poly_rates > 50).sum()
        total_polygons = len(poly_rates)
        if n_problematic_polygons > 0:
            print(
                f"\n  → {n_problematic_polygons} / {total_polygons} polygons have >50% failure rate"
            )
            print("  → Failures are CONCENTRATED in specific degenerate geometries")
        else:
            print("\n  → Failures are DISTRIBUTED across polygons")
            print("  → Not due to specific bad geometries")

        # 5. By specific points
        print("\n" + "-" * 70)
        print("MOST PROBLEMATIC POINTS (top 10):")
        point_failures = failed.groupby("test_point_id").size()
        point_totals = df.groupby("test_point_id").size()
        point_rates = (point_failures / point_totals * 100).sort_values(ascending=False)

        for point_id, rate in point_rates.head(min(5, len(point_failures))).items():
            count = point_failures[point_id]
            total = point_totals[point_id]
            print(f"  point_id={point_id:2d}: {count:3d} / {total:3d} ({rate:5.2f}%)")

        # Check if failures are concentrated
        n_problematic_points = (point_rates > 50).sum()
        total_points = len(point_rates)
        if n_problematic_points > 0:
            print(
                f"\n  → {n_problematic_points} / {total_points} points have >50% failure rate"
            )
            print("  → Failures are CONCENTRATED in specific degenerate geometries")
        else:
            print("\n  → Failures are DISTRIBUTED across points")
            print("  → Not due to specific bad geometries")

        # 6. Iterations analysis
        print("\n" + "-" * 70)
        print("ITERATION ANALYSIS:")

        max_iter = (
            df["hp_max_iterations"].iloc[0]
            if "hp_max_iterations" in df.columns
            else self.max_iter
        )

        if total_failures > 0:
            print("\nFailed cases:")
            print(f"  Mean iterations:   {failed['iterations'].mean():.1f}")
            print(f"  Median iterations: {failed['iterations'].median():.1f}")
            print(f"  Max iterations:    {failed['iterations'].max()}")
            print(f"  Max allowed:       {max_iter}")

            # Check if hitting max iterations
            at_max = (failed["iterations"] == max_iter).sum()
            at_max_rate = 100 * at_max / total_failures

            print(
                f"\n  Cases at/near max iterations: {at_max} / {total_failures} ({at_max_rate:.1f}%)"
            )

            if at_max_rate > 80:
                print("\n  ⚠️  DIAGNOSIS: Most failures hit max iterations!")
                print(
                    f"  → Solution: Increase max_iterations to {max_iter * 2}-{max_iter * 3}"
                )
                print("  → The solver is making progress but needs more time")
            elif at_max_rate > 20:
                print("\n  ⚠️  DIAGNOSIS: Some failures hit max iterations")
                print("  → Mixed cause: both iteration limit and problem issues")
            else:
                print("\n  ℹ️  DIAGNOSIS: Failures NOT due to iteration limit")
                print("  → Failures occur early, likely due to:")
                print("     - Numerical issues (ill-conditioning)")
                print("     - Degenerate geometries")
                print("     - Infeasible subproblems")

        # 7. Objective values at failure
        print("\n" + "-" * 70)
        print("OBJECTIVE VALUES AT FAILURE:")

        failed_with_obj = failed[np.isfinite(failed["final_objective"])]
        if len(failed_with_obj) > 0:
            print(f"  Mean:   {failed_with_obj['final_objective'].mean():.3e}")
            print(f"  Median: {failed_with_obj['final_objective'].median():.3e}")
            print(f"  Min:    {failed_with_obj['final_objective'].min():.3e}")
            print(f"  Max:    {failed_with_obj['final_objective'].max():.3e}")

            # Compare to successful cases
            if len(success) > 0:
                success_median = success["final_objective"].median()
                failed_median = failed_with_obj["final_objective"].median()
                ratio = failed_median / success_median

                print(f"\n  Success median:  {success_median:.3e}")
                print(f"  Failed median:   {failed_median:.3e}")
                print(f"  Ratio:           {ratio:.2f}×")

                if ratio < 10:
                    print("\n  ℹ️  Failed cases are close to optimal!")
                    print("  → May just need slightly looser convergence criteria")
                else:
                    print("\n  ⚠️  Failed cases have much higher objective")
                    print("  → Solver is stuck far from optimum")

        print("\n" + "=" * 70)

        return failed

    def plot_failure_analysis(self, df: pd.DataFrame = None, save_prefix: str = None):
        """Generate visualizations for failure analysis"""
        if df is None:
            df = self.get_results_dataframe()

        failed = df[~df["converged"]]
        success = df[df["converged"]]

        if len(failed) == 0:
            print("No failures to visualize!")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Failure Analysis", fontsize=16, fontweight="bold")

        # 1. Failure rate by solver
        ax = axes[0, 0]
        failure_by_solver = []
        for solver in df["solver_type"].unique():
            rate = (
                100
                * len(failed[failed["solver_type"] == solver])
                / len(df[df["solver_type"] == solver])
            )
            failure_by_solver.append((solver, rate))

        solvers, rates = zip(*failure_by_solver)
        ax.bar(solvers, rates, color=["steelblue", "coral"])
        ax.set_ylabel("Failure Rate (%)")
        ax.set_title("Failure Rate by Solver")
        ax.set_ylim([0, max(rates) * 1.2 if max(rates) > 0 else 10])
        ax.grid(axis="y", alpha=0.3)

        # 2. Failure rate by problem size
        ax = axes[0, 1]
        n_verticess = sorted(df["n_vertices"].unique())
        failure_rates = []
        for n in n_verticess:
            rate = (
                100
                * len(failed[failed["n_vertices"] == n])
                / len(df[df["n_vertices"] == n])
            )
            failure_rates.append(rate)

        ax.plot(
            n_verticess, failure_rates, "o-", linewidth=2, markersize=8, color="crimson"
        )
        ax.set_xlabel("N_vertices")
        ax.set_ylabel("Failure Rate (%)")
        ax.set_title("Failure Rate vs Problem Size")
        ax.grid(True, alpha=0.3)

        # 3. Iterations: failed vs success
        ax = axes[0, 2]
        if len(success) > 0 and len(failed) > 0:
            data_to_plot = [success["iterations"].values, failed["iterations"].values]
            bp = ax.boxplot(
                data_to_plot, labels=["Success", "Failed"], patch_artist=True
            )
            bp["boxes"][0].set_facecolor("lightgreen")
            bp["boxes"][1].set_facecolor("lightcoral")
            ax.set_ylabel("Iterations")
            ax.set_title("Iterations: Success vs Failed")
            ax.grid(axis="y", alpha=0.3)

        # 4. Objective: failed vs success
        ax = axes[1, 0]
        if len(success) > 0 and len(failed) > 0:
            success_obj = success[np.isfinite(success["final_objective"])][
                "final_objective"
            ]
            failed_obj = failed[np.isfinite(failed["final_objective"])][
                "final_objective"
            ]

            if len(success_obj) > 0 and len(failed_obj) > 0:
                data_to_plot = [
                    np.log10(success_obj.values + 1e-15),
                    np.log10(failed_obj.values + 1e-15),
                ]
                bp = ax.boxplot(
                    data_to_plot, labels=["Success", "Failed"], patch_artist=True
                )
                bp["boxes"][0].set_facecolor("lightgreen")
                bp["boxes"][1].set_facecolor("lightcoral")
                ax.set_ylabel("log10(Final Objective)")
                ax.set_title("Final Objective: Success vs Failed")
                ax.grid(axis="y", alpha=0.3)

        # 6. Time distribution: failed vs success
        ax = axes[1, 2]
        if len(success) > 0 and len(failed) > 0:
            success_time = success[np.isfinite(success["time_ms"])]["time_ms"]
            failed_time = failed[np.isfinite(failed["time_ms"])]["time_ms"]

            if len(success_time) > 0 and len(failed_time) > 0:
                bins = np.linspace(0, max(success_time.max(), failed_time.max()), 30)
                ax.hist(
                    success_time, bins=bins, alpha=0.6, label="Success", color="green"
                )
                ax.hist(failed_time, bins=bins, alpha=0.6, label="Failed", color="red")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Count")
                ax.set_title("Time Distribution")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_prefix:
            filename = f"{save_prefix}_failure_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✓ Saved: {filename}")

        plt.show()

    def print_top_configurations(self, df: pd.DataFrame = None, top_n: int = 3):
        """Print top N configurations with full hyperparameters"""
        if df is None:
            df = self.get_results_dataframe()

        print("\n" + "=" * 70)
        print("TOP CONFIGURATIONS (by median time)")
        print("=" * 70)

        for solver in ["PGD", "ADMM"]:
            solver_df = df[df["solver_type"] == solver]

            if len(solver_df) == 0:
                continue

            # Group by config and calculate metrics
            config_stats = (
                solver_df.groupby("config_name")
                .agg(
                    {
                        "time_ms": "median",
                        "iterations": "mean",
                        "converged": lambda x: 100 * x.mean(),
                    }
                )
                .sort_values("time_ms")
            )

            print(f"\n{'=' * 70}")
            print(f"{solver} - TOP {top_n} CONFIGURATIONS")
            print(f"{'=' * 70}")

            for rank, (config_name, stats) in enumerate(
                config_stats.head(top_n).iterrows(), 1
            ):
                print(f"\n{'#' * 70}")
                print(f"RANK {rank}: {config_name}")
                print(f"{'#' * 70}")
                print(f"  Median Time:      {stats['time_ms']:.3f} ms")
                print(f"  Mean Iterations:  {stats['iterations']:.1f}")
                print(f"  Convergence Rate: {stats['converged']:.1f}%")
                print("\n  Full Hyperparameter Configuration:")
                print(f"  {'-' * 66}")

                # Get the hyperparameters for this config
                sample_result = solver_df[solver_df["config_name"] == config_name].iloc[
                    0
                ]
                hp_cols = [col for col in solver_df.columns if col.startswith("hp_")]

                config_dict = {}
                for col in hp_cols:
                    key = col.replace("hp_", "")
                    value = sample_result[col]
                    config_dict[key] = value

                # Pretty print the dict
                print("  {")
                for key, value in sorted(config_dict.items()):
                    if isinstance(value, float):
                        print(f"    '{key}': {value},")
                    elif isinstance(value, bool):
                        print(f"    '{key}': {value},")
                    else:
                        print(f"    '{key}': '{value}',")
                print("  }")

        print(f"\n{'=' * 70}\n")

    def analyze_by_config(self, df: pd.DataFrame = None):
        """Analyze results grouped by configuration"""
        if df is None:
            df = self.get_results_dataframe()

        print("\n" + "=" * 70)
        print("PERFORMANCE BY CONFIGURATION")
        print("=" * 70)

        grouped = (
            df.groupby(["solver_type", "config_name"])
            .agg(
                {
                    "iterations": ["mean", "std", "min", "max"],
                    "time_ms": ["mean", "std", "median"],
                    "converged": ["sum", "count"],
                    "final_objective": "mean",
                }
            )
            .round(3)
        )

        grouped[("converged", "rate_%")] = (
            100 * grouped[("converged", "sum")] / grouped[("converged", "count")]
        ).round(1)

        grouped = grouped.sort_values(("time_ms", "median"))

        print("\nSummary table (sorted by median time):")
        print(grouped.to_string())

        return grouped

    def plot_config_comparison(self, df: pd.DataFrame = None, save_prefix: str = None):
        """Generate detailed comparison plots"""
        if df is None:
            df = self.get_results_dataframe()

        for solver in df["solver_type"].unique():
            solver_df = df[df["solver_type"] == solver]

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f"{solver} Hyperparameter Configuration Comparison",
                fontsize=16,
                fontweight="bold",
            )

            # 1. Time by config
            ax = axes[0, 0]
            config_stats = (
                solver_df.groupby("config_name")["time_ms"].median().sort_values()
            )
            config_stats.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("Median Time (ms)")
            ax.set_title("Median Computation Time")
            ax.grid(axis="x", alpha=0.3)

            # 2. Iterations by config
            ax = axes[0, 1]
            config_stats = (
                solver_df.groupby("config_name")["iterations"].median().sort_values()
            )
            config_stats.plot(kind="barh", ax=ax, color="coral")
            ax.set_xlabel("Median Iterations")
            ax.set_title("Median Iterations")
            ax.grid(axis="x", alpha=0.3)

            # 3. Convergence rate
            ax = axes[1, 0]
            conv_rate = solver_df.groupby("config_name")["converged"].mean() * 100
            conv_rate = conv_rate.sort_values(ascending=False)
            conv_rate.plot(kind="barh", ax=ax, color="seagreen")
            ax.set_xlabel("Convergence Rate (%)")
            ax.set_title("Convergence Rate")
            ax.set_xlim([0, 105])
            ax.grid(axis="x", alpha=0.3)

            # 4. Time vs problem size
            ax = axes[1, 1]
            top_configs = (
                solver_df.groupby("config_name")["time_ms"].median().nsmallest(5).index
            )
            for config in top_configs:
                subset = solver_df[solver_df["config_name"] == config]
                grouped = subset.groupby("n_vertices")["time_ms"].median()
                ax.plot(grouped.index, grouped.values, "o-", label=config, linewidth=2)
            ax.set_xlabel("N_vertices")
            ax.set_ylabel("Median Time (ms)")
            ax.set_title("Time vs Problem Size (top 5 configs)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_prefix:
                filename = f"{save_prefix}_{solver}_configs.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"✓ Saved: {filename}")

            plt.show()

    def plot_hyperparameter_effects(
        self, df: pd.DataFrame = None, save_prefix: str = None
    ):
        """Plot effects of individual hyperparameters"""
        if df is None:
            df = self.get_results_dataframe()

        hp_cols = [col for col in df.columns if col.startswith("hp_")]

        for solver in df["solver_type"].unique():
            solver_df = df[df["solver_type"] == solver]

            varying_hps = []
            for col in hp_cols:
                if solver_df[col].nunique() > 1:
                    varying_hps.append(col)

            if len(varying_hps) == 0:
                continue

            n_plots = len(varying_hps)
            n_cols = min(2, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

            if not hasattr(axes, "__len__"):
                axes = np.array([axes])
            axes = axes.flatten()

            fig.suptitle(
                f"{solver} Hyperparameter Effects on Performance",
                fontsize=16,
                fontweight="bold",
            )

            for idx, hp_col in enumerate(varying_hps):
                ax = axes[idx]
                hp_name = hp_col.replace("hp_", "")

                hp_stats = solver_df.groupby(hp_col).agg(
                    {
                        "time_ms": "median",
                        "iterations": "median",
                    }
                )

                x_vals = hp_stats.index.astype(str)
                ax2 = ax.twinx()

                ax.plot(
                    x_vals,
                    hp_stats["time_ms"],
                    "o-",
                    color="steelblue",
                    linewidth=2,
                    markersize=8,
                    label="Time",
                )
                ax2.plot(
                    x_vals,
                    hp_stats["iterations"],
                    "s-",
                    color="coral",
                    linewidth=2,
                    markersize=8,
                    label="Iterations",
                )

                ax.set_xlabel(hp_name, fontsize=11, fontweight="bold")
                ax.set_ylabel("Median Time (ms)", color="steelblue", fontsize=10)
                ax2.set_ylabel("Median Iterations", color="coral", fontsize=10)
                ax.tick_params(axis="y", labelcolor="steelblue")
                ax2.tick_params(axis="y", labelcolor="coral")
                ax.grid(True, alpha=0.3)

                if len(x_vals) > 5:
                    ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            if save_prefix:
                filename = f"{save_prefix}_{solver}_hyperparams.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"✓ Saved: {filename}")

            plt.show()

    def extract_failure_examples(self, df: pd.DataFrame = None, n_examples: int = 4):
        """Extract actual polygon geometries and test points that caused failures"""
        if df is None:
            df = self.get_results_dataframe()

        failed = df[~df["converged"]]

        if len(failed) == 0:
            print("No failures to extract!")
            return []

        print("\n" + "=" * 70)
        print(f"EXTRACTING {n_examples} FAILURE CASE EXAMPLES")
        print("=" * 70)

        # Select diverse failure examples
        # Strategy: pick from different problem sizes and polygons
        examples = []

        # Group by n_vertices and polygon_id to get diversity
        failure_groups = failed.groupby(["n_vertices", "problem_id"])

        selected_groups = []
        for (n_vertices, pb_id), group in failure_groups:
            if len(group) > 0:
                selected_groups.append((n_vertices, pb_id, group))

        # Sort by failure count (most problematic first)
        selected_groups.sort(key=lambda x: len(x[2]), reverse=True)

        # Select n_examples from different groups
        for idx, (n_vertices, pb_id, group) in enumerate(selected_groups[:n_examples]):
            # Pick one failure case from this group
            failure_case = group.iloc[0]

            # Find the original test problem
            test_problem = None
            for problem in self.test_problems:
                if (
                    problem["n_vertices"] == n_vertices
                    and problem["problem_id"] == pb_id
                ):
                    test_problem = problem
                    break

            if test_problem is None:
                continue

            # Get the test point
            test_point_id = failure_case["test_point_id"]
            test_point = self.test_points[test_point_id]

            config = {
                key.replace("hp_", ""): value
                for key, value in failure_case.items()
                if key.startswith("hp_")
            }

            example = {
                "example_id": idx + 1,
                "n_vertices": n_vertices,
                "problem_id": pb_id,
                "test_point_id": test_point_id,
                "config_name": failure_case["config_name"],
                "iterations": failure_case["iterations"],
                "final_objective": failure_case["final_objective"],
                "vis": test_problem["vis"].copy(),
                "mu": test_problem["mu"],
                "ker_precompute": self.ker_precompute,
                "solver_type": failure_case["solver_type"],
                "solver_kwargs": config,
                "test_point": test_point.copy(),
            }

            examples.append(example)

            # Print example
            print(f"\n{'=' * 70}")
            print(f"FAILURE EXAMPLE #{idx + 1}")
            print(f"{'=' * 70}")
            print(f"Problem Size:     n_vertices = {n_vertices}")
            print(f"Polygon ID:       {pb_id}")
            print(f"Test Point ID:    {test_point_id}")
            print(f"Solver:           {failure_case['solver_type']}")
            print(f"Configuration:    {failure_case['config_name']}")
            print(f"Iterations:       {failure_case['iterations']}")
            print(f"Final Objective:  {failure_case['final_objective']:.6e}")
            print(f"\nPolygon vertices shape: {test_problem['vis'].shape}")
            print("Test point (force in R^6):")
            print(f"  {test_point}")
            print(f"  Norm: {np.linalg.norm(test_point):.6f}")

        print(f"\n{'=' * 70}")
        print(f"Extracted {len(examples)} failure examples")
        print(f"{'=' * 70}\n")

        return examples

    def save_failure_examples(
        self, examples: List[Dict], filename: str = "failure_examples.npz"
    ):
        """Save failure examples to file for later reproduction"""
        if len(examples) == 0:
            print("No examples to save!")
            return

        # Prepare data for saving
        save_dict = {}

        for ex in examples:
            prefix = f"example_{ex['example_id']}"
            for k, v in ex.items():
                save_dict[f"{prefix}_{k}"] = v

        np.savez(filename, **save_dict)
        print(f"✓ Saved {len(examples)} failure examples to: {filename}")

    def print_failure_examples_code(self, examples: List[Dict]):
        """Print Python code to reproduce failure cases"""
        if len(examples) == 0:
            print("No examples to print!")
            return

        print("\n" + "=" * 70)
        print("PYTHON CODE TO REPRODUCE FAILURE CASES")
        print("=" * 70)
        print("\n# Copy-paste this code to reproduce the failure cases:\n")

        print("import numpy as np")
        print("from your_module import PolygonContactPatch  # Adjust import as needed")
        print()

        for ex in examples:
            print(f"# {'=' * 66}")
            print(f"# FAILURE EXAMPLE #{ex['example_id']}")
            print(f"# {ex['solver_type']} - {ex['config_name']}")
            print(f"# n_vertices={ex['n_vertices']}, problem_id={ex['problem_id']}")
            print(f"# {'=' * 66}")
            print()

            # Pretty print the dict
            print(f"kwargs_{ex['example_id']} = ", "{")
            for key, value in sorted(ex["solver_kwargs"].items()):
                if isinstance(value, str):
                    print(f"    '{key}': '{value}',")
                else:
                    print(f"    '{key}': {value},")
            print("}")

            # Print vis array
            print(f"vis_{ex['example_id']} = np.array([")
            for v in ex["vis"]:
                print(f"    {list(v)},")
            print("])")
            print()

            # Print test point
            print(f"test_point_{ex['example_id']} = np.array({list(ex['test_point'])})")
            print()

            # Print test code
            print("# Test this case:")
            print(f"mu = {ex['mu']}")
            print(f"ker_precompute = {self.ker_precompute}")
            print(f"poly_{ex['example_id']} = PolygonContactPatch(")
            print(f"    vis=vis_{ex['example_id']},")
            print("    mu=mu,")
            print("    ker_precompute=ker_precompute,")
            print("    warmstart_strat=None,")
            print(f"    solver_tyep='{ex['solver_type']}',")
            print(f"    solver_kwargs=kwargs_{ex['example_id']}")
            print(")")
            print()
            print("# Try to project (should fail or take many iterations):")
            print(f"history_{ex['example_id']} = []")
            print(f"result_{ex['example_id']} = poly_{ex['example_id']}.project_cone(")
            print(f"    test_point_{ex['example_id']}, ")
            print(f"    history=history_{ex['example_id']}")
            print(")")
            print(
                f"print(f'Converged: {{len(history_{ex['example_id']}) < 5000}}, Iterations: {{len(history_{ex['example_id']})}}')"
            )
            print()
            print()

    def visualize_failure_geometry(self, examples: List[Dict], save_prefix: str = None):
        """Visualize the 2D polygon geometries that caused failures"""
        if len(examples) == 0:
            print("No examples to visualize!")
            return

        n_examples = len(examples)
        n_cols = min(2, n_examples)
        n_rows = (n_examples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

        # Handle single subplot case
        if not hasattr(axes, "__len__"):
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, ex in enumerate(examples):
            ax = axes[idx]

            vis = ex["vis"]
            test_point = ex["test_point"]

            # Extract x, y coordinates (assuming vis is N x 2)
            x_coords = vis[:, 0]
            y_coords = vis[:, 1]

            # Plot polygon
            # Close the polygon by adding first vertex at the end
            x_poly = np.append(x_coords, x_coords[0])
            y_poly = np.append(y_coords, y_coords[0])

            ax.fill(x_poly, y_poly, color="lightblue", alpha=0.3, label="Polygon")
            ax.plot(x_poly, y_poly, "b-", linewidth=2, label="Edges")

            # Plot vertices
            ax.scatter(
                x_coords,
                y_coords,
                c="blue",
                marker="o",
                s=100,
                zorder=5,
                label="Vertices",
            )

            # Number the vertices
            for i, (x, y) in enumerate(vis):
                ax.annotate(
                    f"{i}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

            # Plot centroid
            centroid = vis.mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                c="red",
                marker="x",
                s=200,
                linewidths=3,
                zorder=6,
                label="Centroid",
            )

            # Plot origin
            ax.scatter(
                0,
                0,
                c="black",
                marker="+",
                s=200,
                linewidths=3,
                zorder=6,
                label="Origin",
            )

            # Add title with info
            ax.set_title(
                f"Failure #{ex['example_id']}: n={ex['n_vertices']}, "
                f"problem_id={ex['problem_id']}\n"
                f"{ex['solver_type']} - {ex['iterations']} iter\n"
                f"||f||={np.linalg.norm(test_point[:3]):.2f}, "
                f"||τ||={np.linalg.norm(test_point[3:]):.2f}",
                fontsize=10,
            )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")

            # Set nice limits with some padding
            x_range = x_coords.max() - x_coords.min()
            y_range = y_coords.max() - y_coords.min()
            padding = 0.2 * max(x_range, y_range)

            ax.set_xlim(x_coords.min() - padding, x_coords.max() + padding)
            ax.set_ylim(y_coords.min() - padding, y_coords.max() + padding)

        # Hide unused subplots
        for idx in range(n_examples, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_prefix:
            filename = f"{save_prefix}_failure_geometries.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✓ Saved: {filename}")

        plt.show()

    # Update generate_full_report to include failure examples
    def generate_full_report(
        self, df: pd.DataFrame = None, save_prefix: str = "benchmark"
    ):
        """Generate complete analysis report with all visualizations and failure examples"""
        if df is None:
            df = self.get_results_dataframe()

        print("\n" + "=" * 70)
        print("GENERATING COMPLETE BENCHMARK REPORT")
        print("=" * 70)

        # 1. Solution quality analysis
        print("\n[1/8] Analyzing solution quality...")
        self.analyze_solution_quality(df)

        # 2. Performance by configuration
        print("\n[2/8] Analyzing performance by configuration...")
        grouped = self.analyze_by_config(df)

        # 3. Top configurations
        print("\n[3/8] Identifying top configurations...")
        self.print_top_configurations(df, top_n=3)

        # 4. Failure diagnosis
        print("\n[4/8] Diagnosing failures...")
        failed = self.diagnose_failures(df)

        # 5. Extract failure examples
        if failed is not None and len(failed) > 0:
            print("\n[5/8] Extracting failure examples...")
            examples = self.extract_failure_examples(df, n_examples=5)

            if len(examples) > 0:
                # Save examples
                self.save_failure_examples(
                    examples, filename=f"{save_prefix}_failure_examples.npz"
                )

                # Print reproduction code
                self.print_failure_examples_code(examples)

                # Visualize geometries
                self.visualize_failure_geometry(examples, save_prefix=save_prefix)
        else:
            print("\n[5/8] No failures to analyze - skipping failure examples")
            examples = []

        # 6. Generate comparison plots
        print("\n[6/8] Generating comparison plots...")
        self.plot_config_comparison(df, save_prefix=save_prefix)

        # 7. Generate hyperparameter plots
        print("\n[7/8] Generating hyperparameter effect plots...")
        self.plot_hyperparameter_effects(df, save_prefix=save_prefix)

        # 8. Generate failure plots
        if failed is not None and len(failed) > 0:
            print("\n[8/8] Generating failure analysis plots...")
            self.plot_failure_analysis(df, save_prefix=save_prefix)
        else:
            print("\n[8/8] Skipping failure plots (no failures)")

        # Save CSV
        csv_filename = f"{save_prefix}_results.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n✓ Results saved to: {csv_filename}")

        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE!")
        print("=" * 70)

        return examples


# ====================================================================================
# USAGE EXAMPLES
# ====================================================================================


def run_quick_test(
    n_vertice_min: int = 3,
    n_vertice_max: int = 5,
    mu_min: float = MU,
    mu_max: float = MU,
    n_problems: int = 10,
    n_test_points: int = 5,
    ker_precompute: bool = KER_PRECOMPUTE,
    max_time: float = MAX_TIME,
    max_iter: int = MAX_ITER,
    n_jobs: int = -1,
    verbose: int = 10,
):
    """Quick test (~2-3 minutes)"""
    print("Running quick test...")
    benchmark = OptimizationBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_test_points=n_test_points,
        ker_precompute=ker_precompute,
        max_time=max_time,
        max_iter=max_iter,
    )
    benchmark.run_benchmark_suite(
        pgd_modes=["baseline"], admm_modes=["baseline"], n_jobs=n_jobs, verbose=verbose
    )
    df = benchmark.get_results_dataframe()
    benchmark.generate_full_report(df, save_prefix="quick_test")
    return benchmark, df


def run_standard_benchmark(
    n_vertice_min: int = 3,
    n_vertice_max: int = 15,
    mu_min: float = 0.1,
    mu_max: float = 2.0,
    n_problems: int = 20,
    n_test_points: int = 10,
    ker_precompute: bool = KER_PRECOMPUTE,
    max_time: float = MAX_TIME,
    max_iter: int = MAX_ITER,
    n_jobs: int = -1,
    verbose: int = 10,
):
    """Standard hyperparameter benchmark (~20-30 minutes)"""
    benchmark = OptimizationBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_test_points=n_test_points,
        ker_precompute=ker_precompute,
        max_time=max_time,
        max_iter=max_iter,
    )

    benchmark.run_benchmark_suite(
        pgd_modes=["baseline", "alpha_sweep", "feature_combinations"],
        admm_modes=["baseline", "rho_init_sweep", "alpha_sweep", "momentum_sweep"],
        n_jobs=n_jobs,
        verbose=verbose,
    )

    df = benchmark.get_results_dataframe()
    benchmark.generate_full_report(df, save_prefix="standard")

    return benchmark, df


def run_extensive_benchmark(
    n_vertice_min: int = 3,
    n_vertice_max: int = 25,
    mu_min: float = 0.05,
    mu_max: float = 2.0,
    n_problems: int = 30,
    n_test_points: int = 20,
    ker_precompute: bool = KER_PRECOMPUTE,
    max_time: float = MAX_TIME,
    max_iter: int = MAX_ITER,
    n_jobs: int = -1,
    verbose: int = 10,
):
    """Extensive hyperparameter sweep (~2-4 hours)"""
    benchmark = OptimizationBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_test_points=n_test_points,
        ker_precompute=ker_precompute,
        max_time=max_time,
        max_iter=max_iter,
    )

    benchmark.run_benchmark_suite(
        pgd_modes=[
            "baseline",
            "alpha_sweep",
            "armijo_sigma_sweep",
            "armijo_beta_sweep",
            "feature_combinations",
        ],
        admm_modes=[
            "baseline",
            "rho_init_sweep",
            "alpha_sweep",
            "momentum_sweep",
            "rho_factor_sweep",
            "osqp_fraction_sweep",
            "combined_best",
        ],
        n_jobs=n_jobs,
        verbose=verbose,
    )

    df = benchmark.get_results_dataframe()
    benchmark.generate_full_report(df, save_prefix="extensive")

    return benchmark, df


def run_gridsearch_benchmark(
    n_vertice_min: int = 3,
    n_vertice_max: int = 25,
    mu_min: float = 0.05,
    mu_max: float = 2.0,
    n_problems: int = 15,
    n_test_points: int = 10,
    ker_precompute: bool = KER_PRECOMPUTE,
    max_time: float = MAX_TIME,
    max_iter: int = MAX_ITER,
    n_jobs: int = -1,
    verbose: int = 10,
):
    """Extensive hyperparameter sweep (~2-4 hours)"""
    benchmark = OptimizationBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_test_points=n_test_points,
        ker_precompute=ker_precompute,
        max_time=max_time,
        max_iter=max_iter,
    )

    benchmark.run_benchmark_suite(
        pgd_modes=["maxi_GS"], admm_modes=["maxi_GS"], n_jobs=n_jobs, verbose=verbose
    )

    df = benchmark.get_results_dataframe()
    benchmark.generate_full_report(df, save_prefix="gridsearch")

    return benchmark, df


def run_gsbest_benchmark(
    n_vertice_min: int = 3,
    n_vertice_max: int = 25,
    mu_min: float = 0.05,
    mu_max: float = 2.0,
    n_problems: int = 50,
    n_test_points: int = 50,
    ker_precompute: bool = KER_PRECOMPUTE,
    max_time: float = MAX_TIME,
    max_iter: int = MAX_ITER,
    n_jobs: int = -1,
    verbose: int = 10,
):
    benchmark = OptimizationBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_test_points=n_test_points,
        ker_precompute=ker_precompute,
        max_time=max_time,
        max_iter=max_iter,
    )

    benchmark.run_benchmark_suite(
        pgd_modes=["GS_best"], admm_modes=["GS_best"], n_jobs=n_jobs, verbose=verbose
    )

    df = benchmark.get_results_dataframe()
    benchmark.generate_full_report(df, save_prefix="gsbest")

    return benchmark, df


# ====================================================================================
# COMMAND LINE INTERFACE
# ====================================================================================


def main():
    """Command-line interface for running benchmarks"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run solver benchmarks with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test with default parameters
  python solvers_bench.py --benchmark quick

  # Run standard benchmark with custom mu range and max_iter
  python solvers_bench.py --benchmark standard --mu-min 0.1 --mu-max 1.5 --max-iter 5000

  # Run extensive benchmark with precomputed kernel and custom vertex range
  python solvers_bench.py --benchmark extensive --ker-precompute --n-vertice-min 5 --n-vertice-max 30

  # Run gridsearch with custom test points and problems, using 4 parallel jobs
  python solvers_bench.py --benchmark gridsearch --n-test-points 20 --n-problems 25 --n-jobs 4

  # Run quick test without parallelism (single-threaded)
  python solvers_bench.py --benchmark quick --n-jobs 1
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["quick", "standard", "extensive", "gridsearch", "gsbest"],
        default="gridsearch",
        help="Which benchmark to run (default: gridsearch)",
    )

    # Parameter overrides
    parser.add_argument(
        "--n-vertice-min",
        type=int,
        default=None,
        help="Minimum number of vertices (default: varies by benchmark)",
    )

    parser.add_argument(
        "--n-vertice-max",
        type=int,
        default=None,
        help="Maximum number of vertices (default: varies by benchmark)",
    )

    parser.add_argument(
        "--mu-min",
        type=float,
        default=None,
        help="Minimum friction coefficient (default: varies by benchmark)",
    )

    parser.add_argument(
        "--mu-max",
        type=float,
        default=None,
        help="Maximum friction coefficient (default: varies by benchmark)",
    )

    parser.add_argument(
        "--n-problems",
        type=int,
        default=None,
        help="Number of problems to test (default: varies by benchmark)",
    )

    parser.add_argument(
        "--n-test-points",
        type=int,
        default=None,
        help="Number of test points (default: varies by benchmark)",
    )

    parser.add_argument(
        "--ker-precompute",
        action="store_true",
        default=KER_PRECOMPUTE,
        help=f"Precompute kernel (default: {KER_PRECOMPUTE})",
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=MAX_TIME,
        help=f"Maximum time per solver call in seconds (default: {MAX_TIME})",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
        help=f"Maximum iterations per solver call (default: {MAX_ITER})",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for benchmark execution (-1 uses all cores, 1 disables parallelism, default: -1)",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=10,
        help="Verbosity level for parallel execution (0=silent, 10=progress bar, default: 10)",
    )

    args = parser.parse_args()

    # Prepare common kwargs - only include parameters that were explicitly provided
    kwargs = {
        "ker_precompute": args.ker_precompute,
        "max_time": args.max_time,
        "max_iter": args.max_iter,
        "n_jobs": args.n_jobs,
        "verbose": args.verbose,
    }

    # Add optional parameters if provided
    if args.n_vertice_min is not None:
        kwargs["n_vertice_min"] = args.n_vertice_min
    if args.n_vertice_max is not None:
        kwargs["n_vertice_max"] = args.n_vertice_max
    if args.mu_min is not None:
        kwargs["mu_min"] = args.mu_min
    if args.mu_max is not None:
        kwargs["mu_max"] = args.mu_max
    if args.n_problems is not None:
        kwargs["n_problems"] = args.n_problems
    if args.n_test_points is not None:
        kwargs["n_test_points"] = args.n_test_points

    # Run selected benchmark
    print(f"\n{'=' * 70}")
    print(f"Running {args.benchmark.upper()} benchmark")
    print(f"{'=' * 70}")
    print("Parameters:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 70}\n")

    if args.benchmark == "quick":
        benchmark, df = run_quick_test(**kwargs)
    elif args.benchmark == "standard":
        benchmark, df = run_standard_benchmark(**kwargs)
    elif args.benchmark == "extensive":
        benchmark, df = run_extensive_benchmark(**kwargs)
    elif args.benchmark == "gridsearch":
        benchmark, df = run_gridsearch_benchmark(**kwargs)
    elif args.benchmark == "gsbest":
        benchmark, df = run_gsbest_benchmark(**kwargs)

    return benchmark, df


# ====================================================================================
# RUN IT
# ====================================================================================

if __name__ == "__main__":
    benchmark, df = main()
