import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple
import warnings
from joblib import Parallel, delayed

from contactpatch.patches import PolygonContactPatch
from solvers_bench import OptimizationBenchmark, BenchmarkResult


@dataclass
class WarmstartBenchmarkResult:
    """Store results from a single warmstart benchmark run"""

    solver_type: str
    config_name: str
    warmstart_strategy: str
    problem_id: int
    sequence_id: int
    sequence_type: str
    point_in_sequence: int
    n_vertices: int
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


class WarmstartBenchmark(OptimizationBenchmark):
    """
    Benchmark suite for testing warmstart strategies with sequential points.

    Tests warmstart strategies by solving sequences of closely related problems,
    where warmstarting from previous solutions should provide speedup.
    """

    WARMSTART_STRATEGIES = [
        None,
        "prev",
        "prev_state",
        "prev_linadjust",
        "prev_linadjust_state",
    ]

    def __init__(
        self,
        n_vertice_min: int = 3,
        n_vertice_max: int = 25,
        mu_min: float = 0.01,
        mu_max: float = 2.0,
        n_problems: int = 15,
        n_sequences: int = 10,
        sequence_length: int = 10,
        sequence_types: List[str] = ["random_walk", "interpolation"],
        ker_precompute: bool = False,
        max_time: float = 0.05,
        max_iter: int = 2000,
    ):
        """
        Initialize warmstart benchmark.

        Args:
            n_vertice_min: Minimum polygon vertices
            n_vertice_max: Maximum polygon vertices
            mu_min: Minimum friction coefficient
            mu_max: Maximum friction coefficient
            n_problems: Number of polygons to generate
            n_sequences: Total number of point sequences to generate (types randomly sampled)
            sequence_length: Length of each sequence
            sequence_types: Available sequence types to randomly sample from
            ker_precompute: Whether to precompute kernel
            max_time: Maximum time per solve
            max_iter: Maximum iterations per solve
        """
        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.sequence_types = sequence_types

        # Initialize parent class with dummy test points (will be replaced)
        super().__init__(
            n_vertice_min=n_vertice_min,
            n_vertice_max=n_vertice_max,
            mu_min=mu_min,
            mu_max=mu_max,
            n_problems=n_problems,
            n_test_points=1,  # Dummy value, not used
            ker_precompute=ker_precompute,
            max_time=max_time,
            max_iter=max_iter,
        )

        self.warmstart_results = []

    def generate_test_problems(self):
        """Generate test polygons and point sequences"""
        print("Generating test polygons...")

        # Generate polygons using parent class logic
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

        print(f"✓ Generated {len(self.test_problems)} test polygons")

        # Generate point sequences - randomly sample sequence types
        print("Generating point sequences...")
        self.sequences = []

        for seq_id in range(self.n_sequences):
            # Randomly select a sequence type
            seq_type = np.random.choice(self.sequence_types)

            sequence = self._generate_sequence(seq_type, self.sequence_length)
            self.sequences.append(
                {
                    "sequence_id": seq_id,
                    "sequence_type": seq_type,
                    "points": sequence,
                }
            )

        print(f"✓ Generated {len(self.sequences)} point sequences")

        # Print distribution of sequence types
        type_counts = {}
        for seq in self.sequences:
            seq_type = seq["sequence_type"]
            type_counts[seq_type] = type_counts.get(seq_type, 0) + 1

        print(f"  Sequence type distribution:")
        for seq_type in sorted(type_counts.keys()):
            print(f"    {seq_type}: {type_counts[seq_type]}")
        print(f"  Points per sequence: {self.sequence_length}")
        print(
            f"  Total sequential solves: {len(self.test_problems) * len(self.sequences) * self.sequence_length}"
        )

        return self

    def _generate_sequence(self, seq_type: str, length: int) -> np.ndarray:
        """
        Generate a sequence of related points.

        Args:
            seq_type: Type of sequence ('random_walk', 'interpolation', 'spiral', 'perturbation')
            length: Length of sequence

        Returns:
            Array of shape (length, 6) containing wrench vectors
        """
        if seq_type == "random_walk":
            # Random walk in wrench space
            start_point = np.random.randn(6)
            start_point *= np.random.uniform(1.0, 10.0) / (
                np.linalg.norm(start_point) + 1e-10
            )

            sequence = np.zeros((length, 6))
            sequence[0] = start_point

            step_size = (
                np.linalg.norm(start_point) * 0.1
            )  # 10% of initial norm per step
            for i in range(1, length):
                step = np.random.randn(6)
                step = step / (np.linalg.norm(step) + 1e-10) * step_size
                sequence[i] = sequence[i - 1] + step

        elif seq_type == "interpolation":
            # Linear interpolation between two random points
            start_point = np.random.randn(6)
            start_point *= np.random.uniform(1.0, 10.0) / (
                np.linalg.norm(start_point) + 1e-10
            )

            end_point = np.random.randn(6)
            end_point *= np.random.uniform(1.0, 10.0) / (
                np.linalg.norm(end_point) + 1e-10
            )

            t = np.linspace(0, 1, length)
            sequence = np.outer(1 - t, start_point) + np.outer(t, end_point)

        elif seq_type == "spiral":
            # Spiral trajectory in moment-force space
            t = np.linspace(0, 4 * np.pi, length)
            radius = np.random.uniform(1.0, 5.0)

            # Moment components (spiral in xy plane)
            mx = radius * np.cos(t) * (1 + 0.1 * t / (4 * np.pi))
            my = radius * np.sin(t) * (1 + 0.1 * t / (4 * np.pi))
            mz = np.linspace(0, 1, length) * radius

            # Force components (varying with trajectory)
            fx = np.ones(length) * radius * 0.5
            fy = np.sin(t * 2) * radius * 0.3
            fz = np.cos(t * 2) * radius * 0.3

            sequence = np.column_stack([mx, my, mz, fx, fy, fz])

        elif seq_type == "perturbation":
            # Small perturbations around a fixed point
            center_point = np.random.randn(6)
            center_point *= np.random.uniform(1.0, 10.0) / (
                np.linalg.norm(center_point) + 1e-10
            )

            perturbation_size = np.linalg.norm(center_point) * 0.05  # 5% perturbations

            sequence = np.zeros((length, 6))
            for i in range(length):
                perturbation = np.random.randn(6)
                perturbation = (
                    perturbation
                    / (np.linalg.norm(perturbation) + 1e-10)
                    * perturbation_size
                )
                sequence[i] = center_point + perturbation

        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")

        return sequence

    def run_warmstart_benchmark_suite(
        self,
        pgd_modes: List[str] = ["baseline"],
        admm_modes: List[str] = ["baseline"],
        warmstart_strategies: List[str] = None,
        n_jobs: int = -1,
        verbose: int = 10,
    ):
        """
        Run warmstart benchmark with specified configurations.

        Args:
            pgd_modes: List of PGD configuration modes to test
            admm_modes: List of ADMM configuration modes to test
            warmstart_strategies: List of warmstart strategies to test (default: all)
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        if warmstart_strategies is None:
            warmstart_strategies = self.WARMSTART_STRATEGIES

        # Get solver configurations
        all_configs = {}
        for mode in pgd_modes:
            configs = self.get_pgd_configs(mode)
            for name, config in configs.items():
                all_configs[("PGD", name)] = config

        for mode in admm_modes:
            configs = self.get_admm_configs(mode)
            for name, config in configs.items():
                all_configs[("ADMM", name)] = config

        total_sequence_runs = len(self.test_problems) * len(self.sequences)
        total_points = total_sequence_runs * self.sequence_length
        total_configs = len(warmstart_strategies) * len(all_configs)

        print(f"\n{'=' * 70}")
        print("WARMSTART STRATEGY BENCHMARK SUITE")
        print(f"{'=' * 70}")
        print(f"Test polygons: {len(self.test_problems)}")
        print(f"Point sequences: {len(self.sequences)}")
        print(f"Points per sequence: {self.sequence_length}")
        print(f"Warmstart strategies: {len(warmstart_strategies)}")
        print(f"Solver configurations: {len(all_configs)}")
        print(f"\nTotal sequence runs: {total_sequence_runs * total_configs}")
        print(f"Total point projections: {total_points * total_configs}")
        print(f"Parallel jobs: {n_jobs if n_jobs != -1 else 'all cores'}")
        print(f"{'=' * 70}\n")

        # Build list of all benchmark tasks (one task per complete sequence)
        benchmark_tasks = []
        for (solver_type, config_name), solver_config in all_configs.items():
            for warmstart_strat in warmstart_strategies:
                for problem in self.test_problems:
                    for sequence in self.sequences:
                        benchmark_tasks.append(
                            {
                                "solver_type": solver_type,
                                "config_name": config_name,
                                "solver_config": solver_config,
                                "warmstart_strat": warmstart_strat,
                                "problem": problem,
                                "sequence": sequence,
                            }
                        )

        # Run benchmarks in parallel
        print(f"Running {len(benchmark_tasks)} sequence benchmarks in parallel...")
        results_lists = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self._run_sequence_benchmark)(**task) for task in benchmark_tasks
        )

        # Flatten results (each task returns a list of results)
        for results_list in results_lists:
            self.warmstart_results.extend(results_list)

        print(f"\n{'=' * 70}")
        print(
            f"✓ Warmstart benchmark complete! {len(self.warmstart_results)} results collected."
        )
        print(f"{'=' * 70}\n")

        return self

    def _run_sequence_benchmark(
        self,
        solver_type: str,
        config_name: str,
        solver_config: Dict[str, Any],
        warmstart_strat: str,
        problem: Dict[str, Any],
        sequence: Dict[str, Any],
    ) -> List[WarmstartBenchmarkResult]:
        """
        Run a benchmark on a complete sequence of points.

        Returns a list of results, one per point in the sequence.
        """
        results = []

        try:
            # Create patch with warmstart strategy
            poly = PolygonContactPatch(
                vis=problem["vis"],
                mu=problem["mu"],
                ker_precompute=self.ker_precompute,
                warmstart_strat=warmstart_strat,
                solver_tyep=solver_type,
                solver_kwargs=solver_config,
            )

            # Solve sequence
            points = sequence["points"]
            for point_idx, point in enumerate(points):
                history = []
                start_time = time.perf_counter()

                try:
                    projected = poly.project_cone(point, strict=False, history=history)
                    elapsed = time.perf_counter() - start_time

                    max_iter = solver_config["max_iterations"]
                    converged = len(history) < max_iter
                    iterations = len(history)
                    final_obj = 0.0 if not history else float(history[-1][1])

                except Exception as e:
                    warnings.warn(
                        f"Error in problem {problem['problem_id']}, "
                        f"sequence {sequence['sequence_id']}, "
                        f"point {point_idx}: {str(e)[:100]}"
                    )
                    elapsed = self.max_time
                    iterations = solver_config["max_iterations"]
                    converged = False
                    final_obj = float("inf")

                results.append(
                    WarmstartBenchmarkResult(
                        solver_type=solver_type,
                        config_name=config_name,
                        warmstart_strategy=warmstart_strat
                        if warmstart_strat is not None
                        else "none",
                        problem_id=problem["problem_id"],
                        sequence_id=sequence["sequence_id"],
                        sequence_type=sequence["sequence_type"],
                        point_in_sequence=point_idx,
                        n_vertices=problem["n_vertices"],
                        mu=problem["mu"],
                        iterations=iterations,
                        time_seconds=elapsed,
                        converged=converged,
                        final_objective=final_obj if np.isfinite(final_obj) else np.nan,
                        hp_dict=solver_config.copy(),
                    )
                )

        except Exception as e:
            warnings.warn(
                f"Error creating patch for {solver_type}/{config_name}: {str(e)[:100]}"
            )
            # Return failed results for all points in sequence
            for point_idx in range(len(sequence["points"])):
                results.append(
                    WarmstartBenchmarkResult(
                        solver_type=solver_type,
                        config_name=config_name,
                        warmstart_strategy=warmstart_strat
                        if warmstart_strat is not None
                        else "none",
                        problem_id=problem["problem_id"],
                        sequence_id=sequence["sequence_id"],
                        sequence_type=sequence["sequence_type"],
                        point_in_sequence=point_idx,
                        n_vertices=problem["n_vertices"],
                        mu=problem["mu"],
                        iterations=solver_config["max_iterations"],
                        time_seconds=self.max_time,
                        converged=False,
                        final_objective=np.nan,
                        hp_dict=solver_config.copy(),
                    )
                )

        return results

    def get_warmstart_results_dataframe(self) -> pd.DataFrame:
        """Convert warmstart results to pandas DataFrame"""
        data = [result.to_dict() for result in self.warmstart_results]
        df = pd.DataFrame(data)

        if "time_seconds" in df.columns:
            df["time_ms"] = df["time_seconds"] * 1000

        return df

    def analyze_warmstart_effectiveness(self, df: pd.DataFrame = None):
        """Analyze the effectiveness of different warmstart strategies"""
        if df is None:
            df = self.get_warmstart_results_dataframe()

        print("\n" + "=" * 70)
        print("WARMSTART STRATEGY EFFECTIVENESS ANALYSIS")
        print("=" * 70)

        # Group by warmstart strategy and compute aggregates
        warmstart_stats = (
            df.groupby("warmstart_strategy")
            .agg(
                {
                    "iterations": ["mean", "median", "std"],
                    "time_ms": ["mean", "median", "std"],
                    "converged": lambda x: 100 * x.mean(),
                }
            )
            .round(3)
        )

        # Compute speedup relative to no warmstart
        if "none" in df["warmstart_strategy"].values:
            baseline = df[df["warmstart_strategy"] == "none"]
            baseline_time = baseline["time_ms"].median()
            baseline_iter = baseline["iterations"].median()

            print("\n" + "=" * 70)
            print("SPEEDUP RELATIVE TO NO WARMSTART")
            print("=" * 70)
            print(f"Baseline (no warmstart):")
            print(f"  Median time: {baseline_time:.3f} ms")
            print(f"  Median iterations: {baseline_iter:.1f}")
            print()

            for strat in sorted(df["warmstart_strategy"].unique()):
                if strat == "none":
                    continue

                strat_df = df[df["warmstart_strategy"] == strat]
                strat_time = strat_df["time_ms"].median()
                strat_iter = strat_df["iterations"].median()

                time_speedup = baseline_time / strat_time if strat_time > 0 else np.inf
                iter_reduction = (
                    100 * (1 - strat_iter / baseline_iter) if baseline_iter > 0 else 0
                )

                print(f"{strat}:")
                print(
                    f"  Median time: {strat_time:.3f} ms ({time_speedup:.2f}x speedup)"
                )
                print(
                    f"  Median iterations: {strat_iter:.1f} ({iter_reduction:+.1f}% change)"
                )
                print()

        print("\n" + "=" * 70)
        print("DETAILED STATISTICS BY WARMSTART STRATEGY")
        print("=" * 70)
        print(warmstart_stats.to_string())
        print()

        return warmstart_stats

    def analyze_warmstart_by_config(self, df: pd.DataFrame = None):
        """Analyze warmstart effectiveness for each solver configuration"""
        if df is None:
            df = self.get_warmstart_results_dataframe()

        print("\n" + "=" * 70)
        print("WARMSTART EFFECTIVENESS BY SOLVER CONFIGURATION")
        print("=" * 70)

        # For each configuration, find the best warmstart strategy
        best_strategies = []

        for (solver_type, config_name), group in df.groupby(
            ["solver_type", "config_name"]
        ):
            warmstart_comparison = group.groupby("warmstart_strategy").agg(
                {
                    "time_ms": "median",
                    "iterations": "median",
                    "converged": lambda x: 100 * x.mean(),
                }
            )

            best_strat = warmstart_comparison["time_ms"].idxmin()
            best_time = warmstart_comparison.loc[best_strat, "time_ms"]

            # Get baseline (no warmstart)
            if "none" in warmstart_comparison.index:
                baseline_time = warmstart_comparison.loc["none", "time_ms"]
                speedup = baseline_time / best_time if best_time > 0 else np.inf
            else:
                speedup = np.nan

            best_strategies.append(
                {
                    "solver_type": solver_type,
                    "config_name": config_name,
                    "best_warmstart": best_strat,
                    "best_time_ms": best_time,
                    "speedup": speedup,
                }
            )

        best_df = pd.DataFrame(best_strategies).sort_values("speedup", ascending=False)

        print("\nBest warmstart strategy per configuration (top 10 by speedup):")
        print(best_df.head(10).to_string(index=False))

        # Find if there's a clear overall winner
        print("\n" + "=" * 70)
        print("OVERALL BEST WARMSTART STRATEGY")
        print("=" * 70)

        strategy_wins = best_df["best_warmstart"].value_counts()
        print("\nNumber of configurations where each strategy is best:")
        print(strategy_wins.to_string())

        if len(strategy_wins) > 0:
            winner = strategy_wins.idxmax()
            win_rate = 100 * strategy_wins[winner] / len(best_df)
            print(f"\nOverall winner: {winner} ({win_rate:.1f}% of configurations)")

            if win_rate > 50:
                print(
                    f"✓ Clear winner: {winner} is best for majority of configurations"
                )
            else:
                print(f"⚠ No clear winner: best strategy varies by configuration")

        return best_df

    def filter_top_configs(self, df: pd.DataFrame = None, top_n: int = 5):
        """
        Filter dataframe to keep only top N configurations per solver.

        Args:
            df: Full results dataframe
            top_n: Number of top configurations to keep per solver (based on median time with no warmstart)

        Returns:
            Filtered dataframe with only top configurations
        """
        if df is None:
            df = self.get_warmstart_results_dataframe()

        print("\n" + "=" * 70)
        print(f"FILTERING TOP {top_n} CONFIGURATIONS PER SOLVER")
        print("=" * 70)

        # For each solver, find top N configs based on performance without warmstart
        top_configs = []

        for solver_type in df["solver_type"].unique():
            solver_df = df[df["solver_type"] == solver_type]

            # Use only "none" warmstart to rank configs
            baseline_df = solver_df[solver_df["warmstart_strategy"] == "none"]

            if len(baseline_df) == 0:
                print(
                    f"\nWarning: No 'none' warmstart data for {solver_type}, using all data"
                )
                baseline_df = solver_df

            # Rank configs by median time
            config_performance = (
                baseline_df.groupby("config_name")
                .agg(
                    {
                        "time_ms": "median",
                        "iterations": "median",
                        "converged": lambda x: 100 * x.mean(),
                    }
                )
                .sort_values("time_ms")
            )

            top_config_names = config_performance.head(top_n).index.tolist()

            print(f"\n{solver_type} - Top {top_n} configurations:")
            for i, config_name in enumerate(top_config_names, 1):
                stats = config_performance.loc[config_name]
                print(f"  {i}. {config_name}")
                print(
                    f"     Time: {stats['time_ms']:.3f} ms, Iter: {stats['iterations']:.1f}, Conv: {stats['converged']:.1f}%"
                )

            top_configs.extend([(solver_type, cn) for cn in top_config_names])

        # Filter dataframe
        mask = df.apply(
            lambda row: (row["solver_type"], row["config_name"]) in top_configs, axis=1
        )
        filtered_df = df[mask].copy()

        print(f"\n{'=' * 70}")
        print(f"Filtered from {len(df)} to {len(filtered_df)} results")
        print(
            f"Kept {len(top_configs)} configurations out of {df['config_name'].nunique()}"
        )
        print(f"{'=' * 70}")

        return filtered_df

    def analyze_warmstart_by_position(self, df: pd.DataFrame = None):
        """Analyze how warmstart effectiveness varies with position in sequence"""
        if df is None:
            df = self.get_warmstart_results_dataframe()

        print("\n" + "=" * 70)
        print("WARMSTART EFFECTIVENESS VS POSITION IN SEQUENCE")
        print("=" * 70)

        # Group by warmstart strategy and position
        position_stats = df.groupby(["warmstart_strategy", "point_in_sequence"]).agg(
            {
                "iterations": "median",
                "time_ms": "median",
            }
        )

        print("\nMedian iterations by position (first 5 positions):")
        for strat in sorted(df["warmstart_strategy"].unique()):
            print(f"\n{strat}:")
            if strat in position_stats.index.get_level_values(0):
                strat_data = position_stats.loc[strat].head(5)
                print(strat_data["iterations"].to_string())

        return position_stats

    def plot_warmstart_comparison(
        self, df: pd.DataFrame = None, save_prefix: str = None
    ):
        """Generate comparison plots for warmstart strategies"""
        import matplotlib.pyplot as plt

        if df is None:
            df = self.get_warmstart_results_dataframe()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Warmstart Strategy Comparison", fontsize=16, fontweight="bold")

        # 1. Median time by warmstart strategy
        ax = axes[0, 0]
        time_stats = df.groupby("warmstart_strategy")["time_ms"].median().sort_values()
        time_stats.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Median Time (ms)")
        ax.set_title("Computation Time by Warmstart Strategy")
        ax.grid(axis="x", alpha=0.3)

        # 2. Median iterations by warmstart strategy
        ax = axes[0, 1]
        iter_stats = (
            df.groupby("warmstart_strategy")["iterations"].median().sort_values()
        )
        iter_stats.plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Median Iterations")
        ax.set_title("Iterations by Warmstart Strategy")
        ax.grid(axis="x", alpha=0.3)

        # 3. Convergence rate by warmstart strategy
        ax = axes[1, 0]
        conv_stats = df.groupby("warmstart_strategy")["converged"].mean() * 100
        conv_stats = conv_stats.sort_values(ascending=False)
        conv_stats.plot(kind="barh", ax=ax, color="seagreen")
        ax.set_xlabel("Convergence Rate (%)")
        ax.set_title("Convergence Rate by Warmstart Strategy")
        ax.set_xlim([0, 105])
        ax.grid(axis="x", alpha=0.3)

        # 4. Time evolution through sequence
        ax = axes[1, 1]
        for strat in sorted(df["warmstart_strategy"].unique()):
            strat_df = df[df["warmstart_strategy"] == strat]
            position_stats = strat_df.groupby("point_in_sequence")["time_ms"].median()
            ax.plot(
                position_stats.index,
                position_stats.values,
                "o-",
                label=strat,
                linewidth=2,
                markersize=4,
            )
        ax.set_xlabel("Position in Sequence")
        ax.set_ylabel("Median Time (ms)")
        ax.set_title("Time Evolution Through Sequence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_prefix:
            filename = f"{save_prefix}_warmstart_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✓ Saved: {filename}")

        plt.show()

    def generate_warmstart_report(
        self,
        df: pd.DataFrame = None,
        save_prefix: str = "warmstart_bench",
        top_n: int = 5,
    ):
        """
        Generate complete warmstart analysis report.

        Args:
            df: Results dataframe
            save_prefix: Prefix for saved files
            top_n: Number of top configurations to analyze in detail (per solver)
        """
        if df is None:
            df = self.get_warmstart_results_dataframe()

        print("\n" + "=" * 70)
        print("GENERATING WARMSTART BENCHMARK REPORT")
        print("=" * 70)

        # 1. Overall effectiveness (all configs)
        print("\n[1/7] Analyzing overall warmstart effectiveness (all configs)...")
        self.analyze_warmstart_effectiveness(df)

        # 2. By configuration (all configs)
        print("\n[2/7] Analyzing warmstart by configuration (all configs)...")
        best_df = self.analyze_warmstart_by_config(df)

        # 3. Filter to top configurations
        print(f"\n[3/7] Filtering to top {top_n} configurations per solver...")
        filtered_df = self.filter_top_configs(df, top_n=top_n)

        # 4. Re-analyze with top configs only
        print(f"\n[4/7] Re-analyzing with top {top_n} configs only...")
        print("\n" + "#" * 70)
        print(f"# ANALYSIS WITH TOP {top_n} CONFIGURATIONS ONLY")
        print("#" * 70)

        self.analyze_warmstart_effectiveness(filtered_df)
        best_df_filtered = self.analyze_warmstart_by_config(filtered_df)

        # 5. By position in sequence
        print(
            f"\n[5/7] Analyzing warmstart by sequence position (top {top_n} configs)..."
        )
        self.analyze_warmstart_by_position(filtered_df)

        # 6. Generate plots
        print(f"\n[6/7] Generating comparison plots...")
        self.plot_warmstart_comparison(df, save_prefix=f"{save_prefix}_all")
        self.plot_warmstart_comparison(
            filtered_df, save_prefix=f"{save_prefix}_top{top_n}"
        )

        # 7. Save results
        print("\n[7/7] Saving results...")

        # Save full results
        csv_filename = f"{save_prefix}_results_all.csv"
        df.to_csv(csv_filename, index=False)
        print(f"✓ All results saved to: {csv_filename}")

        # Save filtered results
        csv_filename_filtered = f"{save_prefix}_results_top{top_n}.csv"
        filtered_df.to_csv(csv_filename_filtered, index=False)
        print(f"✓ Top {top_n} configs results saved to: {csv_filename_filtered}")

        # Save best strategies (all)
        best_csv = f"{save_prefix}_best_strategies_all.csv"
        best_df.to_csv(best_csv, index=False)
        print(f"✓ Best strategies (all) saved to: {best_csv}")

        # Save best strategies (filtered)
        best_csv_filtered = f"{save_prefix}_best_strategies_top{top_n}.csv"
        best_df_filtered.to_csv(best_csv_filtered, index=False)
        print(f"✓ Best strategies (top {top_n}) saved to: {best_csv_filtered}")

        print("\n" + "=" * 70)
        print("WARMSTART REPORT GENERATION COMPLETE!")
        print("=" * 70)
        print("\nKey findings:")
        print(f"  - Analyzed {len(df)} total results across all configurations")
        print(f"  - Focused analysis on top {top_n} configurations per solver")
        print(f"  - Filtered analysis has {len(filtered_df)} results")

        return best_df, best_df_filtered, filtered_df


# ====================================================================================
# USAGE FUNCTIONS
# ====================================================================================


def run_quick_warmstart_test(
    n_vertice_min: int = 3,
    n_vertice_max: int = 10,
    mu_min: float = 0.5,
    mu_max: float = 1.5,
    n_problems: int = 3,
    n_sequences: int = 5,
    sequence_length: int = 8,
    sequence_types: List[str] = ["random_walk", "interpolation"],
    top_n: int = 3,
    n_jobs: int = -1,
    verbose: int = 10,
):
    """Quick warmstart test"""
    print("Running quick warmstart test...")

    benchmark = WarmstartBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        sequence_types=sequence_types,
    )

    benchmark.run_warmstart_benchmark_suite(
        pgd_modes=["baseline"],
        admm_modes=["baseline"],
        n_jobs=n_jobs,
        verbose=verbose,
    )

    df = benchmark.get_warmstart_results_dataframe()
    best_all, best_top, filtered_df = benchmark.generate_warmstart_report(
        df, save_prefix="quick_warmstart", top_n=top_n
    )

    return benchmark, df, best_all, best_top, filtered_df


def run_standard_warmstart_benchmark(
    n_vertice_min: int = 3,
    n_vertice_max: int = 20,
    mu_min: float = 0.1,
    mu_max: float = 2.0,
    n_problems: int = 10,
    n_sequences: int = 20,
    sequence_length: int = 15,
    sequence_types: List[str] = [
        "random_walk",
        "interpolation",
        "spiral",
        "perturbation",
    ],
    top_n: int = 5,
    n_jobs: int = -1,
    verbose: int = 10,
):
    """Standard warmstart benchmark"""
    print("Running standard warmstart benchmark...")

    benchmark = WarmstartBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        sequence_types=sequence_types,
    )

    benchmark.run_warmstart_benchmark_suite(
        pgd_modes=["baseline", "feature_combinations"],
        admm_modes=["baseline"],
        n_jobs=n_jobs,
        verbose=verbose,
    )

    df = benchmark.get_warmstart_results_dataframe()
    best_all, best_top, filtered_df = benchmark.generate_warmstart_report(
        df, save_prefix="standard_warmstart", top_n=top_n
    )

    return benchmark, df, best_all, best_top, filtered_df


def run_gsbest_warmstart_benchmark(
    n_vertice_min: int = 3,
    n_vertice_max: int = 25,
    mu_min: float = 0.05,
    mu_max: float = 2.0,
    n_problems: int = 30,
    n_sequences: int = 20,
    sequence_length: int = 15,
    sequence_types: List[str] = [
        "random_walk",
        "interpolation",
        "spiral",
        "perturbation",
    ],
    top_n: int = 5,
    n_jobs: int = -1,
    verbose: int = 10,
):
    """Standard warmstart benchmark"""
    print("Running standard warmstart benchmark...")

    benchmark = WarmstartBenchmark(
        n_vertice_min=n_vertice_min,
        n_vertice_max=n_vertice_max,
        mu_min=mu_min,
        mu_max=mu_max,
        n_problems=n_problems,
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        sequence_types=sequence_types,
    )

    benchmark.run_warmstart_benchmark_suite(
        pgd_modes=["GS_best"],
        admm_modes=["GS_best"],
        n_jobs=n_jobs,
        verbose=verbose,
    )

    df = benchmark.get_warmstart_results_dataframe()
    best_all, best_top, filtered_df = benchmark.generate_warmstart_report(
        df, save_prefix="gsbest_warmstart", top_n=top_n
    )

    return benchmark, df, best_all, best_top, filtered_df


# ====================================================================================
# COMMAND LINE INTERFACE
# ====================================================================================


def main():
    """Command-line interface for warmstart benchmarks"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run warmstart strategy benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["quick", "standard", "gsbest"],
        default="quick",
        help="Which benchmark to run (default: quick)",
    )

    parser.add_argument("--n-vertice-min", type=int, default=None)
    parser.add_argument("--n-vertice-max", type=int, default=None)
    parser.add_argument("--mu-min", type=float, default=None)
    parser.add_argument("--mu-max", type=float, default=None)
    parser.add_argument(
        "--n-problems", type=int, default=None, help="Number of polygons to generate"
    )
    parser.add_argument(
        "--n-sequences", type=int, default=None, help="Number of point sequences"
    )
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top configs to analyze in detail",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=10)

    args = parser.parse_args()

    kwargs = {
        "n_jobs": args.n_jobs,
        "verbose": args.verbose,
    }

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
    if args.n_sequences is not None:
        kwargs["n_sequences"] = args.n_sequences
    if args.sequence_length is not None:
        kwargs["sequence_length"] = args.sequence_length
    if args.top_n is not None:
        kwargs["top_n"] = args.top_n

    if args.benchmark == "quick":
        benchmark, df, best_all, best_top, filtered_df = run_quick_warmstart_test(
            **kwargs
        )
    elif args.benchmark == "standard":
        benchmark, df, best_all, best_top, filtered_df = (
            run_standard_warmstart_benchmark(**kwargs)
        )
    elif args.benchmark == "gsbest":
        benchmark, df, best_all, best_top, filtered_df = run_gsbest_warmstart_benchmark(
            **kwargs
        )

    return benchmark, df, best_all, best_top, filtered_df


if __name__ == "__main__":
    benchmark, df, best_all, best_top, filtered_df = main()
