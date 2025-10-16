import numpy as np
import pandas as pd
import time
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from contactpatch.patches import PolygonContactPatch

@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run"""
    solver_type: str
    n_sample: int
    polygon_id: int
    test_point_id: int
    hyperparams: Dict[str, Any]
    iterations: int
    time_seconds: float
    converged: bool
    final_objective: float
    
class OptimizationBenchmark:
    """Benchmark suite for PGD and ADMM solvers"""
    
    def __init__(self, mu: float = 2.0, n_test_points: int = 20):
        self.mu = mu
        self.n_test_points = n_test_points
        self.results = []
        
    def generate_test_problems(self, n_samples_list: List[int], n_polygons_per_size: int = 5):
        """Generate test polygons and points"""
        self.test_problems = []
        
        for n_sample in n_samples_list:
            for poly_id in range(n_polygons_per_size):
                # Generate polygon
                vis = PolygonContactPatch.generate_polygon_vis(
                    N_sample=3*n_sample, 
                    aimed_n=n_sample
                )
                
                # Generate test points
                test_points = []
                for point_id in range(self.n_test_points):
                    # Generate random point in R^6
                    fl = np.random.randn(6)
                    fl = fl / np.linalg.norm(fl) * np.random.uniform(0.5, 5.0)
                    test_points.append(fl)
                
                self.test_problems.append({
                    'n_sample': n_sample,
                    'poly_id': poly_id,
                    'vis': vis,
                    'test_points': test_points
                })
        
        print(f"Generated {len(self.test_problems)} test problems")
        return self
    
    def get_pgd_hyperparameter_grid(self, level: str = 'default'):
        """Define PGD hyperparameter search grid"""
        if level == 'minimal':
            return [{
                'accel': True,
                'precond': True,
                'adaptive_restart': True,
                'armijo': False,
                'rel_crit': 1e-5,
                'abs_crit': 1e-7,
                'alpha_cond': 0.99,
            }]
        
        elif level == 'default':
            return [
                # Baseline: FISTA + Precond
                {
                    'accel': True,
                    'precond': True,
                    'adaptive_restart': False,
                    'armijo': False,
                    'rel_crit': 1e-5,
                    'abs_crit': 1e-7,
                    'alpha_cond': 0.99,
                },
                # + Adaptive Restart
                {
                    'accel': True,
                    'precond': True,
                    'adaptive_restart': True,
                    'armijo': False,
                    'rel_crit': 1e-5,
                    'abs_crit': 1e-7,
                    'alpha_cond': 0.99,
                },
                # + Armijo
                {
                    'accel': True,
                    'precond': True,
                    'adaptive_restart': True,
                    'armijo': True,
                    'armijo_iter': 20,
                    'armijo_sigma': 0.1,
                    'armijo_beta': 0.5,
                    'rel_crit': 1e-5,
                    'abs_crit': 1e-7,
                    'alpha_cond': 1.0,
                },
                # Without preconditioning
                {
                    'accel': True,
                    'precond': False,
                    'adaptive_restart': True,
                    'armijo': False,
                    'rel_crit': 1e-5,
                    'abs_crit': 1e-7,
                    'alpha_free': 0.99,
                },
            ]
        
        elif level == 'extensive':
            grid = []
            for accel, precond, restart, armijo in product(
                [True, False],  # accel
                [True, False],  # precond
                [True, False],  # restart
                [True, False]   # armijo
            ):
                if not precond and not accel:
                    continue  # Skip vanilla PG without anything
                
                config = {
                    'accel': accel,
                    'precond': precond,
                    'adaptive_restart': restart if accel else False,
                    'armijo': armijo,
                    'rel_crit': 1e-5,
                    'abs_crit': 1e-7,
                }
                
                if precond:
                    config['alpha_cond'] = 0.99 if not armijo else 1.0
                else:
                    config['alpha_free'] = 0.99
                
                if armijo:
                    config.update({
                        'armijo_iter': 20,
                        'armijo_sigma': 0.1,
                        'armijo_beta': 0.5,
                    })
                
                grid.append(config)
            
            return grid
    
    def get_admm_hyperparameter_grid(self, level: str = 'default'):
        """Define ADMM hyperparameter search grid"""
        if level == 'minimal':
            return [{
                'rho_update_rule': 'constant',
                'alpha': 1.0,
                'dual_momentum': 0.0,
                'rho_init': 0.1,
                'rel_crit': 1e-4,
                'abs_crit': 1e-5,
            }]
        
        elif level == 'default':
            return [
                # Baseline: Constant rho
                {
                    'rho_update_rule': 'constant',
                    'alpha': 1.0,
                    'dual_momentum': 0.0,
                    'rho_init': 0.1,
                    'rel_crit': 1e-4,
                    'abs_crit': 1e-5,
                },
                # Linear update
                {
                    'rho_update_rule': 'linear',
                    'alpha': 1.0,
                    'dual_momentum': 0.0,
                    'rho_init': 0.1,
                    'rho_lin_factor': 2.0,
                    'rel_crit': 1e-4,
                    'abs_crit': 1e-5,
                },
                # OSQP-style
                {
                    'rho_update_rule': 'osqp',
                    'alpha': 1.0,
                    'dual_momentum': 0.0,
                    'rho_init': 0.1,
                    'rho_adaptive_fraction': 0.4,
                    'rel_crit': 1e-4,
                    'abs_crit': 1e-5,
                },
                # With over-relaxation
                {
                    'rho_update_rule': 'linear',
                    'alpha': 1.5,
                    'dual_momentum': 0.0,
                    'rho_init': 0.1,
                    'rho_lin_factor': 2.0,
                    'rel_crit': 1e-4,
                    'abs_crit': 1e-5,
                },
                # With momentum
                {
                    'rho_update_rule': 'linear',
                    'alpha': 1.0,
                    'dual_momentum': 0.5,
                    'rho_init': 0.1,
                    'rho_lin_factor': 2.0,
                    'rel_crit': 1e-4,
                    'abs_crit': 1e-5,
                },
            ]
        
        elif level == 'extensive':
            grid = []
            for rho_rule, alpha, momentum in product(
                ['constant', 'linear', 'osqp'],
                [1.0, 1.2, 1.5],
                [0.0, 0.3, 0.7]
            ):
                config = {
                    'rho_update_rule': rho_rule,
                    'alpha': alpha,
                    'dual_momentum': momentum,
                    'rho_init': 0.1,
                    'rel_crit': 1e-4,
                    'abs_crit': 1e-5,
                }
                
                if rho_rule == 'linear':
                    config['rho_lin_factor'] = 2.0
                elif rho_rule == 'osqp':
                    config['rho_adaptive_fraction'] = 0.4
                
                grid.append(config)
            
            return grid
    
    def run_single_benchmark(
        self, 
        solver_type: str,
        vis: np.ndarray,
        n_sample: int,
        poly_id: int,
        test_point: np.ndarray,
        test_point_id: int,
        hyperparams: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a single benchmark test"""
        
        # Add common parameters
        full_hyperparams = {
            'max_iterations': 2000,
            'verbose': False,
            **hyperparams
        }
        
        # Create solver
        poly = PolygonContactPatch(
            vis=vis,
            mu=self.mu,
            ker_precompute=False,
            warmstart_strat=None,
            solver_tyep=solver_type,
            solver_kwargs=full_hyperparams
        )
        
        # Run projection with timing
        history = []
        start_time = time.perf_counter()
        try:
            projected = poly.project_cone(test_point, history=history)
            elapsed = time.perf_counter() - start_time
            
            # Extract results
            if len(history) > 0:
                last_entry = history[-1]
                if solver_type == 'PGD':
                    iterations = last_entry[0] + 1  # k + 1
                    final_obj = last_entry[1]
                else:  # ADMM
                    iterations = last_entry[0] + 1
                    final_obj = last_entry[1]
                converged = True
            else:
                iterations = 0
                final_obj = np.inf
                converged = False
                
        except Exception as e:
            print(f"Error in {solver_type}: {e}")
            elapsed = np.inf
            iterations = -1
            converged = False
            final_obj = np.inf
        
        return BenchmarkResult(
            solver_type=solver_type,
            n_sample=n_sample,
            polygon_id=poly_id,
            test_point_id=test_point_id,
            hyperparams=hyperparams,
            iterations=iterations,
            time_seconds=elapsed,
            converged=converged,
            final_objective=final_obj
        )
    
    def run_benchmark(
        self,
        solver_types: List[str] = ['PGD', 'ADMM'],
        grid_level: str = 'default'
    ):
        """Run complete benchmark suite"""
        
        total_runs = 0
        for solver_type in solver_types:
            if solver_type == 'PGD':
                grid = self.get_pgd_hyperparameter_grid(grid_level)
            else:
                grid = self.get_admm_hyperparameter_grid(grid_level)
            
            total_runs += len(grid) * len(self.test_problems) * self.n_test_points
        
        print(f"Starting benchmark: {total_runs} total runs")
        
        run_count = 0
        for solver_type in solver_types:
            print(f"\n{'='*70}")
            print(f"Benchmarking {solver_type}")
            print(f"{'='*70}")
            
            if solver_type == 'PGD':
                grid = self.get_pgd_hyperparameter_grid(grid_level)
            else:
                grid = self.get_admm_hyperparameter_grid(grid_level)
            
            for hyperparam_id, hyperparams in enumerate(grid):
                print(f"\nHyperparameter config {hyperparam_id + 1}/{len(grid)}")
                print(f"  Config: {hyperparams}")
                
                for problem in self.test_problems:
                    for point_id, test_point in enumerate(problem['test_points']):
                        result = self.run_single_benchmark(
                            solver_type=solver_type,
                            vis=problem['vis'],
                            n_sample=problem['n_sample'],
                            poly_id=problem['poly_id'],
                            test_point=test_point,
                            test_point_id=point_id,
                            hyperparams=hyperparams
                        )
                        
                        self.results.append(result)
                        run_count += 1
                        
                        if run_count % 50 == 0:
                            print(f"  Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")
        
        print(f"\nBenchmark complete! {len(self.results)} results collected.")
        return self
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []
        for result in self.results:
            row = {
                'solver': result.solver_type,
                'n_sample': result.n_sample,
                'polygon_id': result.polygon_id,
                'test_point_id': result.test_point_id,
                'iterations': result.iterations,
                'time_seconds': result.time_seconds,
                'time_ms': result.time_seconds * 1000,
                'converged': result.converged,
                'final_objective': result.final_objective,
            }
            
            # Add hyperparameters as separate columns
            for key, value in result.hyperparams.items():
                row[f'hp_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def analyze_results(self):
        """Generate analysis and visualizations"""
        df = self.get_results_dataframe()
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS ANALYSIS")
        print("="*70)
        
        # Overall statistics
        print("\n1. OVERALL STATISTICS")
        print("-" * 70)
        for solver in df['solver'].unique():
            solver_df = df[df['solver'] == solver]
            print(f"\n{solver}:")
            print(f"  Total runs: {len(solver_df)}")
            print(f"  Converged: {solver_df['converged'].sum()} ({100*solver_df['converged'].mean():.1f}%)")
            print(f"  Avg iterations: {solver_df['iterations'].mean():.1f} ± {solver_df['iterations'].std():.1f}")
            print(f"  Avg time: {solver_df['time_ms'].mean():.2f} ± {solver_df['time_ms'].std():.2f} ms")
            print(f"  Median time: {solver_df['time_ms'].median():.2f} ms")
        
        # By problem size
        print("\n2. PERFORMANCE BY PROBLEM SIZE (N_SAMPLE)")
        print("-" * 70)
        for n_sample in sorted(df['n_sample'].unique()):
            print(f"\nN_sample = {n_sample}:")
            for solver in df['solver'].unique():
                subset = df[(df['solver'] == solver) & (df['n_sample'] == n_sample)]
                print(f"  {solver}: {subset['iterations'].mean():.1f} iter, "
                      f"{subset['time_ms'].mean():.2f} ms")
        
        # Best hyperparameters
        print("\n3. BEST HYPERPARAMETER CONFIGURATIONS")
        print("-" * 70)
        
        for solver in df['solver'].unique():
            solver_df = df[df['solver'] == solver]
            
            # Group by hyperparameters
            hp_cols = [col for col in solver_df.columns if col.startswith('hp_')]
            if len(hp_cols) > 0:
                grouped = solver_df.groupby(hp_cols).agg({
                    'iterations': ['mean', 'std'],
                    'time_ms': ['mean', 'std'],
                    'converged': 'mean'
                }).reset_index()
                
                # Sort by average time
                grouped = grouped.sort_values(('time_ms', 'mean'))
                
                print(f"\n{solver} - Top 3 configurations by speed:")
                for i, row in grouped.head(3).iterrows():
                    print(f"\n  Rank {i+1}:")
                    print(f"    Time: {row[('time_ms', 'mean')]:.2f} ± {row[('time_ms', 'std')]:.2f} ms")
                    print(f"    Iterations: {row[('iterations', 'mean')]:.1f} ± {row[('iterations', 'std')]:.1f}")
                    print(f"    Convergence rate: {row[('converged', 'mean')]*100:.1f}%")
                    print(f"    Config:")
                    for col in hp_cols:
                        print(f"      {col.replace('hp_', '')}: {row[col]}")
        
        return df
    
    def plot_results(self, df: pd.DataFrame = None, save_path: str = None):
        """Generate visualization plots"""
        if df is None:
            df = self.get_results_dataframe()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Optimization Solver Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Time distribution by solver
        ax = axes[0, 0]
        df.boxplot(column='time_ms', by='solver', ax=ax)
        ax.set_title('Computation Time Distribution')
        ax.set_xlabel('Solver')
        ax.set_ylabel('Time (ms)')
        plt.sca(ax)
        plt.xticks(rotation=0)
        
        # 2. Iterations distribution by solver
        ax = axes[0, 1]
        df.boxplot(column='iterations', by='solver', ax=ax)
        ax.set_title('Iterations Distribution')
        ax.set_xlabel('Solver')
        ax.set_ylabel('Iterations')
        plt.sca(ax)
        plt.xticks(rotation=0)
        
        # 3. Time vs n_sample
        ax = axes[0, 2]
        for solver in df['solver'].unique():
            subset = df[df['solver'] == solver]
            grouped = subset.groupby('n_sample')['time_ms'].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                       marker='o', label=solver, capsize=5)
        ax.set_title('Time vs Problem Size')
        ax.set_xlabel('N_sample')
        ax.set_ylabel('Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Iterations vs n_sample
        ax = axes[1, 0]
        for solver in df['solver'].unique():
            subset = df[df['solver'] == solver]
            grouped = subset.groupby('n_sample')['iterations'].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                       marker='o', label=solver, capsize=5)
        ax.set_title('Iterations vs Problem Size')
        ax.set_xlabel('N_sample')
        ax.set_ylabel('Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Convergence rate
        ax = axes[1, 1]
        convergence = df.groupby('solver')['converged'].mean() * 100
        convergence.plot(kind='bar', ax=ax, color=["#16476a", '#ff7f0e'])
        ax.set_title('Convergence Rate')
        ax.set_ylabel('Convergence Rate (%)')
        ax.set_xlabel('Solver')
        plt.sca(ax)
        plt.xticks(rotation=0)
        ax.grid(True, axis='y', alpha=0.3)
        
        # 6. Time vs Iterations scatter
        ax = axes[1, 2]
        for solver in df['solver'].unique():
            subset = df[df['solver'] == solver]
            ax.scatter(subset['iterations'], subset['time_ms'], 
                      alpha=0.5, label=solver, s=20)
        ax.set_title('Time vs Iterations')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
        
        return fig

# Usage example
def run_complete_benchmark():
    """Run the complete benchmark suite"""
    
    # Create benchmark
    benchmark = OptimizationBenchmark(mu=2.0, n_test_points=20)
    
    # Generate test problems
    benchmark.generate_test_problems(
        n_samples_list=[3, 5, 10],
        n_polygons_per_size=5
    )
    
    # Run benchmark (use 'minimal', 'default', or 'extensive')
    benchmark.run_benchmark(
        solver_types=['PGD', 'ADMM'],
        grid_level='default'  # Start with 'default', then try 'extensive'
    )
    
    # Analyze results
    df = benchmark.analyze_results()
    
    # Save results
    df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to: benchmark_results.csv")
    
    # Generate plots
    benchmark.plot_results(df, save_path='benchmark_plots.png')
    
    return benchmark, df

# Run it!
if __name__ == '__main__':
    benchmark, results_df = run_complete_benchmark()
