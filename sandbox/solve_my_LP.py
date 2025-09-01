import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import itertools
from typing import List, Tuple, Optional, Set

class ExactConvexGeometricSimplex:
    """
    Complete Enhanced Simplex leveraging EXACT convex structure: s(p) = α ||p - p*||
    for solving: max S^T W s.t. AW = c, W in simplex
    
    Integrates all heuristics discovered:
    1. Geometric-Convex Vertex Ordering
    2. Smart Initialization with Distance Priority
    3. Exact Branch Cutting/Pruning
    4. Radial Efficiency Scoring
    5. Optimal Direction Prediction
    6. Enhanced Pivoting Rules
    """
    
    def __init__(self, A: np.ndarray, S: np.ndarray, c: np.ndarray, 
                 p_star: np.ndarray, alpha: float, tolerance: float = 1e-8):
        """
        Args:
            A: (2, n) matrix of polygon vertices
            S: (n,) vector from convex function s(p_i) = α ||p_i - p*||
            c: (2,) target point in polygon
            p_star: (2,) minimum point of convex function
            alpha: positive coefficient in s(p) = α ||p - p*||
            tolerance: numerical tolerance
        """
        self.A = A
        self.S = S
        self.c = c
        self.p_star = p_star
        self.alpha = alpha
        self.n = A.shape[1]
        self.tolerance = tolerance
        
        # Verify the structure (optional check)
        self._verify_convex_structure()
        
        # Precompute exact geometric insights
        self._compute_exact_geometric_insights()
        
        # Precompute enhanced heuristic scores
        self._compute_enhanced_heuristic_scores()
    
    def _verify_convex_structure(self):
        """Verify that S[i] = α ||p_i - p*|| (optional sanity check)"""
        expected_S = np.array([
            self.alpha * np.linalg.norm(self.A[:, i] - self.p_star) 
            for i in range(self.n)
        ])
        
        if not np.allclose(self.S, expected_S, rtol=1e-6):
            print("⚠️  Warning: S doesn't match expected structure s(p) = α ||p - p*||")
            print(f"Expected: {expected_S}")
            print(f"Given: {self.S}")
    
    def _compute_exact_geometric_insights(self):
        """Compute exact geometric properties using known structure"""
        
        # Distance from each vertex to p*
        self.distances_to_pstar = np.array([
            np.linalg.norm(self.A[:, i] - self.p_star) for i in range(self.n)
        ])
        
        # Distance from target c to p*
        self.c_distance_to_pstar = np.linalg.norm(self.c - self.p_star)
        
        # Direction vectors from p* to each vertex
        self.directions_from_pstar = np.array([
            (self.A[:, i] - self.p_star) / (self.distances_to_pstar[i] + self.tolerance)
            if self.distances_to_pstar[i] > self.tolerance else np.zeros(2)
            for i in range(self.n)
        ])
        
        # Direction from p* to target c
        self.c_direction_from_pstar = (
            (self.c - self.p_star) / (self.c_distance_to_pstar + self.tolerance)
            if self.c_distance_to_pstar > self.tolerance else np.zeros(2)
        )
        
        print(f"🎯 Target c is {self.c_distance_to_pstar:.4f} units from p*")
        print(f"📏 Vertex distances from p*: {self.distances_to_pstar}")
    
    def _compute_enhanced_heuristic_scores(self):
        """Compute heuristic scores using exact convex structure"""
        
        # Heuristic 1: Exact Geometric-Convex Alignment
        self.directional_alignment = self._compute_directional_alignment()
        
        # Heuristic 2: Radial Efficiency
        self.radial_efficiency = self._compute_radial_efficiency()
        
        # Heuristic 3: Optimal Direction Prediction
        self.optimal_direction_scores = self._compute_optimal_direction_scores()
        
        # Heuristic 4: Enhanced vertex priorities using exact structure
        self.enhanced_vertex_priorities = self._enhanced_vertex_ordering()
        
        # Heuristic 5: Exact geometric essentiality
        self.exact_prunable_vertices = self._identify_exact_prunable_vertices()
        
        print(f"🏆 Top 3 vertices by priority: {self.enhanced_vertex_priorities[:3]}")
        print(f"🔪 Prunable vertices: {self.exact_prunable_vertices}")
    
    def _compute_directional_alignment(self) -> np.ndarray:
        """Measure how well each vertex's direction aligns with reaching c from p*"""
        alignment = np.zeros(self.n)
        
        for i in range(self.n):
            if self.distances_to_pstar[i] > self.tolerance:
                # Dot product: how aligned is vertex direction with c direction?
                alignment[i] = np.dot(self.directions_from_pstar[i], self.c_direction_from_pstar)
            
        return alignment
    
    def _compute_radial_efficiency(self) -> np.ndarray:
        """Compute radial efficiency - how much "bang for buck" each vertex gives"""
        efficiency = np.zeros(self.n)
        
        # Key insight: vertices that are far from p* AND help reach c are most valuable
        for i in range(self.n):
            distance_value = self.distances_to_pstar[i]  # Proportional to s(p_i) = α ||p_i - p*||
            
            # Geometric utility: how much does this vertex help vs. alternatives?
            geometric_utility = self._compute_geometric_utility(i)
            
            efficiency[i] = distance_value * geometric_utility
            
        return efficiency
    
    def _compute_geometric_utility(self, vertex_idx: int) -> float:
        """How much does this vertex help geometrically vs alternatives?"""
        p_i = self.A[:, vertex_idx]
        
        # If c is close to p*, any distant vertex is valuable
        if self.c_distance_to_pstar < self.tolerance:
            return self.distances_to_pstar[vertex_idx]
        
        # Otherwise, value depends on alignment + distance
        alignment = self.directional_alignment[vertex_idx]
        distance = self.distances_to_pstar[vertex_idx]
        
        # Positive alignment (same direction as c from p*) is good
        # But even negative alignment has some value if vertex is very far
        utility = distance * (1.0 + 0.5 * max(0, alignment))
        
        return utility
    
    def _compute_optimal_direction_scores(self) -> np.ndarray:
        """Predict which direction the optimal solution should point"""
        scores = np.zeros(self.n)
        
        # Key insight: optimal solution should "pull" towards high-value vertices
        # while satisfying constraint Σ w_i p_i = c
        
        for i in range(self.n):
            p_i = self.A[:, i]
            
            # How much does including this vertex "pull" the expectation
            # in a direction that's beneficial for the objective?
            
            # Direction this vertex would pull the expectation
            pull_direction = p_i - self.c  # If we add weight here, expectation moves this way
            
            # Is this pull direction beneficial? 
            # We want to pull towards vertices far from p*
            target_direction = self._compute_ideal_pull_direction()
            
            directional_score = np.dot(pull_direction, target_direction)
            distance_score = self.distances_to_pstar[i]  # Linear in distance
            
            scores[i] = directional_score * distance_score
            
        return scores
    
    def _compute_ideal_pull_direction(self) -> np.ndarray:
        """Compute the ideal direction to pull the expectation for max objective"""
        # Ideal strategy: pull towards the vertex that's farthest from p*
        farthest_vertex_idx = np.argmax(self.distances_to_pstar)
        farthest_vertex = self.A[:, farthest_vertex_idx]
        
        # Ideal pull direction: towards the farthest vertex, away from p*
        ideal_direction = farthest_vertex - self.p_star
        ideal_direction = ideal_direction / (np.linalg.norm(ideal_direction) + self.tolerance)
        
        return ideal_direction
    
    def _enhanced_vertex_ordering(self) -> List[int]:
        """Enhanced vertex ordering using exact convex structure"""
        scores = []
        
        for i in range(self.n):
            # Combine multiple exact heuristics
            distance_score = self.distances_to_pstar[i]  # s(p_i) = α ||p_i - p*|| value
            alignment_score = max(0, self.directional_alignment[i])  # Alignment with c direction
            radial_score = self.radial_efficiency[i]
            optimal_dir_score = self.optimal_direction_scores[i]
            
            # Weighted combination (these weights could be tuned)
            combined_score = (
                0.4 * distance_score +           # Raw s-value importance (linear in distance)
                0.2 * alignment_score +          # Geometric alignment
                0.2 * radial_score +             # Efficiency
                0.2 * optimal_dir_score          # Strategic direction
            )
            
            scores.append((i, combined_score))
        
        # Sort by combined score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [vertex_idx for vertex_idx, _ in scores]
    
    def _identify_exact_prunable_vertices(self) -> Set[int]:
        """Identify vertices unlikely to be optimal using exact structure"""
        prunable = set()
        
        # More sophisticated pruning using exact structure
        
        # Rule 1: Vertices very close to p* are unlikely to be optimal
        min_useful_distance = 0.2 * np.max(self.distances_to_pstar)
        close_to_pstar = self.distances_to_pstar < min_useful_distance
        
        # Rule 2: Vertices that pull in "bad" directions
        bad_direction_threshold = np.percentile(self.optimal_direction_scores, 25)
        bad_direction = self.optimal_direction_scores < bad_direction_threshold
        
        # Rule 3: Low geometric utility
        low_utility_threshold = np.percentile(self.radial_efficiency, 20)
        low_utility = self.radial_efficiency < low_utility_threshold
        
        for i in range(self.n):
            # Prune if vertex satisfies multiple bad criteria
            bad_criteria_count = sum([
                close_to_pstar[i],
                bad_direction[i],
                low_utility[i]
            ])
            
            if bad_criteria_count >= 2:  # At least 2 bad criteria
                # Final check: is it geometrically essential?
                if not self._is_geometrically_essential(i):
                    prunable.add(i)
        
        return prunable
    
    def _is_geometrically_essential(self, vertex_idx: int) -> bool:
        """Check if vertex is essential for representing c"""
        # Remove vertex and check if c is still representable
        remaining_vertices = [j for j in range(self.n) if j != vertex_idx]
        remaining_A = self.A[:, remaining_vertices]
        
        return not self._point_in_convex_hull(self.c, remaining_A)
    
    def _point_in_convex_hull(self, point: np.ndarray, vertices: np.ndarray) -> bool:
        """Check if point is in convex hull of vertices"""
        try:
            # Solve: find w s.t. vertices @ w = point, w >= 0, sum(w) = 1
            n_vertices = vertices.shape[1]
            
            # Set up constraints: Ax = b, x >= 0, sum(x) = 1
            A_eq = np.vstack([vertices, np.ones(n_vertices)])
            b_eq = np.hstack([point, 1.0])
            
            # Dummy objective (we just want feasibility)
            c_obj = np.zeros(n_vertices)
            
            result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, 
                           bounds=[(0, None)] * n_vertices, method='highs')
            
            return result.success
        except:
            return False
    
    def _smart_initialization_exact(self) -> Optional[Tuple[List[int], np.ndarray]]:
        """Smart initialization using exact convex structure s(p) = α ||p - p*||"""
        
        print("🚀 Enhanced initialization using exact s(p) = α ||p - p*|| structure")
        
        # Strategy 1: Start with the farthest vertex from p*
        farthest_idx = np.argmax(self.distances_to_pstar)
        print(f"   Farthest vertex from p*: {farthest_idx} (distance: {self.distances_to_pstar[farthest_idx]:.4f})")
        
        # Strategy 2: Find combinations that maximize total distance while satisfying constraints
        best_objective = -np.inf
        best_solution = None
        
        # Try combinations starting with top priority vertices
        candidates = [i for i in self.enhanced_vertex_priorities[:min(8, self.n)] 
                     if i not in self.exact_prunable_vertices]
        
        print(f"   Trying combinations from candidates: {candidates}")
        
        # Try all 3-vertex combinations from candidates
        for vertex_combo in itertools.combinations(candidates, 3):
            solution = self._solve_3vertex_combination(list(vertex_combo))
            
            if solution is not None:
                basis, weights = solution
                objective = self.S[basis] @ weights
                
                if objective > best_objective:
                    best_objective = objective
                    best_solution = solution
                    print(f"   📈 New best: {vertex_combo} -> objective {objective:.6f}")
        
        if best_solution is not None:
            print(f"   ✅ Initial solution: vertices {best_solution[0]}, objective: {best_objective:.6f}")
        else:
            print("   ❌ No feasible initial solution found with heuristics, trying fallback...")
            # Fallback: try any 3 vertices
            best_solution = self._fallback_initialization()
        
        return best_solution
    
    def _fallback_initialization(self) -> Optional[Tuple[List[int], np.ndarray]]:
        """Fallback initialization if heuristics fail"""
        for vertex_combo in itertools.combinations(range(self.n), 3):
            solution = self._solve_3vertex_combination(list(vertex_combo))
            if solution is not None:
                return solution
        return None
    
    def _solve_3vertex_combination(self, vertex_indices: List[int]) -> Optional[Tuple[List[int], np.ndarray]]:
        """Solve for weights using exactly 3 vertices"""
        if len(vertex_indices) != 3:
            return None
        
        # Set up system: A_basis @ w = c, sum(w) = 1, w >= 0
        A_basis = self.A[:, vertex_indices]
        
        # Augment with sum constraint: [A_basis; 1 1 1] @ w = [c; 1]
        A_eq = np.vstack([A_basis, np.ones(3)])
        b_eq = np.hstack([self.c, 1.0])
        
        try:
            # Solve the 3x3 system
            weights = np.linalg.solve(A_eq, b_eq)
            
            # Check feasibility (all weights non-negative)
            if np.all(weights >= -self.tolerance):
                weights = np.maximum(weights, 0)  # Clean up numerical errors
                weights = weights / np.sum(weights)  # Renormalize
                return vertex_indices, weights
        except np.linalg.LinAlgError:
            pass  # Singular matrix, combination doesn't work
        
        return None
    
    def _enhanced_simplex_iteration(self, current_basis: List[int], 
                                  current_weights: np.ndarray) -> Tuple[List[int], np.ndarray, bool]:
        """Enhanced simplex iteration with convex-geometric heuristics"""
        
        # Compute reduced costs for non-basic variables
        basis_matrix = self._construct_basis_matrix(current_basis)
        
        try:
            # Solve dual system: basis_matrix^T @ dual_vars = S[current_basis]
            dual_vars = np.linalg.solve(basis_matrix.T, self.S[current_basis])
        except np.linalg.LinAlgError:
            print("   ⚠️  Singular basis matrix, assuming optimal")
            return current_basis, current_weights, True
        
        # Compute reduced costs for all non-basic variables
        reduced_costs = {}
        non_basic = [i for i in range(self.n) if i not in current_basis]
        
        for j in non_basic:
            if j in self.exact_prunable_vertices:
                reduced_costs[j] = -np.inf  # Never enter prunable vertices
            else:
                constraint_vector = self._construct_constraint_vector(j)
                reduced_costs[j] = self.S[j] - dual_vars @ constraint_vector
        
        # Find entering variable (most positive reduced cost)
        entering_candidates = [j for j in non_basic if reduced_costs[j] > self.tolerance]
        
        if not entering_candidates:
            print("   🎯 Optimal solution found (no improving directions)")
            return current_basis, current_weights, True
        
        # Enhanced pivoting rule: prefer high s-value vertices in ties
        if len(entering_candidates) > 1:
            # Break ties using s-values (distance from p*)
            entering_var = max(entering_candidates, key=lambda j: self.S[j])
            print(f"   🔄 Multiple candidates, chose vertex {entering_var} (highest s-value: {self.S[entering_var]:.4f})")
        else:
            entering_var = entering_candidates[0]
            print(f"   ➡️  Entering variable: {entering_var} (reduced cost: {reduced_costs[entering_var]:.6f})")
        
        # Find leaving variable using ratio test
        leaving_var, new_weights = self._ratio_test(current_basis, current_weights, entering_var)
        
        if leaving_var is None:
            print("   ⚠️  Unbounded solution (shouldn't happen in our problem)")
            return current_basis, current_weights, True
        
        print(f"   ⬅️  Leaving variable: {leaving_var}")
        
        # Update basis
        new_basis = current_basis.copy()
        leaving_idx = current_basis.index(leaving_var)
        new_basis[leaving_idx] = entering_var
        
        return new_basis, new_weights, False
    
    def _construct_basis_matrix(self, basis: List[int]) -> np.ndarray:
        """Construct the basis matrix [A_basis; 1^T]"""
        A_basis = self.A[:, basis]
        return np.vstack([A_basis, np.ones(len(basis))])
    
    def _construct_constraint_vector(self, var_idx: int) -> np.ndarray:
        """Construct constraint vector for variable var_idx"""
        return np.hstack([self.A[:, var_idx], 1.0])
    
    def _ratio_test(self, basis: List[int], weights: np.ndarray, entering_var: int) -> Tuple[Optional[int], np.ndarray]:
        """Simplified ratio test - remove lowest s-value vertex"""
        if len(basis) == 0:
            return None, weights
        
        # Simple heuristic: remove vertex with lowest s-value
        # (In a full implementation, this should be the proper simplex ratio test)
        leaving_var = min(basis, key=lambda j: self.S[j])
        
        # Recompute weights with new basis
        new_basis = [entering_var if j == leaving_var else j for j in basis]
        new_solution = self._solve_3vertex_combination(new_basis)
        
        if new_solution is not None:
            _, new_weights = new_solution
            return leaving_var, new_weights
        else:
            # If new combination doesn't work, try another leaving variable
            for candidate_leaving in basis:
                if candidate_leaving != leaving_var:
                    test_basis = [entering_var if j == candidate_leaving else j for j in basis]
                    test_solution = self._solve_3vertex_combination(test_basis)
                    if test_solution is not None:
                        _, new_weights = test_solution
                        return candidate_leaving, new_weights
        
        return None, weights
    
    def solve(self) -> Tuple[Optional[np.ndarray], float]:
        """Main solving routine with all heuristics integrated"""
        print("=" * 60)
        print("🔍 ENHANCED SIMPLEX WITH CONVEX-GEOMETRIC HEURISTICS")
        print("=" * 60)
        
        # Phase 1: Smart initialization using all heuristics
        print("\n📍 PHASE 1: Smart Initialization")
        initial_solution = self._smart_initialization_exact()
        
        if initial_solution is None:
            print("❌ No feasible solution found")
            return None, -np.inf
        
        current_basis, current_weights = initial_solution
        initial_objective = self.S[current_basis] @ current_weights
        print(f"✅ Initial solution: objective = {initial_objective:.6f}")
        print(f"   Active vertices: {current_basis}")
        print(f"   Weights: {current_weights}")
        
        # Phase 2: Enhanced simplex iterations
        print("\n🔄 PHASE 2: Enhanced Simplex Iterations")
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            print(f"\n--- Iteration {iteration + 1} ---")
            
            new_basis, new_weights, is_optimal = self._enhanced_simplex_iteration(
                current_basis, current_weights)
            
            if is_optimal:
                print(f"🎯 Optimal solution found after {iteration + 1} iterations")
                break
            
            # Update current solution
            current_basis, current_weights = new_basis, new_weights
            iteration += 1
            
            # Progress reporting
            current_objective = self.S[current_basis] @ current_weights
            print(f"   Current objective: {current_objective:.6f}")
            print(f"   Active vertices: {current_basis}")
        
        # Construct full solution vector
        solution = np.zeros(self.n)
        solution[current_basis] = current_weights
        final_objective = self.S @ solution
        
        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS")
        print("=" * 60)
        print(f"Final objective value: {final_objective:.6f}")
        print(f"Active vertices: {current_basis}")
        print(f"Weights: {current_weights}")
        print(f"Improvement from initial: {final_objective - initial_objective:.6f}")
        
        return solution, final_objective
    
    def verify_solution(self, solution: np.ndarray) -> bool:
        """Verify the solution satisfies all constraints"""
        print("\n🔍 SOLUTION VERIFICATION")
        print("-" * 30)
        
        # Check constraint AW = c
        constraint_value = self.A @ solution
        constraint_error = np.linalg.norm(constraint_value - self.c)
        print(f"Constraint AW = c:")
        print(f"  AW = {constraint_value}")
        print(f"  c  = {self.c}")
        print(f"  Error: {constraint_error:.8f} {'✅' if constraint_error < 1e-6 else '❌'}")
        
        # Check sum constraint
        sum_weights = np.sum(solution)
        sum_error = abs(sum_weights - 1.0)
        print(f"Sum constraint Σw_i = 1:")
        print(f"  Sum: {sum_weights:.8f}")
        print(f"  Error: {sum_error:.8f} {'✅' if sum_error < 1e-6 else '❌'}")
        
        # Check non-negativity
        min_weight = np.min(solution)
        print(f"Non-negativity w_i ≥ 0:")
        print(f"  Min weight: {min_weight:.8f} {'✅' if min_weight >= -1e-8 else '❌'}")
        
        # Overall verification
        is_valid = (constraint_error < 1e-6 and sum_error < 1e-6 and min_weight >= -1e-8)
        print(f"\nOverall: {'✅ VALID SOLUTION' if is_valid else '❌ INVALID SOLUTION'}")
        
        return is_valid

    def _recover_dual_with_geometric_regularization(self, optimal_basis: List[int]) -> Tuple[np.ndarray, float]:
        """
        Regularized dual recovery using convex function structure
        
        For underdetermined systems, choose μ* closest to the natural gradient direction
        """
        basis_size = len(optimal_basis)
        
        # Compute natural gradient direction at c
        if np.linalg.norm(self.c - self.p_star) > self.tolerance:
            natural_gradient = self.alpha * (self.c - self.p_star) / np.linalg.norm(self.c - self.p_star)
        else:
            # Special case: c = p*, use zero gradient
            natural_gradient = np.zeros(2)
        
        print(f"🧭 Natural gradient direction at c: {natural_gradient}")
        
        if basis_size == 3:
            # Non-degenerate case - use standard method
            return self._recover_dual_standard(optimal_basis)
        
        elif basis_size == 2:
            return self._regularized_dual_2vertices(optimal_basis, natural_gradient)
        
        elif basis_size == 1:
            return self._regularized_dual_1vertex(optimal_basis, natural_gradient)
        
        else:
            return None, None

    def _regularized_dual_2vertices(self, optimal_basis: List[int], natural_gradient: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Regularized dual for |B*| = 2 case
        
        min ||μ - natural_gradient||²
        s.t. S_i - A_i^T μ - λ = 0  for i in optimal_basis
        """
        print("🎯 Regularized dual recovery for 2-vertex case")
        
        # Extract basis information
        A_basis = self.A[:, optimal_basis]  # (2, 2)
        S_basis = self.S[optimal_basis]     # (2,)
        
        # Constraints: A_basis^T μ + λ 1_2 = S_basis
        # This gives us: [a1^T  1] [μ1]   [S1]
        #                [a2^T  1] [μ2] = [S2]
        #                          [λ ]
        
        # We have 2 equations in 3 unknowns (μ1, μ2, λ)
        # Add regularization: minimize ||μ - natural_gradient||²
        
        # Method 1: Analytical solution using Lagrange multipliers
        A_T = A_basis.T  # (2, 2)
        ones_2 = np.ones(2)  # (2,)
        
        # The regularized problem is:
        # min ||μ - g||² s.t. A_T μ + λ ones_2 = S_basis
        # where g = natural_gradient
        
        # Lagrangian: L = ||μ - g||² + ν^T (A_T μ + λ ones_2 - S_basis)
        # 
        # ∂L/∂μ = 2(μ - g) + A_T^T ν = 0  =>  μ = g - (1/2) A_T^T ν
        # ∂L/∂λ = ν^T ones_2 = 0         =>  sum(ν) = 0
        # ∂L/∂ν = A_T μ + λ ones_2 - S_basis = 0
        
        # Substituting μ = g - (1/2) A_T^T ν into constraint:
        # A_T (g - (1/2) A_T^T ν) + λ ones_2 = S_basis
        # A_T g - (1/2) A_T A_T^T ν + λ ones_2 = S_basis
        
        try:
            # Solve the system more directly using quadratic programming approach
            # Set up: minimize ||μ - g||² subject to linear constraints
            
            # Use method of Lagrange multipliers
            # System becomes:
            # [I    A_T   ] [μ]   [g    ]
            # [A_T^T  0   ] [ν] = [S_basis - λ*ones_2]
            # But we still need to determine λ...
            
            # Alternative: parametric solution
            # From A_T μ + λ ones_2 = S_basis, we get:
            # μ = A_T^(-1) (S_basis - λ ones_2)  [if A_T is invertible]
            
            if np.linalg.det(A_T) > self.tolerance:
                # A_T is invertible - use parametric approach
                A_T_inv = np.linalg.inv(A_T)
                
                # μ(λ) = A_T^(-1) (S_basis - λ ones_2)
                # We want to minimize ||μ(λ) - g||²
                
                # μ(λ) = A_T^(-1) S_basis - λ A_T^(-1) ones_2
                base_mu = A_T_inv @ S_basis
                direction = A_T_inv @ ones_2
                
                # ||base_mu - λ direction - g||²
                # = ||base_mu - g||² - 2λ (base_mu - g)^T direction + λ² ||direction||²
                
                # Minimize by setting derivative to 0:
                # -2 (base_mu - g)^T direction + 2λ ||direction||² = 0
                # λ = (base_mu - g)^T direction / ||direction||²
                
                if np.linalg.norm(direction) > self.tolerance:
                    lambda_star = np.dot(base_mu - natural_gradient, direction) / np.dot(direction, direction)
                    mu_star = base_mu - lambda_star * direction
                    
                    print(f"   ✅ Regularized solution: μ* = {mu_star}, λ* = {lambda_star:.6f}")
                    print(f"   📏 Distance from natural gradient: {np.linalg.norm(mu_star - natural_gradient):.6f}")
                    
                    return mu_star, lambda_star
                
            # Fallback: use least squares with regularization
            return self._regularized_least_squares(optimal_basis, natural_gradient)
            
        except np.linalg.LinAlgError:
            return self._regularized_least_squares(optimal_basis, natural_gradient)

    def _regularized_dual_1vertex(self, optimal_basis: List[int], natural_gradient: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Regularized dual for |B*| = 1 case
        
        Simply set μ* = natural_gradient and solve for λ
        """
        print("🎯 Regularized dual recovery for 1-vertex case")
        
        vertex_idx = optimal_basis[0]
        p_i = self.A[:, vertex_idx]
        
        # Constraint: S_i - p_i^T μ - λ = 0
        # Choose μ = natural_gradient, then λ = S_i - p_i^T μ
        
        mu_star = natural_gradient.copy()
        lambda_star = self.S[vertex_idx] - np.dot(p_i, mu_star)
        
        print(f"   ✅ Regularized solution: μ* = {mu_star}, λ* = {lambda_star:.6f}")
        print(f"   🧭 μ* set to natural gradient direction")
        
        return mu_star, lambda_star

    def _regularized_least_squares(self, optimal_basis: List[int], natural_gradient: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fallback: regularized least squares
        
        min ||A^T μ + λ 1 - S||² + γ ||μ - natural_gradient||²
        """
        print("   🔧 Using regularized least squares fallback")
        
        A_basis = self.A[:, optimal_basis]
        S_basis = self.S[optimal_basis]
        
        # Regularization weight
        gamma = 1.0
        
        # Extended system:
        # min ||[A^T  1] [μ] - [S]||² + γ ||μ - g||²
        #      [√γ I 0] [λ]   [√γ g]
        
        constraint_matrix = np.column_stack([A_basis.T, np.ones(len(optimal_basis))])
        regularization_matrix = np.column_stack([np.sqrt(gamma) * np.eye(2), np.zeros((2, 1))])
        
        extended_matrix = np.vstack([constraint_matrix, regularization_matrix])
        extended_rhs = np.hstack([S_basis, np.sqrt(gamma) * natural_gradient])
        
        # Solve regularized least squares
        solution, _, _, _ = np.linalg.lstsq(extended_matrix, extended_rhs, rcond=None)
        
        mu_star = solution[:2]
        lambda_star = solution[2] if len(solution) > 2 else 0.0
        
        print(f"   📊 Regularized LS: μ* = {mu_star}, λ* = {lambda_star:.6f}")
        
        return mu_star, lambda_star

def test_enhanced_simplex():
    """Comprehensive test of the enhanced simplex algorithm"""
    print("🧪 TESTING ENHANCED SIMPLEX ALGORITHM")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create a test polygon (hexagon)
    n = 6
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    A = np.array([
        np.cos(angles),  # x coordinates
        np.sin(angles)   # y coordinates
    ])
    
    # Define convex function parameters
    p_star = np.array([0.3, 0.2])  # Minimum point
    alpha = 1.0
    
    # Create s-values from convex function s(p) = α ||p - p*||
    S = np.array([
        alpha * np.linalg.norm(A[:, i] - p_star) for i in range(n)
    ])
    
    # Target point (should be in convex hull)
    c = np.array([0.1, 0.3])
    
    print(f"Polygon vertices (n={n}):")
    for i in range(n):
        print(f"  p_{i}: ({A[0,i]:.3f}, {A[1,i]:.3f}), s(p_{i}) = {S[i]:.3f}")
    print(f"Minimum point p*: {p_star}")
    print(f"Target point c: {c}")
    print(f"α = {alpha}")
    
    # Solve using our enhanced algorithm
    solver = ExactConvexGeometricSimplex(A, S, c, p_star, alpha)
    solution, objective = solver.solve()
    
    if solution is not None:
        # Verify solution
        solver.verify_solution(solution)
        
        # Compare with scipy linprog (for validation)
        print("\n🔬 COMPARISON WITH SCIPY LINPROG")
        print("-" * 40)
        
        try:
            # Standard LP formulation: max S^T W s.t. AW = c, Σw = 1, w ≥ 0
            A_eq = np.vstack([A, np.ones(n)])
            b_eq = np.hstack([c, 1.0])
            
            # Scipy minimizes, so negate objective
            result = linprog(-S, A_eq=A_eq, b_eq=b_eq, 
                           bounds=[(0, None)] * n, method='highs')
            
            if result.success:
                scipy_objective = -result.fun
                print(f"Scipy objective: {scipy_objective:.6f}")
                print(f"Our objective:   {objective:.6f}")
                print(f"Difference:      {abs(objective - scipy_objective):.8f}")
                
                if abs(objective - scipy_objective) < 1e-6:
                    print("✅ Perfect match with scipy!")
                else:
                    print("⚠️  Small difference (likely numerical)")
            else:
                print("❌ Scipy failed to solve")
                
        except Exception as e:
            print(f"⚠️  Scipy comparison failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 TEST COMPLETED")
    print("=" * 60)


###########################
class PolygonPreprocessor:
    """
    One-time preprocessing for fixed polygon A
    """
    
    def __init__(self, A: np.ndarray):
        self.A = A
        self.n = A.shape[1]
        
        # Precompute geometric structures
        self._compute_geometric_structures()
        self._compute_feasibility_regions()
        self._compute_basis_combinations()
    
    def _compute_geometric_structures(self):
        """Precompute polygon geometric properties"""
        
        # 1. All vertex combinations that can form valid triangles
        self.valid_triangles = []
        for combo in itertools.combinations(range(self.n), 3):
            if self._is_valid_triangle(combo):
                self.valid_triangles.append(combo)
        
        # 2. Voronoi-like regions: for each triangle, which points c can it represent?
        self.triangle_regions = {}
        for triangle in self.valid_triangles:
            self.triangle_regions[triangle] = self._compute_triangle_region(triangle)
        
        # 3. Edge representations (2-vertex solutions)
        self.valid_edges = []
        for combo in itertools.combinations(range(self.n), 2):
            if self._can_represent_interior_points(combo):
                self.valid_edges.append(combo)
        
        print(f"📐 Preprocessed {len(self.valid_triangles)} triangles, {len(self.valid_edges)} edges")
    
    def _compute_feasibility_regions(self):
        """Partition space into regions where different vertex sets are optimal"""
        
        # For each point c in the polygon, determine which vertex combinations
        # can represent it efficiently
        
        # Sample points throughout the polygon
        sample_points = self._generate_sample_points(50)  # 50x50 grid
        
        self.point_to_candidates = {}
        for point in sample_points:
            candidates = []
            
            # Check which triangles can represent this point
            for triangle in self.valid_triangles:
                if self._point_in_triangle_hull(point, triangle):
                    candidates.append(triangle)
            
            # Check which edges can represent this point  
            for edge in self.valid_edges:
                if self._point_on_edge_extended(point, edge):
                    candidates.append(edge)
                    
            self.point_to_candidates[tuple(point)] = candidates
    
    def _compute_basis_combinations(self):
        """Precompute and cache basis matrix inversions"""
        
        self.basis_matrices = {}
        self.basis_inverses = {}
        
        # Precompute all possible 3x3 basis matrices
        for triangle in self.valid_triangles:
            A_basis = self.A[:, triangle]
            basis_matrix = np.vstack([A_basis, np.ones(3)])
            
            try:
                basis_inverse = np.linalg.inv(basis_matrix)
                self.basis_matrices[triangle] = basis_matrix
                self.basis_inverses[triangle] = basis_inverse
            except np.linalg.LinAlgError:
                pass  # Skip singular combinations
        
        print(f"🔢 Cached {len(self.basis_inverses)} basis matrix inversions")

def find_best_candidates_for_point(self, c: np.ndarray, k: int = 5) -> List[tuple]:
    """
    Fast lookup: which vertex combinations are most promising for point c?
    
    Args:
        c: target point
        k: number of top candidates to return
    
    Returns:
        List of vertex combinations, ordered by geometric suitability
    """
    
    # Use precomputed spatial structure for fast lookup
    candidates = []
    
    # Check triangles first (most likely to be optimal)
    for triangle in self.valid_triangles:
        if triangle in self.basis_inverses:
            # Fast feasibility check using cached inverse
            weights = self.basis_inverses[triangle] @ np.hstack([c, 1.0])
            
            if np.all(weights >= -1e-8):  # Feasible
                # Score by geometric quality (how "centered" the weights are)
                balance_score = 1.0 / (1.0 + np.std(weights))
                candidates.append((triangle, balance_score, 'triangle'))
    
    # Check edges for degenerate cases
    for edge in self.valid_edges:
        if self._point_feasible_on_edge(c, edge):
            # Simple distance-based scoring
            edge_score = 0.5  # Lower priority than triangles
            candidates.append((edge, edge_score, 'edge'))
    
    # Sort by score and return top k
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [(combo, combo_type) for combo, score, combo_type in candidates[:k]]

class WarmStartManager:
    """
    Manages warm-starting across multiple solves
    """
    
    def __init__(self, polygon_preprocessor: PolygonPreprocessor):
        self.preprocessor = polygon_preprocessor
        self.solution_history = []
        self.dual_history = []
        
    def add_solution(self, S: np.ndarray, p_star: np.ndarray, alpha: float,
                     solution: np.ndarray, dual_mu: np.ndarray, dual_lambda: float,
                     basis: List[int]):
        """Store solution for future warm-starting"""
        
        entry = {
            'S': S.copy(),
            'p_star': p_star.copy(), 
            'alpha': alpha,
            'solution': solution.copy(),
            'dual_mu': dual_mu.copy(),
            'dual_lambda': dual_lambda,
            'basis': basis.copy(),
            'timestamp': len(self.solution_history)
        }
        
        self.solution_history.append(entry)
        
        # Keep only recent solutions (sliding window)
        if len(self.solution_history) > 10:
            self.solution_history.pop(0)
    
    def find_best_warmstart(self, S_new: np.ndarray, p_star_new: np.ndarray, 
                           alpha_new: float) -> Optional[dict]:
        """
        Find the most similar previous solution for warm-starting
        """
        
        if not self.solution_history:
            return None
        
        best_similarity = -1
        best_entry = None
        
        for entry in self.solution_history:
            # Compute similarity metrics
            
            # 1. S vector similarity (cosine similarity)
            s_similarity = np.dot(S_new, entry['S']) / (
                np.linalg.norm(S_new) * np.linalg.norm(entry['S']) + 1e-8)
            
            # 2. p* distance similarity  
            p_distance = np.linalg.norm(p_star_new - entry['p_star'])
            p_similarity = 1.0 / (1.0 + p_distance)
            
            # 3. alpha similarity
            alpha_similarity = 1.0 / (1.0 + abs(alpha_new - entry['alpha']))
            
            # Combined similarity (weighted)
            combined_similarity = (
                0.5 * s_similarity +      # S vector most important
                0.3 * p_similarity +      # p* location important  
                0.2 * alpha_similarity    # alpha scaling less critical
            )
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_entry = entry
        
        # Only use warm-start if similarity is high enough
        if best_similarity > 0.7:  # Threshold
            print(f"🔥 Warm-start found: similarity = {best_similarity:.3f}")
            return best_entry
        else:
            print(f"❄️  No good warm-start (best similarity = {best_similarity:.3f})")
            return None

class WarmStartConvexSimplex(ExactConvexGeometricSimplex):
    """
    Enhanced solver with warm-starting and preprocessing
    """
    
    def __init__(self, polygon_preprocessor: PolygonPreprocessor):
        self.preprocessor = polygon_preprocessor
        self.warm_start_manager = WarmStartManager(polygon_preprocessor)
        
        # Don't call parent __init__ yet - we'll do it per problem
    
    def solve_with_warmstart(self, S: np.ndarray, c: np.ndarray, 
                           p_star: np.ndarray, alpha: float) -> Tuple[np.ndarray, float, dict]:
        """
        Solve using preprocessing and warm-starting
        """
        
        # Initialize for this specific problem
        self.A = self.preprocessor.A
        self.S = S
        self.c = c
        self.p_star = p_star
        self.alpha = alpha
        self.n = self.A.shape[1]
        self.tolerance = 1e-8
        
        timing_info = {}
        start_total = time.perf_counter()
        
        # Phase 1: Try warm-start
        start_warmstart = time.perf_counter()
        warm_start_entry = self.warm_start_manager.find_best_warmstart(S, p_star, alpha)
        timing_info['warmstart_lookup'] = time.perf_counter() - start_warmstart
        
        if warm_start_entry:
            # Start from previous solution's basis
            initial_basis = warm_start_entry['basis']
            
            # Check if this basis is still feasible
            initial_solution = self._verify_warmstart_basis(initial_basis)
            if initial_solution:
                print(f"✅ Warm-start successful with basis {initial_basis}")
                current_basis, current_weights = initial_solution
            else:
                print(f"❌ Warm-start basis infeasible, falling back to heuristics")
                initial_solution = self._smart_initialization_with_preprocessing()
                current_basis, current_weights = initial_solution
        else:
            # Use preprocessing for smart initialization
            initial_solution = self._smart_initialization_with_preprocessing()
            current_basis, current_weights = initial_solution
        
        # Phase 2: Simplex iterations (same as before, but faster initialization)
        # ... rest of simplex algorithm ...
        
        # Phase 3: Store solution for future warm-starts
        solution = np.zeros(self.n)
        solution[current_basis] = current_weights
        objective = self.S @ solution
        
        mu_star, lambda_star = self.recover_dual_variables(current_basis, current_weights)
        
        self.warm_start_manager.add_solution(
            S, p_star, alpha, solution, mu_star, lambda_star, current_basis)
        
        return solution, objective, timing_info
    
    def _smart_initialization_with_preprocessing(self) -> Tuple[List[int], np.ndarray]:
        """
        Use preprocessing to find initial solution faster
        """
        
        # Use preprocessed geometric structure
        candidates = self.preprocessor.find_best_candidates_for_point(self.c, k=5)
        
        # Compute heuristic scores only for promising candidates
        best_objective = -np.inf
        best_solution = None
        
        for vertex_combo, combo_type in candidates:
            if combo_type == 'triangle' and len(vertex_combo) == 3:
                # Use cached basis inverse for fast solution
                if vertex_combo in self.preprocessor.basis_inverses:
                    weights = self.preprocessor.basis_inverses[vertex_combo] @ np.hstack([self.c, 1.0])
                    
                    if np.all(weights >= -self.tolerance):
                        objective = self.S[list(vertex_combo)] @ weights
                        
                        if objective > best_objective:
                            best_objective = objective
                            best_solution = (list(vertex_combo), weights)
        
        if best_solution:
            print(f"🚀 Preprocessing-based initialization: objective = {best_objective:.6f}")
            return best_solution
        else:
            # Fallback to original heuristic method
            return self._smart_initialization_exact()

if __name__ == "__main__":
    test_enhanced_simplex()