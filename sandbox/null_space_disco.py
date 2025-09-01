import numpy as np
import scipy as sc

def _compute_kerA(self, orthogonalize):
    """
    Direct Geometric Construction of null space basis V of A
    
    Constructs the basis by exploiting the geometric meaning:
    1. Internal xy-modes
    2. Internal z-modes
        
    Construct:
        V: (3n-6)x(3n) basis matrix for Ker(A)
    """
    # Type 1: XY-MODES (2n-3 vectors typically)
    xy_modes = _construct_xy_deformation_modes()  # 2n-3 x 3n

    # Type 2: Z-MODES (n-3 vectors typically) 
    z_modes = _construct_z_deformation_modes()  # n-3 x 3n
    
    # Stack all vectors
    V_matrix = np.concatenate((xy_modes, z_modes), axis=0)  # 3n-6 x 3n

    if orthogonalize:
        # Orthogonalize using QR decomposition
        Q, R = np.linalg.qr(V_matrix.T, mode='reduced')
        
        # Keep only linearly independent columns
        rank = np.sum(np.abs(np.diag(R)) > 1e-12)
        assert rank == (3 * self.n - 6)
        V_matrix = Q[:, :rank]

    print(f"Constructed null space basis:")
    print(f"  - {len(xy_modes)} xy-deformation modes")
    print(f"  - {len(z_modes)} z-deformation modes") 
    print(f"  - Total: {V_matrix.shape[1]} vectors (expected: {3*self.n-6})")
    
    return V_matrix

def _construct_xy_deformation_modes(self):
    """
    Construct internal xy-deformation modes
    
    Null vector of form (a1,b1,0,...,an,bn,0)

    These represent motions in the xy-plane that preserve:
    1. Centroid: Σaᵢ = 0, Σbᵢ = 0
    2. Shape constraint: Σ(xᵢaᵢ + yᵢbᵢ) = 0
    
    Method: Find null space of the 3×2n constraint matrix
    """
    if self.n <= 2:
        return []  # Need at least 3 vertices for non-trivial deformations
    
    # Build constraint matrix for (a₁,b₁,a₂,b₂,...,aₙ,bₙ)
    C_xy = np.zeros((3, 2*self.n))

    # Row 0: Σaᵢ = 0 (sum of x-displacements = 0)
    C_xy[0, 0::2] = 1  # Coefficients of a₁, a₂, ..., aₙ
    # Row 1: Σbᵢ = 0 (sum of y-displacements = 0)  
    C_xy[1, 1::2] = 1  # Coefficients of b₁, b₂, ..., bₙ    
    # Row 2: Σ(xᵢaᵢ + yᵢbᵢ) = 0 (orthogonal to polygon shape)
    C_xy[2, :] = self.wis.reshape(2*self.n)

    # Find null space of C_xy of size 3x2n
    _, s, Vt = sc.linalg.svd(C_xy, full_matrices=True, compute_uv=True)
    # The rank should be 3
    assert len(s) == 3
    assert np.all(~np.isclose(s, 0.))
    
    # Extract null space vectors (should be 2n-3 of them)
    xy_null = Vt[3:, :]  # (2n-3) × 2n
    
    # Convert to full 3n-dimensional vectors
    xy_modes  = np.concatenate(
        (xy_null.reshape(2*self.n-3, self.n, 2), np.zeros(2*self.n-3, self.n, 1)),
        axis=2
    ).reshape(2*self.n-3, 3*self.n)

    return xy_modes

def _construct_z_deformation_modes(self):
    """
    Construct z-deformation modes (bending/twisting)
    
    These represent motions in z-direction that preserve:
    1. Net displacement: Σcᵢ = 0
    2. Moment about y-axis: Σxᵢcᵢ = 0  
    3. Moment about x-axis: Σyᵢcᵢ = 0
    
    Method: Find null space of the 3×n constraint matrix
    """
    if n <= 3:
        return []  # Need at least 4 vertices for non-trivial z-deformations
    
    # Build constraint matrix for (c₁, c₂, ..., cₙ)
    C_z = np.array([
        np.ones(n),  # Row 0: Σcᵢ = 0
        x,           # Row 1: Σxᵢcᵢ = 0  
        y            # Row 2: Σyᵢcᵢ = 0
    ])  # 3 × n matrix
    
    # Find null space of C_z
    U, s, Vt = svd(C_z, full_matrices=True)
    rank = np.sum(s > 1e-12)
    
    if rank >= Vt.shape[0]:
        return []  # No null space
    
    # Extract null space vectors (should be n-3 of them)
    z_null = Vt[rank:, :]  # (n-3) × n
    
    # Convert to full 3n-dimensional vectors
    z_modes = []
    for k in range(z_null.shape[0]):
        v = np.zeros(3*n)
        for i in range(n):
            v[3*i] = 0              # aᵢ = 0
            v[3*i + 1] = 0          # bᵢ = 0
            v[3*i + 2] = z_null[k, i]  # cᵢ component
        
        # Normalize
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v /= norm
            z_modes.append(v)
    
    return z_modes






def verify_null_space_basis(V, x_poly, y_poly):
    """
    Verify that V is a valid null space basis for the structured matrix A
    """
    n = len(x_poly)
    
    # Build constraint matrix A
    A = build_constraint_matrix(x_poly, y_poly)
    
    # Check A @ V ≈ 0
    product = A @ V
    max_error = np.max(np.abs(product))
    
    # Check orthogonality of basis vectors
    gram_matrix = V.T @ V
    identity_error = np.max(np.abs(gram_matrix - np.eye(V.shape[1])))
    
    print(f"Verification results:")
    print(f"  - Max |A @ V|: {max_error:.2e}")
    print(f"  - Orthogonality error: {identity_error:.2e}")
    print(f"  - Basis vectors: {V.shape[1]}")
    print(f"  - Expected: {3*n - 6}")
    
    return max_error < 1e-10 and identity_error < 1e-12

def build_constraint_matrix(x_poly, y_poly):
    """
    Build the 6 × 3n constraint matrix A for verification
    """
    n = len(x_poly)
    A = np.zeros((6, 3*n))
    
    for i in range(n):
        A[0, 3*i + 2] = x_poly[i]    # Σ xᵢcᵢ = 0
        A[1, 3*i + 2] = y_poly[i]    # Σ yᵢcᵢ = 0
        A[2, 3*i] = x_poly[i]        # Σ(xᵢaᵢ + yᵢbᵢ) = 0
        A[2, 3*i + 1] = y_poly[i]
        A[3, 3*i] = 1                # Σ aᵢ = 0
        A[4, 3*i + 1] = 1            # Σ bᵢ = 0
        A[5, 3*i + 2] = 1            # Σ cᵢ = 0
    
    return A

# =====================================================================
# EXAMPLE USAGE AND TESTING
# =====================================================================

def example_usage():
    """
    Example usage with a triangle satisfying polygon constraints
    """
    print("=== Direct Geometric Construction Example ===\n")
    
    # Define triangle vertices (centroid at origin, moment condition satisfied)
    x_poly = [1.0, -0.5, -0.5]
    y_poly = [0.0, np.sqrt(3)/2, -np.sqrt(3)/2]
    
    print(f"Polygon vertices: {list(zip(x_poly, y_poly))}")
    print(f"Constraints check:")
    print(f"  - Sum xᵢ: {sum(x_poly):.6f}")
    print(f"  - Sum yᵢ: {sum(y_poly):.6f}")  
    print(f"  - Sum xᵢyᵢ: {sum(x_poly[i]*y_poly[i] for i in range(len(x_poly))):.6f}")
    print()
    
    # Construct basis
    V = construct_V_direct_geometric(x_poly, y_poly)
    print()
    
    # Verify
    is_valid = verify_null_space_basis(V, x_poly, y_poly)
    print(f"\nBasis construction successful: {is_valid}")
    
    # Show first few basis vectors
    print(f"\nFirst basis vector (rotation mode):")
    v1 = V[:, 0]
    n = len(x_poly)
    for i in range(n):
        print(f"  Vertex {i+1}: ({v1[3*i]:.3f}, {v1[3*i+1]:.3f}, {v1[3*i+2]:.3f})")
    
    return V

if __name__ == "__main__":
    V_basis = example_usage()