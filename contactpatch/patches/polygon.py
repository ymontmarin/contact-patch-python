import numpy as np
import itertools as itt
import scipy as sc

from contactpatch.coneproject import ProjectedGradient

SOLVERS = {
    "PGD": ProjectedGradient
}
class PolygonContactPatch:
    def __init__(self, vis, mu, ker_precompute=False, warmstart_strat=None, solver_tyep='PGD', solver_kwargs=None):
        """
        vis (nx2): the n vertices of a convex polygon P centered in its barycenter and align with principle axis
            Without loss of generality:
                sum_i vi = (0,0)
                sum_i vi[0]vi[1] = 0
                counterclockwise sequence
        vis = [[x1,y1],
                 ...
               [xn,yn]]
        The flatten version is V = [x1,y1,...,xn,yn] (2n)

        mu: the friction coeficient s.t. |fi_T| leq mu fi_N or eq. fi in K_mu

        K = {(sum_i=1..n (vi,0)xfi, sum_i=1..n fi) | fi in K_mu )}
          = {AF | F in K_mu^n )}

        Where F = [fx1,fy1,fz1,...,fx1,fy1,fz1] (3n)
        The unflatten version is fis = [[fx1,fy1,fz1],
                                             ...
                                        [fxn,fyn,fzn]]

        A = [[ B1  ...  Bn ]
             [ I3  ...  I3 ]] (6x3n)
        Bi = [[     02       vi^perp ]
              [ -vi^perp^T     01    ]] (3x3)
    
        Because the vi are centered: Sum_i Bi = 03

        We note wi = vi^perp and mwi = -wi^perp
        """
        self.vis = self.process_polygon(vis)[0]

        self.mu = mu
        self.n = len(vis)
        self.hidden_shape = (self.n, 3)
        self.size = 6
        assert self.n > 2

        self.solver = SOLVERS[solver_tyep](self, **(solver_kwargs if solver_kwargs is not None else {}))

        self._force_cone_precompute()

        # Optional precompute
        self.ker_precompute = ker_precompute
        if ker_precompute:
            self._compute_pker()

    @classmethod
    def generate_polygon_vis(cls, N_sample=10, aimed_n=None, min_n=3, maxit=1000):
        if aimed_n is None:
            aimed_n = min_n
        assert aimed_n >= min_n
        for _ in range(maxit):
            points = np.random.randn(N_sample, 2)
            ch = sc.spatial.ConvexHull(points)
            vis = points[ch.vertices]  # Counterclockwise sequence
            n = vis.shape[0]
            if n >= aimed_n:
                # Choose aimed_n of the n vertices
                # We balance as much as possible and use randomness if unknown
                eq = np.array([aimed_n // n] * aimed_n, dtype=np.int32)
                eq[np.random.choice(np.arange(aimed_n, dtype=np.int32), size=(aimed_n % n), replace=False)] += 1
                assert np.sum(eq) == aimed_n
                query = np.cumsum(eq)
                query[-1] = 0
                vis = vis[query]
                return cls.process_polygon(vis)[0]
        raise Exception("N_sample must be increased")

    @staticmethod
    def test_polygon_vis(vis):
        return np.all(np.isclose(np.sum(vis, axis=0), 0)) and np.isclose(np.sum(vis[:,0] * vis[:,1]), 0)

    @staticmethod
    def process_polygon(vis):
        # Center the vis:
        t = - vis.sum(axis=0, keepdims=True) / vis.shape[0]
        vis += t

        # Find the principle axis:
        # The matrix M = Sum_i vi viT = [[ a c ]
        #                                [ c b ]]
        a,b = np.sum(vis * vis, axis=0)  # Sum xi^2, Sum yi^2
        c = np.sum(vis[:,0] * vis[:,1])  # Sum xiyi

        if np.isclose(c, 0.):
            return vis, np.eye(2), t

        # Let us find P ortho s.t.
        #   Sum_i (Pvi) (Pvi)T = P(Sum_i vi viT)PT
        #                      = P M PT
        #                      = diag(λ₁, λ₂)
        # I.e. we diagonalize M
        # Eigen value of [[ a c ]
        #                 [ c b ]]
        # are:
        # λ₁,₂ = (a + b ± √((a - b)² + 4c²))/2
        det = np.sqrt((a-b)**2 + 4*c**2)

        ev1 = .5 * (a + b - det)
        ev2 = .5 * (a + b + det)
        # ev1 <= ev2
        # with eigenvector v₁,₂ = [1, (λ₁,₂ - a)/c]
        # And u1 u2 the normalized vector
        # We want that [(u1,0),(s u2,0),(0,0,1)] be a bond
        # It is equivalent to:
        # sg(s*((λ₂ - a)/c - (λ₁ - a)/c)) = sg(s*√((a - b)² + 4c²)/c) > 0
        # s = sg(c)
        fact1 = (ev1 - a) / c
        fact2 = (ev2 - a) / c

        N1 = np.sqrt(1 + fact1**2)
        N2 = np.sqrt(1 + fact2**2)

        s = -1. if c < 0 else 1.

        P = np.array([[1 / N1, fact1 / N1],
                      [s / N2, s * fact2 / N2]])

        new_vis = np.einsum('ni,ji->nj',vis,P)

        # vis' = P(vis + t) = Pvis + Pt
        # So the relative placement of the new frame w.r.t the old one is:
        # new^M_old = [[ P 0 Pt ]
        #              [ 0 1 0  ]
        #              [ 0 0 1  ]                      
        # NOTE: the relative placement must be included in the contact jacobian
        return np.einsum('ni,ji->nj',vis,P), P, t

    def _force_cone_precompute(self):
        # In the code we use wi = vi^perp and mwi = -vi^perp
        self.wis = np.stack(
            (self.vis[:,1], -self.vis[:,0]),
            axis=1
        )  # wi = vi^perp = [a,b]^perp = [-b,a]
        self.mwis = -self.wis  # Could be unstored if needed

        # Moment of the polygon
        self.mx, self.my = np.sum(self.wis * self.wis, axis=0) # Sum xi^2, Sum yi^2
        # mxy = Sum xiyi is 0 ! Because of hypothesis

        # It is AAT and also the non null spectrum of ATA
        self.aat = np.array([self.mx, self.my, self.mx + self.my, self.n, self.n, self.n])
        self.aat_inv = 1 / self.aat
        self.aat_inv_sq = self.aat_inv**2

        # Min and max eigen values
        self.L = self.aat.max()
        self.l = self.aat.min()

    """
    Dense linear algebra API
    """
    def get_A(self):
        # Block construction
        res = np.zeros((6, 3 * self.n))
        for i in range(self.n):
            res[:2, 3*i+2] = self.wis[i]
            res[2, 3*i:3*i+2] = self.mwis[i]
            res[3:, 3*i:3*i+3] = np.eye(3)
        return res

    def get_AT(self):
        return self.get_A().T

    def get_AAT(self):
        """
        A = [[ B1  ...  Bn ]
             [ I3  ...  I3 ]] (6x3n)
        Bi = [[     02       vi^perp ]
              [ -vi^perp^T     01    ]] (3x3)

        AAT = [[ Sum_i BiBi^T   Sum_i Bi ]
               [ Sum_i Bi^T        nI3   ]]
        BiBiT = [[ wiwiT    0      ]
                 [    0      wiTwi ]]
        wiwiT = [[ xi^2 xiyi ]
                 [ xiyi yi^2 ]]

        But polygon
        - is centered: Sum_i Bi = 0
        - principal alligned: Sum_i xiyi = 0

        So      
        AAT = [[ C   0  ]
               [ 0  nId ]]
        C = [[ mx  0   0     ]
             [ 0   my  0     ]
             [ 0   0   mx+my ]]

        i.e. AAT = diag((mx,1),(my,1),(mx+my,1),(n,3))
        """
        return np.diag(self.aat)

    def get_AAT_inv(self):
        """
        Inverse of diag is trivial
        """
        return np.diag(self.aat_inv)

    def get_A_pinv(self):
        """
        A is 6,3n of rank 6
        so pinv(A) = A^T(AA^T)^-1
        """
        return self.get_AT() * self.aat_inv[np.newaxis, :]

    def get_ATA(self):
        """
        A = [[ B1  ...  Bn ]
             [ I3       I3 ]] (6x3n)
        Bi = [[  02  wi ]
              [ -wiT  01 ]] (3x3)

        ATA = [[(B1TB1 + I3)  ...  (B1TBn + I3)]
                        ...
               [(BnTB1 + I3)  ...  (BnTBn + I3)]]
        BiTBj = [[ wiwjT 0     ]
                 [   0   wiTwj ]]
        """
        res = np.zeros((3 * self.n, 3 * self.n))

        # Add upper left part
        res.reshape(self.n, 3, self.n, 3)[:,:2,:,:2] = (
            np.einsum('ia,jb->iajb', self.wis, self.wis)
            + np.eye(2)[np.newaxis, :, np.newaxis, :]
        )
        # Add scalar part
        res.reshape(self.n, 3, self.n, 3)[:,2,:,2] = np.einsum('ia,ja->ij', self.wis, self.wis) + 1.
        return res

    def get_ATA_pinv(self):
        """
        pinv(ATA) = pinv(A)pinv(A)T
                  = A^T(AA^T)^-1 (AA^T)^-T A
                  = A^T diag(aat_inv**2) A

        Recall:
        A = [[ B1  ...  Bn ]
             [ I3       I3 ]] (6x3n)
        Bi = [[  02  wi ]
              [ -wiT  01 ]] (3x3)

        If we block write a 6 element diagonal D=diag(e1,e2,e3,f1,f2,f3)
        as D = [[E,0],[0,F]], with E = diag(e1,e2,e3), F = diag(f1,f2,f3)
        we have:
        A^TDA = [[(B1TEB1 + F)  ...  (B1TDBn + F)]
                         ...
                 [(BnTEB1 + F)  ...  (BnTDBn + F)]]

        BiTEBj = [[ e3 wi wjT       0     ]
                  [   0          wiT E' wj ]]
        where E' = diag(e1,e2)
        """
        res = np.zeros((3 * self.n, 3 * self.n))

        diag = self.aat_inv_sq

        # Add upper left part
        res.reshape(self.n, 3, self.n, 3)[:,:2,:,:2] = (
            np.einsum('ia,jb->iajb', self.wis, diag[2] * self.wis)
            + np.diag(diag[3:5])[np.newaxis, :, np.newaxis, :]
        )
        # Add scalar part
        res.reshape(self.n, 3, self.n, 3)[:,2,:,2] = np.einsum('ia,ja->ij', self.wis,  diag[np.newaxis, :2] * self.wis) + diag[5]
        return res

    def get_ATA_reg_inv(self, rho):
        """
        WOODBURY FORMULA:
            (M + UCV)^ = M^ - M^ U(C^ + V M^ U)^ V M^

        So (rI + ATA)^ = 1/rI - 1/rI A (I + A 1/rI AT)^ AT 1/rI
        Recall that AAT = diag(d) so:
            (I + A 1/rI AT) = diag(1 + d/r)
            -1/r(I + A 1/rI AT)^ = diag(-1/(r + d))
            -1/r^2(I + A 1/rI AT)^ = diag(-1/(r(r + d)))

        And:
            (rI + ATA)^ = 1/r[I + A diag(-1/(r + d)) AT]
                        = 1/rI + A diag(-1/r(r + d)) AT

        We can reuse the calculus trick of pinv(ATA) with the new diagonal
        """
        res = np.zeros((3 * self.n, 3 * self.n))

        rho_inv = 1. / rho
        diag = -rho_inv / (self.aat + rho)

        # Add upper left part
        res.reshape(self.n, 3, self.n, 3)[:,:2,:,:2] = (
            np.einsum('ia,jb->iajb', self.wis, diag[2] * self.wis)
            + np.diag(diag[3:5])[np.newaxis, :, np.newaxis, :]
        )
        # Add scalar part
        res.reshape(self.n, 3, self.n, 3)[:,2,:,2] = np.einsum('ia,ja->ij', self.wis,  diag[np.newaxis, :2] * self.wis) + diag[5]
        # Add identity term
        res += rho_inv * np.eye(3 * self.n)
        return res

    """
    Sparse linear algebra API
    """
    def apply_A(self, fis, _out=None):
        """
        fis: (nx3)
        return: l (6) as m,f
        l = [
            sum_i xi fzi,
            sum_i yi fzi,
            -sum_i (xi fxi + yi fyi),
            sum_i fxi,
            sum_i fyi,
            sum_i fzi
        ]
        """
        if _out is None:
            _out = np.zeros(6)

        _out[:2] = np.sum(self.wis * fis[:,2][:, np.newaxis], axis=0)
        _out[2] = np.sum(self.mwis * fis[:,:2])
        _out[3:] = np.sum(fis, axis=0)
        return _out

    def apply_AT(self, l, _out=None):
        """
        l: (6) as m,f
        return fis: (nx3)
        """
        if _out is None:
            _out = np.zeros((self.n, 3))

        _out[:, :2] = l[2] * self.mwis
        _out[:, 2] = self.wis[:, 0] * l[0] + self.wis[:, 1] * l[1]
        _out += l[3:][np.newaxis, :]
        return _out

    def apply_AAT(self, l, _out=None):
        if _out is None:
            return l * self.aat
        _out[...] = l * self.aat
        return _out

    def apply_AAT_(self, l):
        l *= self.aat
        return l

    def apply_AAT_inv(self, l, _out=None):
        if _out is None:
            return l * self.aat_inv
        _out[...] = l * self.aat_inv
        return _out

    def apply_AAT_inv_(self, l):
        l *= self.aat_inv
        return l

    def apply_A_pinv(self, l, _out=None):
        return self.apply_AT(self.apply_AAT_inv(l), _out=_out)

    def apply_ATA(self, fis, _out=None):
        return self.apply_AT(self.apply_A(fis), _out=_out)

    def apply_ATA_(self, fis):
        return self.apply_ATA(fis, _out=fis)

    def apply_ATA_pinv(self, fis, _out=None):
        return self.apply_AT(self.aat_inv_sq * self.apply_A(fis), _out=_out)

    def apply_ATA_pinv_(self, fis):
        return self.apply_ATA_pinv(fis, _out=fis)

    def apply_ATA_reg_inv(self, fis, rho, _out=None):
        """
        (rI + ATA)^ F = 1/r (F + AT diag(-1/(r + d)) A F)

        """
        diag = (-1.) / (self.aat + rho)
        _out = self.apply_AT(diag * self.apply_A(fis), _out=_out)
        _out += fis
        _out /= rho
        return _out

    def apply_ATA_reg_inv_(self, fis, rho):
        diag = (-1.) / (self.aat + rho)
        fis += self.apply_AT(diag * self.apply_A(fis))
        fis /= rho
        return fis

    """
    Precompute about the Kernel of A
    """
    def _compute_kerA(self, orthogonalize=False):
        """
        Direct Geometric Construction of null space basis V of A
        
        Constructs the basis by exploiting the geometric meaning:
        1. Internal xy-modes
        2. Internal z-modes
            
        Construct:
            V: (3n)x(3n-6) basis matrix for Ker(A)
            -> A V = 0 of size 6 x (3n-6)
        """
        # Type 1: XY-MODES (2n-3 vectors typically)
        xy_modes = self._construct_xy_modes().T  # 3n x 2n-3

        # Type 2: Z-MODES (n-3 vectors typically) 
        z_modes = self._construct_z_modes().T  # 3n x n-3

        # Stack all vectors
        V_matrix = np.concatenate((xy_modes, z_modes), axis=1)  # 3n x 3n-6

        if orthogonalize:
            # Orthogonalize using QR decomposition
            Q, R = np.linalg.qr(V_matrix, mode='reduced')
            
            # Keep only linearly independent columns
            rank = np.sum(np.abs(np.diag(R)) > 1e-12)
            assert rank == (3 * self.n - 6)
            V_matrix = Q[:, :rank]

        return V_matrix

    def _construct_xy_modes(self):
        """
        Construct internal xy-deformation modes
        
        Null vector of form (a1,b1,0,...,an,bn,0)

        These represent motions in the xy-plane that preserve:
        1. Centroid: Σaᵢ = 0, Σbᵢ = 0
        2. Shape constraint: Σ(xᵢaᵢ + yᵢbᵢ) = 0
        
        Method: Find null space of the 3×2n constraint matrix
        """
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
            (xy_null.reshape(2*self.n-3, self.n, 2), np.zeros((2*self.n-3, self.n, 1))),
            axis=2
        ).reshape(2*self.n-3, 3*self.n)

        return xy_modes

    def _construct_z_modes(self):
        """
        Construct z-deformation modes (bending/twisting)
        
        These represent motions in z-direction that preserve:
        1. Net displacement: Σcᵢ = 0
        2. Moment about y-axis: Σxᵢcᵢ = 0  
        3. Moment about x-axis: Σyᵢcᵢ = 0
        
        Method: Find null space of the 3×n constraint matrix
        """
        if self.n == 3:
            return np.zeros((0, 3*self.n))  # Need at least 4 vertices for non-trivial z-deformations

        # Build constraint matrix for (c₁, c₂, ..., cₙ)
        C_z = np.concatenate((self.wis, np.ones((self.n, 1))), axis=1).T  # 3 x n
        
        # Find null space of 3n
        _, s, Vt = sc.linalg.svd(C_z, full_matrices=True, compute_uv=True)
        # The rank should be 3
        assert len(s) == 3
        assert np.all(~np.isclose(s, 0.))
        
        # Extract null space vectors (should be n-3 of them)
        z_null = Vt[3:, :]  # (n-3) × n
        
        # Convert to full 3n-dimensional vectors
        z_modes  = np.concatenate(
            (np.zeros((self.n-3, self.n, 2)), z_null.reshape(self.n-3, self.n, 1)),
            axis=2
        ).reshape(self.n-3, 3*self.n)
        
        return z_modes

    def _compute_pker(self):
        kerA = self._compute_kerA()  # 3n x 3n-6
        self.pkers = {}
        for i in range(self.n):
            U, s, _ = sc.linalg.svd(kerA[3*i:3*i+3], full_matrices=False, compute_uv=True)
            assert len(s) == 3
            rank = sum(~np.isclose(s, 0.))
            if rank == 3:
                # Whole space, can not help to discrimine solution
                continue
            self.pkers[i] = U[:,:rank]  # A basis of the proj Kernel

    """
    Cone belongings
    """
    def is_inside_hidden_cone(self, fis):
        """
        Check if F is in K_mu^n
        """
        norms = np.linalg.norm(fis[:,:2], axis=1)
        muzs = self.mu * fis[:, 2]
        return np.all(np.logical_or(norms <= muzs, np.isclose(norms, muzs)))

    def is_inside_cone(self, l, pinv_l=None, do_solve=False):
        """
        Check if l is in K
        Return True if sure in, False if sure out and None if unsure

        l in K <=> pinv(A)(l) + Ker(A) cap K_mu^n neq emptyset

        so pinv(A)(l) in K_mu^n => l in K

        pinv(A)(l) + Ker(A) cap K_mu^n neq 0 => for all i, p_i(pinv(A)(l)) + p_i(Ker(A)) cap K_mu neq emptyset
        So if there is i s.t.  p_i(pinv(A)(l)) + p_i(Ker(A)) cap K_mu = emptyset => l not in K

        Otherwise we do not know for sure and must do some optimization...
        => boils down to project and check identity P^2 = P
        """
        if pinv_l is None:
            pinv_l = self.apply_A_pinv(l)
        if self.is_inside_hidden_cone(pinv_l):
            return True
        if self.ker_precompute and self.reject_nc_ker(pinv_l):
            # We sure it is outside
            return False
        if do_solve:
            # We are not sure we need to project to verify
            return np.all(np.isclose(self.project_cone(l),l))
        return None

    def reject_nc_ker(self, fis):
        """
        Check if there exist i s.t. p_i(fis) + p_i(Ker(A)) cap K_mu = emptyset
        The converse of a necessary condition based on A to be in the cone
        """
        for i, U in self.pkers.items():
            rank = U.shape[1]
            p = fis[3*i:3*i+3]
            if rank == 0:
                # it is a point, if outside, reject
                if p[2] < 0 or ((p[0]**2 + p[1]**2) > (self.mu**2 * p[2]**2)):
                    return True
                continue
            if rank == 1:
                # if it is a line there must be at least a t such that
                # p + tu is on the boundary of the cone
                # i.e. there is t:
                #    (px + tux)^2 + (py + tuy)^2 = mu^2 (pz + tuz)^2
                #    pz + tuz >= 0
                u = U[:, 0]
                A = u[0]**2 + u[1]**2 - self.mu**2 * u[2]**2
                half_B = u[0]*p[0] + u[1]*p[1] - self.mu**2 * u[2]*p[2]
                C = p[0]**2 + p[1]**2 - self.mu**2 * p[2]**2
                # Equation is At^2 + 2 hB t + C = 0 
                if np.isclose(A, 0):
                    # The line has one intersection with entire cone
                    # (angle with z-axis is the same as cone apperture)
                    if np.isclose(half_B, 0):
                        # There is either an infinity of intersection (C=0) or None
                        # We reject if None
                        if not np.isclose(C, 0):
                            return True
                        continue
                    if (p[2] - C / (2 * half_B) * u[2]) < 0:
                        # The intersection is : i = p - C / 2 hB u
                        # We reject if the intersection is not on the cone: iz < 0 
                        return True
                    continue
                det = half_B**2 - A * C
                if det < 0:
                    # There is no intersection, we can reject
                    return True
                if (p[2] - half_B /A * u[2]) + np.sqrt(det) * np.abs(u[2] / A) < 0:
                    # There is at most two intersection, verify that the one with highest z is
                    # not in the cone
                    # z = pz + (-hb +/- sqrt(det))/a uz
                    #   = (pz - hb/a uz) +/- sqrt(det) abs(uz/a)
                    return True
                continue
            if rank == 2:
                # It is a plane of normal n passing through p
                # the set of x s.s. n^Tx = n^Tp
                n = np.cross(U[:, 0], U[:, 1])
                if np.isclose(n[2], 0):
                    # Vertical plane, always an intersection
                    continue
                # The plane cross the z-axis in [0,0,alpha]
                alpha = np.dot(n, p) / n[2]
                if alpha >= 0:
                    # We sure have an intersection, can not reject
                    continue
                # When alpha < 0, we need that the normal is strictly in the dual cone to reject
                if n[0]**2 + n[1]**2 < n[2]**2 / self.mu**2:
                    return True
                continue
        return False

    """
    Cone elements generation
    """
    def generate_point_in_hidden_cone(self, fn_max=1):
        gene = np.random.uniform(0., 1., (self.n, 3))

        h = fn_max * (gene[:,0] ** (1/3))
        r = self.mu * h * np.sqrt(gene[:,1])
        t = 2 * np.pi * gene[:,2]

        return np.stack([r * np.cos(t), r * np.sin(t), h], axis=1)

    def generate_point_in_hidden_cone_space(self):
        return np.random.randn(self.n, 3)

    def generate_point_in_cone(self):
        return self.apply_A(self.generate_point_in_hidden_cone())

    def generate_point_in_cone_space(self):
        return np.random.randn(6)


    """
    Cone projection:
    Using solver and warmstart strat
    """
    def project_hidden_cone(self, fis, _out=None):
        """
        Project in K_mu^n
        """
        if _out is None:
            _out = np.zeros(fis.shape)
        xy = fis[:,:2]
        norm_xy = np.linalg.norm(xy, axis=1)
        z = fis[:, 2]

        almost_vertical = norm_xy < 1e-10
        inside_cone = np.logical_and(z >= 0, norm_xy <= self.mu * z)
        inside_polar_cone = np.logical_and(z < 0, np.logical_or(norm_xy <= (-1 / self.mu) * z,almost_vertical))

        not_cones = np.logical_not(np.logical_or(inside_cone, inside_polar_cone))
        numerical_area = np.logical_and(not_cones, almost_vertical)
        rest = np.logical_and(not_cones, np.logical_not(numerical_area))

        _out[inside_cone] = fis[inside_cone]
        _out[inside_polar_cone] = 0.
        _out[numerical_area, :2] = 0.

        alpha = (z[rest] + self.mu * norm_xy[rest]) / (self.mu**2 + 1)
        alpha = np.clip(alpha, 0., None)
        _out[rest, :2] = alpha[:, np.newaxis] * self.mu * xy[rest] / norm_xy[rest, np.newaxis]
        _out[rest, 2] = alpha

        return _out


    def project_hidden_cone_(self, fis):
        """
        Check if F is in K_mu^n
        """
        norm_xy = np.linalg.norm(fis[:,:2], axis=1)
        z = fis[:, 2]

        almost_vertical = norm_xy < 1e-10
        inside_cone = np.logical_and(z >= 0, norm_xy <= self.mu * z)
        inside_polar_cone = np.logical_and(z < 0, np.logical_or(norm_xy <= (-1 / self.mu) * z,almost_vertical))

        not_cones = np.logical_not(np.logical_or(inside_cone, inside_polar_cone))
        numerical_area = np.logical_and(not_cones, almost_vertical)
        rest = np.logical_and(not_cones, np.logical_not(numerical_area))

        fis[inside_polar_cone] = 0.
        fis[numerical_area, :2] = 0.

        alpha = (z[rest] + self.mu * norm_xy[rest]) / (self.mu**2 + 1)
        alpha = np.clip(alpha, 0., None)

        fis[rest, :2] *= (alpha * self.mu / norm_xy[rest])[:, np.newaxis]
        fis[rest, 2] = alpha
        return fis

    def project_cone(self, l, _out=None):
        """
        Check if F is in K_mu^n
        """
        pinv_l = self.apply_A_pinv(l)
        if self.is_inside_cone(l, pinv_l=pinv_l, do_solve=False):
            return l

        # Warmstart
        if self.warmstart_strat == 'prev' and self.fis_prev is not None:
            fis0 = self.fis_prev
        elif self.warmstart_strat == 'prev_linadjust' and self.fis_prev is not None:
            fis0 = self.project_hidden_cone(self.fis_prev + self.apply_A_pinv(l - self.l_prev))
        else:
            fis0 = self.project_hidden_cone(self.apply_A_pinv(l))

        # Update state for warmstart 1
        self.l_prev = l.copy()

        # Optim solve and retrieve projection
        fis_opti, success = self.solver.solve(l, x0=fis0)
        if not success:
            raise Exception('No solution found')

        proj = self.apply_A(fis_opti, _out=_out)

        # Update state for warmstart 2
        self.fis_prev = fis_opti.copy()
        self.proj_prev = proj.copy()

        return proj

    def project_cone_(self, l):
        return self.project_cone(l, _out=l)

    # """
    # Patch support function
    # """

    # """
    # Patch LP
    # """
