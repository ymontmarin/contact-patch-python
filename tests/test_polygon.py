import numpy as np
from contactpatch.patches import PolygonContactPatch

ATOL = 1e-5
RTOL = 1e-2

def assert_close(a, b, atol=ATOL, rtol=RTOL):
    assert np.all(np.isclose(a, b, atol=atol, rtol=rtol))

def test_polygon_generation():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=5)
    assert PolygonContactPatch.test_polygon_vis(vis)

    mu = 2.
    poly = PolygonContactPatch(vis=vis, mu=mu, ker_precompute=False)


    PolygonContactPatch.test_polygon_vis(poly.vis)
    pvis = PolygonContactPatch.process_polygon(vis)[0]
    assert_close(pvis, poly.vis)

def test_polygon_linear_algebra_dense():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=4)
    mu = 2.
    poly = PolygonContactPatch(vis=vis, mu=mu, ker_precompute=True)
    n = poly.n
    rho = .1

    A = poly.get_A()
    AT = poly.get_AT()
    A_pinv = poly.get_A_pinv()

    ATA = poly.get_ATA()
    AAT = poly.get_AAT()
    AAT_inv = poly.get_AAT_inv()
    ATA_pinv = poly.get_ATA_pinv()
    ATA_reg_inv = poly.get_ATA_reg_inv(rho=rho)

    # Check dimension
    assert A.shape == (6, 3 * n)
    assert AT.shape == (3 * n, 6)
    assert A_pinv.shape == (3 * n, 6)

    assert ATA.shape == (3 * n, 3 * n)
    assert AAT.shape == (6, 6)
    assert AAT_inv.shape == (6, 6)
    assert ATA_pinv.shape == (3 * n, 3 * n)
    assert ATA_reg_inv.shape == (3 * n, 3 * n)

    # Check formula
    assert_close(A.T, AT)
    assert_close(np.linalg.pinv(A), A_pinv)

    assert_close(A @ AT, AAT)
    assert_close(AT @ A, ATA)

    assert_close(np.linalg.inv(AAT), AAT_inv)
    assert_close(np.linalg.pinv(ATA), ATA_pinv)
    assert_close(np.linalg.inv(ATA + rho * np.eye(3 * n)), ATA_reg_inv)

    # Check logic
    assert_close(A @ A_pinv, np.eye(6))
    assert_close(AAT @ AAT_inv, np.eye(6))


def test_polygon_linear_algebra_sparse():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=4)
    mu = 2.
    poly = PolygonContactPatch(vis=vis, mu=mu, ker_precompute=True)
    n = poly.n
    rho = .1

    A = poly.get_A()
    AT = poly.get_AT()
    A_pinv = poly.get_A_pinv()

    ATA = poly.get_ATA()
    AAT = poly.get_AAT()
    AAT_inv = poly.get_AAT_inv()
    ATA_pinv = poly.get_ATA_pinv()
    ATA_reg_inv = poly.get_ATA_reg_inv(rho=rho)

    # Random element and apply
