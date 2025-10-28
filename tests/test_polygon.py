import numpy as np
from contactpatch.patches import PolygonContactPatch

ATOL = 1e-5
RTOL = 1e-2


def assert_close(a, b, atol=ATOL, rtol=RTOL):
    assert np.all(np.isclose(a, b, atol=atol, rtol=rtol))


def test_polygon_generation():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=5)
    assert PolygonContactPatch.test_polygon_vis(vis)

    mu = 2.0
    poly = PolygonContactPatch(vis=vis, mu=mu, ker_precompute=False)

    PolygonContactPatch.test_polygon_vis(poly.vis)
    pvis = PolygonContactPatch.process_polygon(vis)[0]
    assert_close(pvis, poly.vis)


def test_polygon_linear_algebra_dense():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=4)
    mu = 2.0
    poly = PolygonContactPatch(vis=vis, mu=mu, ker_precompute=True)
    N = poly.N
    rho = 0.1

    A = poly.get_A()
    AT = poly.get_AT()
    A_pinv = poly.get_A_pinv()

    ATA = poly.get_ATA()
    AAT = poly.get_AAT()
    AAT_inv = poly.get_AAT_inv()
    ATA_pinv = poly.get_ATA_pinv()
    ATA_reg_inv = poly.get_ATA_reg_inv(rho=rho)

    # Check dimension
    assert A.shape == (6, 3 * N)
    assert AT.shape == (3 * N, 6)
    assert A_pinv.shape == (3 * N, 6)

    assert ATA.shape == (3 * N, 3 * N)
    assert AAT.shape == (6, 6)
    assert AAT_inv.shape == (6, 6)
    assert ATA_pinv.shape == (3 * N, 3 * N)
    assert ATA_reg_inv.shape == (3 * N, 3 * N)

    # Check formula
    assert_close(A.T, AT)
    assert_close(np.linalg.pinv(A), A_pinv)

    assert_close(A @ AT, AAT)
    assert_close(AT @ A, ATA)

    assert_close(np.linalg.inv(AAT), AAT_inv)
    assert_close(np.linalg.pinv(ATA), ATA_pinv)
    assert_close(np.linalg.inv(ATA + rho * np.eye(3 * N)), ATA_reg_inv)

    # Check logic
    assert_close(A @ A_pinv, np.eye(6))
    assert_close(AAT @ AAT_inv, np.eye(6))


def test_polygon_linear_algebra_sparse():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=4)
    mu = 2.0
    poly = PolygonContactPatch(vis=vis, mu=mu, ker_precompute=True)
    N = poly.N
    rho = 0.1

    # Generate random elements
    vfis = poly.generate_point_in_hidden_cone()
    assert poly.is_inside_hidden_cone(vfis)

    while True:
        ffis = poly.generate_point_in_hidden_cone_space()
        if not poly.is_inside_hidden_cone(ffis):
            break

    vfis_flat = vfis.reshape((3 * N))
    ffis_flat = ffis.reshape((3 * N))

    vl = poly.generate_point_in_cone()
    fl = poly.generate_point_in_cone_space()

    A = poly.get_A()
    AT = poly.get_AT()
    A_pinv = poly.get_A_pinv()

    ATA = poly.get_ATA()
    AAT = poly.get_AAT()
    AAT_inv = poly.get_AAT_inv()
    ATA_pinv = poly.get_ATA_pinv()
    ATA_reg_inv = poly.get_ATA_reg_inv(rho=rho)

    # Test matrix vector methods
    assert_close(poly.apply_A(vfis), A @ vfis_flat)
    assert_close(poly.apply_A(ffis), A @ ffis_flat)

    assert_close(poly.apply_AT(vl).reshape((3 * N,)), AT @ vl)
    assert_close(poly.apply_AT(fl).reshape((3 * N,)), AT @ fl)

    assert_close(poly.apply_A_pinv(vl).reshape((3 * N,)), A_pinv @ vl)
    assert_close(poly.apply_A_pinv(fl).reshape((3 * N,)), A_pinv @ fl)

    assert_close(poly.apply_AAT(vl), AAT @ vl)
    assert_close(poly.apply_AAT(fl), AAT @ fl)

    assert_close(poly.apply_AAT_inv(vl), AAT_inv @ vl)
    assert_close(poly.apply_AAT_inv(fl), AAT_inv @ fl)

    assert_close(poly.apply_ATA(vfis).reshape((3 * N,)), ATA @ vfis_flat)
    assert_close(poly.apply_ATA(ffis).reshape((3 * N,)), ATA @ ffis_flat)

    assert_close(poly.apply_ATA_pinv(vfis).reshape((3 * N,)), ATA_pinv @ vfis_flat)
    assert_close(poly.apply_ATA_pinv(ffis).reshape((3 * N,)), ATA_pinv @ ffis_flat)

    assert_close(
        poly.apply_ATA_reg_inv(vfis, rho).reshape((3 * N,)), ATA_reg_inv @ vfis_flat
    )
    assert_close(
        poly.apply_ATA_reg_inv(ffis, rho).reshape((3 * N,)), ATA_reg_inv @ ffis_flat
    )

    # Test inplace methods
    vl_c = vl.copy()
    fl_c = fl.copy()
    poly.apply_AAT_(vl_c)
    poly.apply_AAT_(fl_c)
    assert_close(vl_c, AAT @ vl)
    assert_close(fl_c, AAT @ fl)

    vl_c = vl.copy()
    fl_c = fl.copy()
    poly.apply_AAT_inv_(vl_c)
    poly.apply_AAT_inv_(fl_c)
    assert_close(vl_c, AAT_inv @ vl)
    assert_close(fl_c, AAT_inv @ fl)

    vfis_c = vfis.copy()
    ffis_c = ffis.copy()
    poly.apply_ATA_(vfis_c)
    poly.apply_ATA_(ffis_c)
    assert_close(vfis_c.reshape((3 * N,)), ATA @ vfis_flat)
    assert_close(ffis_c.reshape((3 * N,)), ATA @ ffis_flat)

    vfis_c = vfis.copy()
    ffis_c = ffis.copy()
    poly.apply_ATA_pinv_(vfis_c)
    poly.apply_ATA_pinv_(ffis_c)
    assert_close(vfis_c.reshape((3 * N,)), ATA_pinv @ vfis_flat)
    assert_close(ffis_c.reshape((3 * N,)), ATA_pinv @ ffis_flat)

    vfis_c = vfis.copy()
    ffis_c = ffis.copy()
    poly.apply_ATA_reg_inv_(vfis_c, rho)
    poly.apply_ATA_reg_inv_(ffis_c, rho)
    assert_close(vfis_c.reshape((3 * N,)), ATA_reg_inv @ vfis_flat)
    assert_close(ffis_c.reshape((3 * N,)), ATA_reg_inv @ ffis_flat)


def test_polygon_hidden_projection():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=4)
    mu = 2.0
    poly = PolygonContactPatch(vis=vis, mu=mu, ker_precompute=True)
    N = poly.N
    rho = 0.1

    # Generate random elements
    vfis = poly.generate_point_in_hidden_cone()
    assert_close(vfis, poly.project_hidden_cone(vfis))

    ffis = poly.generate_point_in_hidden_cone_space()
    p_ffis = poly.project_hidden_cone(ffis)

    assert poly.is_inside_hidden_cone(p_ffis)
    # p°p = p
    assert_close(poly.project_hidden_cone(p_ffis), p_ffis)

    # Test inplace
    vfis_c = vfis.copy()
    poly.project_hidden_cone_(vfis_c)
    assert_close(vfis, vfis_c)

    ffis_c = ffis.copy()
    poly.project_hidden_cone_(ffis_c)
    assert_close(p_ffis, ffis_c)


def test_projection_with_pgs():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=4)
    mu = 2.0
    solver_kwargs = {
        "max_iterations": 2000,
        "accel": True,
        "precond": True,
        "adaptive_restart": True,
        "armijo": True,
        "armijo_iter": 20,
        "armijo_sigma": 0.1,
        "armijo_beta": 0.5,
        "armijo_force_restart": 0.8,
        "rel_crit": 1e-6,
        "abs_crit": 1e-8,
        "rel_obj_crit": 1e-6,
        "abs_obj_crit": 1e-12,
        "optim_crit": 1e-12,
        "alpha": 1.0,
        "verbose": True,
    }
    poly = PolygonContactPatch(
        vis=vis,
        mu=mu,
        ker_precompute=False,
        warmstart_strat=None,
        solver_tyep="PGD",
        solver_kwargs=solver_kwargs,
    )

    fl = poly.generate_point_in_cone_space()

    pfl = poly.project_cone(fl)
    ppfl = poly.project_cone(pfl)
    print(pfl, " VS ", ppfl)
    assert_close(pfl, ppfl, atol=1e-6)


def test_projection_with_admm():
    vis = PolygonContactPatch.generate_polygon_vis(N_sample=10, aimed_n=4)
    mu = 2.0
    solver_kwargs = {
        "max_iterations": 2000,
        "rel_crit": 1e-4,
        "abs_crit": 1e-5,
        "abs_obj_crit": 1e-12,
        "min_residual_threshold": 1e-8,
        "rho_clip": 1e6,
        "prox": 1e-6,
        "alpha": 1.1,
        "rho_init": 1e-1,
        "rho_power": 0.3,
        "rho_power_factor": 0.15,
        "rho_lin_factor": 2.0,
        "rho_update_ratio": 10.0,
        "rho_update_cooldown": 5,
        "rho_adaptive_fraction": 0.4,
        "rho_update_rule": "osqp",
        "dual_momentum": 0.1,
        "verbose": True,
    }
    poly = PolygonContactPatch(
        vis=vis,
        mu=mu,
        ker_precompute=False,
        warmstart_strat=None,
        solver_tyep="ADMM",
        solver_kwargs=solver_kwargs,
    )

    fl = poly.generate_point_in_cone_space()

    pfl = poly.project_cone(fl)
    ppfl = poly.project_cone(pfl)
    print(pfl, " VS ", ppfl)
    assert_close(pfl, ppfl, atol=1e-6)
