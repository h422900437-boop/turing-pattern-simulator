"""
Unit tests for the Turing pattern solvers.

Run from the TuringPattern/ directory:

    python -m unittest discover -v
    python -m unittest turing_core.test_solvers -v

The CPU test cases (model, Laplacian, Crank-Nicolson, seeding) have no GPU
dependency and always run. The explicit-scheme test case requires CuPy and a
working CUDA device; it is skipped automatically when either is unavailable,
so the suite stays green on a CPU-only machine.
"""

import unittest

import numpy as np

from turing_core.models import Grey_Scott, leopard_model, giraffe_model
from turing_core.seeding.Leo_seeding import leo_seeding
from turing_core.seeding.Giraffe_seeding import giraffe_seeding
from turing_core.solvers.Implicit import CrankNicolsonScheme

# CuPy is an optional dependency: the implicit solver runs on CPU only, so the
# test suite must remain usable without a CUDA toolchain. Probe for it here and
# let the GPU test case skip itself rather than failing at import time.
try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    from turing_core.solvers.explicit import ExplicitScheme

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def make_cn_scheme(N=8, dx=1.0, dt=0.1, Du=1.0, Dv=0.5, steps=1, F=0.0367, K=0.0649):
    """Build a CrankNicolsonScheme with neutral fields, for operator-level tests."""
    dummy = np.zeros((N, N))
    return CrankNicolsonScheme(
        N=N, dx=dx, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K,
        u=dummy.copy(), v=dummy.copy(),
    )


class GreyScottModelTests(unittest.TestCase):
    """Reaction terms of the Gray-Scott system."""

    def test_reaction_terms_match_analytic_values(self):
        # f = -u*v^2 + F*(1 - u)  and  g = u*v^2 - (F + K)*v
        F, K = 0.04, 0.06
        u = np.array([[0.5]])
        v = np.array([[0.2]])

        f, g = Grey_Scott(F, K, u, v)

        uvv = 0.5 * 0.2 ** 2                      # 0.02
        self.assertAlmostEqual(f[0, 0], -uvv + F * (1.0 - 0.5))
        self.assertAlmostEqual(g[0, 0], uvv - (F + K) * 0.2)

    def test_trivial_steady_state_has_zero_reaction(self):
        # (u, v) = (1, 0) is the homogeneous steady state of the Gray-Scott
        # system: both reaction terms must vanish exactly there.
        u = np.ones((4, 4))
        v = np.zeros((4, 4))

        f, g = Grey_Scott(0.0367, 0.0649, u, v)

        np.testing.assert_allclose(f, 0.0, atol=1e-15)
        np.testing.assert_allclose(g, 0.0, atol=1e-15)

    def test_presets_expose_the_expected_parameters(self):
        for preset in (leopard_model(), giraffe_model()):
            for key in ("N", "dx", "dt", "steps", "Du", "Dv", "F", "K"):
                self.assertIn(key, preset)
            # Turing instability requires the inhibitor to diffuse faster than
            # the activator; both presets must satisfy Du > Dv.
            self.assertGreater(preset["Du"], preset["Dv"])


class LaplacianOperatorTests(unittest.TestCase):
    """Second-order finite-difference Laplacian with periodic boundaries."""

    def test_five_point_stencil_on_a_delta_field(self):
        # A unit impulse at the centre of a 3x3 grid with dx = 1 must produce
        # -4 at the impulse and +1 at each of its four neighbours.
        scheme = make_cn_scheme(N=3, dx=1.0)
        field = np.zeros((3, 3))
        field[1, 1] = 1.0

        lap = scheme._apply_laplacian(field)

        self.assertAlmostEqual(lap[1, 1], -4.0)
        self.assertAlmostEqual(lap[0, 1], 1.0)
        self.assertAlmostEqual(lap[2, 1], 1.0)
        self.assertAlmostEqual(lap[1, 0], 1.0)
        self.assertAlmostEqual(lap[1, 2], 1.0)

    def test_constant_field_has_zero_laplacian(self):
        scheme = make_cn_scheme(N=6)
        field = np.full((6, 6), 3.7)

        np.testing.assert_allclose(scheme._apply_laplacian(field), 0.0, atol=1e-12)

    def test_boundaries_wrap_around(self):
        # An impulse in the top-left corner must diffuse onto the opposite
        # edges, which only happens if the boundary condition is periodic.
        scheme = make_cn_scheme(N=4, dx=1.0)
        field = np.zeros((4, 4))
        field[0, 0] = 1.0

        lap = scheme._apply_laplacian(field)

        self.assertAlmostEqual(lap[3, 0], 1.0)   # wrapped along axis 0
        self.assertAlmostEqual(lap[0, 3], 1.0)   # wrapped along axis 1
        # Total stencil weight is zero, so the Laplacian of any field must
        # integrate to zero over a periodic domain.
        self.assertAlmostEqual(lap.sum(), 0.0, places=12)

    def test_second_order_spatial_convergence(self):
        # For U = sin(2*pi*x) * cos(2*pi*y) the exact Laplacian is -8*pi^2*U.
        # Halving dx must reduce the error by roughly a factor of four.
        errors, spacings = [], []
        for N in (16, 32, 64, 128):
            dx = 1.0 / N
            grid = np.linspace(0.0, 1.0, N, endpoint=False)
            X, Y = np.meshgrid(grid, grid, indexing="ij")
            U = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
            exact = -8.0 * np.pi ** 2 * U

            scheme = make_cn_scheme(N=N, dx=dx)
            errors.append(np.max(np.abs(scheme._apply_laplacian(U) - exact)))
            spacings.append(dx)

        order, _ = np.polyfit(np.log(spacings), np.log(errors), 1)
        self.assertAlmostEqual(order, 2.0, delta=0.1)


class CrankNicolsonOperatorTests(unittest.TestCase):
    """Matrix-free left/right-hand-side operators and the CG solve."""

    def test_lhs_and_rhs_reduce_to_identity_on_constant_fields(self):
        # The Laplacian of a constant field vanishes, so both operators must
        # leave such a field untouched.
        scheme = make_cn_scheme(N=5, dt=0.4, Du=1.3)
        x = np.full(25, 2.5)

        np.testing.assert_allclose(scheme._apply_lhs_matrix(x, scheme.Du), x, atol=1e-12)
        np.testing.assert_allclose(scheme._apply_rhs_matrix(x, scheme.Du), x, atol=1e-12)

    def test_lhs_and_rhs_differ_by_the_full_diffusion_step(self):
        # (I + dt/2 D L)x - (I - dt/2 D L)x == dt * D * L x
        scheme = make_cn_scheme(N=6, dx=1.0, dt=0.3, Du=0.8)
        rng = np.random.default_rng(0)
        x = rng.normal(size=36)

        difference = (
            scheme._apply_rhs_matrix(x, scheme.Du)
            - scheme._apply_lhs_matrix(x, scheme.Du)
        )
        expected = scheme.dt * scheme.Du * scheme._apply_laplacian(
            x.reshape(6, 6)
        ).flatten()

        np.testing.assert_allclose(difference, expected, atol=1e-12)

    def test_linear_operator_matches_the_explicit_matvec(self):
        # The LinearOperator registered for CG must apply exactly the same
        # stencil as the underlying helper, otherwise the solve is inconsistent
        # with the right-hand side it is given.
        scheme = make_cn_scheme(N=7, dt=0.25, Du=1.1)
        rng = np.random.default_rng(1)
        x = rng.normal(size=49)

        np.testing.assert_allclose(
            scheme.A_u_op.matvec(x),
            scheme._apply_lhs_matrix(x, scheme.Du),
            atol=1e-12,
        )

    def test_conjugate_gradient_inverts_the_lhs_operator(self):
        # Solving A x = A x_ref must recover x_ref, which verifies that the
        # operator is symmetric positive definite and CG is applied correctly.
        import scipy.sparse.linalg as splinalg

        scheme = make_cn_scheme(N=8, dx=1.0, dt=0.2, Du=0.9)
        rng = np.random.default_rng(2)
        x_ref = rng.normal(size=64)

        rhs = scheme.A_u_op.matvec(x_ref)
        x_solved, info = splinalg.cg(scheme.A_u_op, rhs, rtol=1e-10, atol=0.0)

        self.assertEqual(info, 0, "CG did not converge")
        np.testing.assert_allclose(x_solved, x_ref, atol=1e-6)


class CrankNicolsonIntegrationTests(unittest.TestCase):
    """End-to-end behaviour of the implicit time integrator."""

    def test_homogeneous_steady_state_is_stationary(self):
        # (u, v) = (1, 0) is an exact fixed point: reaction and diffusion both
        # vanish, so time stepping must not move the solution at all. This is
        # the numerical analogue of a conservation check.
        N = 12
        scheme = CrankNicolsonScheme(
            N=N, dx=1.0, dt=1.0, Du=0.16, Dv=0.08, steps=5,
            F=0.0367, K=0.0649,
            u=np.ones((N, N)), v=np.zeros((N, N)),
        )

        v_final, elapsed = scheme.run()

        np.testing.assert_allclose(v_final, 0.0, atol=1e-10)
        np.testing.assert_allclose(scheme.u, 1.0, atol=1e-10)
        self.assertGreater(elapsed, 0.0)

    def test_run_returns_a_finite_field_of_the_right_shape(self):
        N = 16
        u, v = leo_seeding(N)
        scheme = CrankNicolsonScheme(
            N=N, dx=1.0, dt=1.0, Du=0.16, Dv=0.08, steps=10,
            F=0.0367, K=0.0649, u=u, v=v,
        )

        v_final, _ = scheme.run()

        self.assertEqual(v_final.shape, (N, N))
        self.assertTrue(np.all(np.isfinite(v_final)), "solver produced NaN/Inf")


class SeedingTests(unittest.TestCase):
    """Initial-condition generators."""

    def test_leopard_seeding_shape_and_perturbation(self):
        N = 32
        u, v = leo_seeding(N)

        self.assertEqual(u.shape, (N, N))
        self.assertEqual(v.shape, (N, N))
        # u starts from the unreacted state (~1) and v from ~0, with sparse
        # nuclei injected to break symmetry; without any v > 0 no pattern can
        # ever form.
        self.assertTrue(np.any(v > 0.1), "no reaction nuclei were seeded")
        self.assertAlmostEqual(u.mean(), 1.0, delta=0.05)

    def test_giraffe_seeding_is_reproducible(self):
        # Giraffe_seeding fixes the RNG seed, so two calls must be identical.
        first_u, first_v = giraffe_seeding(20)
        second_u, second_v = giraffe_seeding(20)

        np.testing.assert_array_equal(first_u, second_u)
        np.testing.assert_array_equal(first_v, second_v)

    def test_giraffe_seeding_stays_in_the_physical_range(self):
        u, v = giraffe_seeding(24)

        self.assertGreaterEqual(u.min(), 0.0)
        self.assertLessEqual(u.max(), 1.0)
        self.assertGreaterEqual(v.min(), 0.0)
        self.assertLessEqual(v.max(), 1.0)


@unittest.skipUnless(CUPY_AVAILABLE, "CuPy with a CUDA device is not available")
class ExplicitSchemeGpuTests(unittest.TestCase):
    """CUDA kernel of the explicit Euler scheme (skipped without a GPU)."""

    def test_steady_state_is_stationary_on_gpu(self):
        # Same fixed point as the implicit test: the kernel must leave
        # (u, v) = (1, 0) unchanged.
        N = 16
        scheme = ExplicitScheme(
            dx=1.0, dt=1.0, Du=0.16, Dv=0.08, steps=20,
            F=0.0367, K=0.0649,
            u=np.ones((N, N)), v=np.zeros((N, N)),
        )

        v_final, _ = scheme.run()

        np.testing.assert_allclose(v_final, 0.0, atol=1e-6)

    def test_gpu_result_matches_a_numpy_reference_step(self):
        # One explicit Euler step on the GPU must reproduce the same update a
        # plain NumPy implementation of the scheme produces.
        N = 16
        rng = np.random.default_rng(3)
        u0 = 0.5 + 0.01 * rng.normal(size=(N, N))
        v0 = 0.25 + 0.01 * rng.normal(size=(N, N))
        dx, dt, Du, Dv, F, K = 1.0, 0.5, 0.16, 0.08, 0.0367, 0.0649

        scheme = ExplicitScheme(
            dx=dx, dt=dt, Du=Du, Dv=Dv, steps=1, F=F, K=K,
            u=u0.copy(), v=v0.copy(),
        )
        v_gpu, _ = scheme.run()

        def laplacian(field):
            return (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                - 4.0 * field
            ) / dx ** 2

        f, g = Grey_Scott(F, K, u0, v0)
        v_reference = v0 + dt * (Dv * laplacian(v0) + g)

        # The kernel computes in float32, so compare at single precision.
        np.testing.assert_allclose(v_gpu, v_reference, rtol=1e-4, atol=1e-5)

    def test_explicit_and_implicit_agree_over_a_short_horizon(self):
        # Both schemes discretise the same PDE. Over a few steps, and well
        # inside the stability limit, their solutions must stay close.
        N = 16
        u0, v0 = giraffe_seeding(N)
        dx, dt, Du, Dv, F, K, steps = 1.0, 0.05, 0.8, 0.4, 0.089, 0.060, 20

        v_exp, _ = ExplicitScheme(
            dx=dx, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K,
            u=u0.copy(), v=v0.copy(),
        ).run()
        v_imp, _ = CrankNicolsonScheme(
            N=N, dx=dx, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K,
            u=u0.copy(), v=v0.copy(),
        ).run()

        self.assertLess(np.max(np.abs(v_exp - v_imp)), 1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)