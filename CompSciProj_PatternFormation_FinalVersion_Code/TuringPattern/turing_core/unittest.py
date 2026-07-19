import unittest
import numpy as np
import scipy.sparse as sparse 
from turing_core.solvers.explicit import ExplicitScheme   

class   ExplicitSchemeTests(unittest.TestCase):
    
    def test_laplacian_2d(self):
        """test the accuracy of the second-order spatial finite difference approximation for the Laplacian"""
        dx = 1.0
        scheme = ExplicitScheme(dx=dx, dt=0.01, Du=1.0, Dv=1.0)
        
        #we create a simple test field with a single non-zero point in the center to verify the correctness of the Laplacian implementation
        field = np.zeros((3, 3))
        field[1, 1] = 1.0
        
        lap = scheme.laplacian_2d(field)
        
        # Verify the Laplacian value at the center point: (0+0+0+0 - 4*1)/1^2 = -4
        self.assertEqual(lap[1, 1], -4.0)
        # Verify the adjacent points: (1 - 4*0) / 1^2 = 1
        self.assertEqual(lap[0, 1], 1.0)
        self.assertEqual(lap[2, 1], 1.0)
        self.assertEqual(lap[1, 0], 1.0)
        self.assertEqual(lap[1, 2], 1.0)

    def test_explicit_euler_step(self):
        """test the correctness of a single time step of the explicit Euler scheme by comparing the computed next state against the expected values based on the known reaction and diffusion terms for a simple test case"""
        dt = 0.1
        Du, Dv = 1.0, 2.0
        scheme = ExplicitScheme(dx=1.0, dt=dt, Du=Du, Dv=Dv)
        
        # We use a uniform field where the Laplacian should be zero, allowing us to isolate the effect of the reaction terms in the update step
        u = np.ones((3, 3))
        v = np.ones((3, 3)) * 2.0
        
        # We assume constant reaction terms for this test, which allows us to directly calculate the expected next state based on the explicit Euler update formula without the influence of diffusion
        react_u = np.ones((3, 3)) * 0.5
        react_v = np.ones((3, 3)) * -0.5
        
        u_next, v_next = scheme.step(u, v, react_u, react_v)
        
        # u_next = u + dt * (Du * 0 + react_u) = 1.0 + 0.1 * 0.5 = 1.05
        self.assertAlmostEqual(u_next[1, 1], 1.05)
        # v_next = v + dt * (Dv * 0 + react_v) = 2.0 + 0.1 * (-0.5) = 1.95
        self.assertAlmostEqual(v_next[1, 1], 1.95)

class   ImplicitSchemeTests(unittest.TestCase):

    def simple_laplacian_test():
        print("run simple_laplacian_test...")
    
    N = 10
    dx = 1.0

    I_1D = sparse.eye(N, format='csc')
    e = np.ones(N)

    # here, the coefficients -2, 1, 1 correspond to the standard finite difference method for the second derivative (Laplacian) in 1D. The diagonals are set up to represent the second derivative with periodic boundary conditions.
    D_1D = sparse.spdiags([e, e, -2*e, e, e], [-N+1, -1, 0, 1, N-1], N, N, format='csc')
    L = (sparse.kron(I_1D, D_1D) + sparse.kron(D_1D, I_1D)) / (dx ** 2)

    # test the Laplacian operator by applying it to a uniform field of ones. The expected result should be zero everywhere due to the nature of the Laplacian, which measures the difference between a point and its neighbors. In a uniform field, there are no differences, so the output should be zero.
    u_flat = np.ones(N * N)
    result = L.dot(u_flat)

    max_error = np.max(np.abs(result))
    
    if max_error < 1e-10:
        print(f"ture,max_error: {max_error:.2e}")
    else:
        print(f"false,max_error: {max_error:.2e}")