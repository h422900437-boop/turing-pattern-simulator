import numpy as np
import matplotlib.pyplot as plt
import time
import unittest

# =====================================================================
# First part：Solver Core
# =====================================================================

class ReactionModelBase:
    """For any reaction model, define a common interface for computing reaction terms f and g"""
    def compute_reaction(self, F, K, u, v):
        raise NotImplementedError

class GreyScottModel(ReactionModelBase):
    """Grey-Scott Model Interface"""
    def compute_reaction(self, F, K, u, v):
        uvv = u * v**2
        f = -uvv + F * (1.0 - u)
        g = uvv - (F + K) * v
        return f, g

class ExplicitScheme:
    """combination 1: Explicit time stepping + second-order finite difference"""
    def __init__(self, dx, dt, Du, Dv):
        #dx is the spatial step size, dt is the time step size, Du and Dv are diffusion coefficients δ
        self.dx2 = dx ** 2
        self.dt = dt
        self.Du = Du
        self.Dv = Dv

    def laplacian_2d(self, field):
        """Second order finite difference approximation for the Laplacian"""
        dx2 = self.dx2
        #lap=(u_i+1,j + u_i-1,j+u_i,j+1 + u_i,j-1 - 4u_i,j) / dx^2；
        #np.roll can also used to implement the periodic boundary conditions
        lap = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
               np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
               4.0 * field) / dx2
        return lap

    def step(self, u, v, f, g):
        """Explicit time stepping"""
       
        dt = self.dt
        Du = self.Du
        Dv = self.Dv
        
        lap_u = self.laplacian_2d(u)
        lap_v = self.laplacian_2d(v)
        
        # 完美对应白板推导形式：u^{k+1} = u^k + dt * (D * Laplacian(u) + f(u))
        # This perfectly matches our derivation: u^{k+1} = u^k + dt * (D * Laplacian(u) + f(u))
        u_next = u + dt * (Du * lap_u + f)
        v_next = v + dt * (Dv * lap_v + g)
        
        return u_next, v_next


class ReactionDiffusionSolver:
    def __init__(self, N, dx, dt, steps, model, F, K, scheme="explicit", Du=1.0, Dv=1.0):
        self.N = N
        self.steps = steps
        self.model = model
        self.F = F
        self.K = K
        self.scheme_name = scheme
        #global uniform initialization strategy:
        # We start with a uniform state where U fills the space and V is zero, representing the unreacted state of the system
        u = np.ones((N, N))
        v = np.zeros((N, N))

        #global seeding strategy: we randomly inject a small amount of V at random locations across the entire grid to serve as seeds that trigger the reaction, creating initial perturbations in the system
        # We randomly select about 1% of the pixels as "reaction centers" where we inject a small amount of V to trigger the reaction, creating initial perturbations in the system
        seed_mask = np.random.rand(N, N) > 0.98
        v[seed_mask] = 0.5
        u[seed_mask] = 0.5

        # 3. Add small Gaussian noise to break perfect symmetry and promote pattern formation
        u += np.random.normal(scale=0.02, size=(N, N))
        v += np.random.normal(scale=0.02, size=(N, N))
        
        self.u = u
        self.v = v
        self.integrator = ExplicitScheme(dx, dt, Du, Dv)
  
    def run(self):
        start_time = time.perf_counter()#record the start time for performance measurement
        
        for i in range(self.steps):
            f, g = self.model.compute_reaction(self.F, self.K, self.u, self.v)#compute the reaction terms based on the current state of u and v
            self.u, self.v = self.integrator.step(self.u, self.v, f, g)#update u and v using the explicit time stepping scheme
            self.u = np.clip(self.u, 0, 1)
            self.v = np.clip(self.v, 0, 1)
            
            if i % 1000 == 0:
                print(f"[{self.scheme_name}] Progress: step {i}/{self.steps}")#periodically print progress every 1000 steps to monitor the simulation
                
        end_time = time.perf_counter()#record the end time and calculate the total generation time for the simulation
        gen_time = end_time - start_time
        return self.v, gen_time

class SolverVisualizer:
    @staticmethod
    def show_result(v_field, gen_time, scheme_name):
        plt.figure(figsize=(6, 6))
        plt.imshow(v_field, cmap='inferno')
        plt.title(f"{scheme_name}\nTime: {gen_time:.2f}s")
        plt.axis('off')
        plt.show()

# =====================================================================
# Second part - Unit Tests - meeting the requirements of section 3.2 in the documentation
# =====================================================================
class TestNumericalMethods(unittest.TestCase):
    
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

def leopard_stage1():
    return {
        'N': 200, 'dx': 1.0, 'dt': 1.0, 'steps': 10000,
        'u': 0.16, 'v': 0.08, 'F': 0.0367, 'K': 0.0649,
    }

# =====================================================================
# Third part - Execution program 
# =====================================================================
if __name__ == "__main__":
        params = leopard_stage1()
        Du = params['u']
        Dv = params['v']
        
        leopard_model = GreyScottModel()  
        
        print("Starting Explicit Euler simulation...")
        solver_ex = ReactionDiffusionSolver(
            N=params['N'], dx=params['dx'], dt=params['dt'], steps=params['steps'],
            model=leopard_model, F=params['F'], K=params['K'], scheme="explicit", Du=Du, Dv=Dv
        )
        v_ex, t_ex = solver_ex.run()
        SolverVisualizer.show_result(v_ex, t_ex, "Explicit Euler")
       
        unittest.main()