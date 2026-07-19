import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import unittest

# =====================================================================
# Solver Core
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


class CrankNicolsonScheme:
    """combination 2: Crank-Nicolson time stepping + second-order finite difference"""
    def __init__(self, N, dx, dt, Du, Dv):
        self.N = N
        self.dt = dt
        
        
        # construct the sparse Laplacian matrix L for 2D grid with periodic boundary conditions
        I_1D = sparse.eye(N, format='csc')
        e = np.ones(N)
        D_1D = sparse.spdiags([e, e, -2*e, e, e], [-N+1, -1, 0, 1, N-1], N, N, format='csc')
        L = (sparse.kron(I_1D, D_1D) + sparse.kron(D_1D, I_1D)) / (dx ** 2)
        
        # construct the CN left-hand side matrix A and right-hand side matrix B
        I_2D = sparse.eye(N**2, format='csc')
        A_u = I_2D - (dt / 2.0) * Du * L
        B_u = I_2D + (dt / 2.0) * Du * L
        A_v = I_2D - (dt / 2.0) * Dv * L
        B_v = I_2D + (dt / 2.0) * Dv * L
        
        # pre-factorize the sparse matrices A_u and A_v for efficient solving during time stepping
        self.B_u = B_u
        self.B_v = B_v
        self.solve_u = splinalg.factorized(A_u)
        self.solve_v = splinalg.factorized(A_v)

    def step(self, u, v, f, g):
        """CN time stepping"""
        dt = self.dt
        
        u_vec = u.flatten()
        v_vec = v.flatten()
        f_vec = f.flatten()
        g_vec = g.flatten()
        
        # The RHS includes the diffusion term from the previous time step and the nonlinear reaction term from the current time step
        rhs_u = self.B_u.dot(u_vec) + dt * f_vec
        rhs_v = self.B_v.dot(v_vec) + dt * g_vec
        
        u_next_vec = self.solve_u(rhs_u)
        v_next_vec = self.solve_v(rhs_v)
        
        return u_next_vec.reshape((self.N, self.N)), v_next_vec.reshape((self.N, self.N))

class SolverVisualizer:
    @staticmethod
    def show_result(v_field, gen_time, scheme_name):
        plt.figure(figsize=(6, 6))
        plt.imshow(v_field, cmap='Greys_r')
        plt.title(f"{scheme_name}\nTime: {gen_time:.2f}s")
        plt.axis('off')
        plt.show()


class ReactionDiffusionSolver:
    def __init__(self, N, dx, dt, steps, model, F, K, scheme="explicit", Du=1.0, Dv=1.0):
        self.N = N
        self.steps = steps
        self.model = model
        self.F = F
        self.K = K
        self.scheme_name = scheme
        #global uniform initialization strategy:
        
        np.random.seed(42)
        
        self.u = 0.5 + np.random.uniform(-0.1, 0.1, (N, N))
        self.v = 0.3 + np.random.uniform(-0.1, 0.1, (N, N))
            
        self.u = np.clip(self.u, 0, 1)
        self.v = np.clip(self.v, 0, 1)

        self.integrator = CrankNicolsonScheme(N, dx, dt, Du, Dv)

    def run(self):
        start_time = time.perf_counter()#record the start time for performance measurement
        
        for i in range(self.steps):
            f, g = self.model.compute_reaction(self.F, self.K, self.u, self.v)#compute the reaction terms based on the current state of u and v
            self.u, self.v = self.integrator.step(self.u, self.v, f, g)#update u and v using the explicit time stepping scheme
            
            if i % 1000 == 0:
                print(f"[{self.scheme_name}] Progress: step {i}/{self.steps}")#periodically print progress every 1000 steps to monitor the simulation
                
        end_time = time.perf_counter()#record the end time and calculate the total generation time for the simulation
        gen_time = end_time - start_time
        return self.v, gen_time

def Giraffe_stage1():
    return {
        'N': 200, 'dx': 1.0, 'dt': 0.02, 'steps': 20000,
        'u': 0.8, 'v': 0.4, 'F': 0.089, 'K': 0.060,
    }
class TestNumericalMethods(unittest.TestCase):

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
    

# =====================================================================
#  Execution program 
# =====================================================================
if __name__ == "__main__":
    
        params = Giraffe_stage1()
        Du = params['u']
        Dv = params['v']
        
        giraffe_model = GreyScottModel()  
    
        cn_steps = 20000 # we can use a larger steps number to simulate longer time periods 
        cn_dt = 0.02 # we increase the time step for efficiency, as Crank-Nicolson is unconditionally stable for linear problems, and can handle larger time steps than the explicit method while still maintaining accuracy
        
        print(f"\nStarting Crank-Nicolson simulation (testing with {cn_steps} steps)...")
        solver_cn = ReactionDiffusionSolver(
            N=params['N'], dx=params['dx'], dt=cn_dt, steps=cn_steps,
            model=giraffe_model, scheme="crank-nicolson",  F=params['F'], K=params['K'], Du=Du, Dv=Dv
        )
        u_cn, time_cn = solver_cn.run()
        SolverVisualizer.show_result(u_cn, time_cn, "Crank-Nicolson")
        