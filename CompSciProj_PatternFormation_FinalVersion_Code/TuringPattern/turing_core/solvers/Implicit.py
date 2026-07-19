# import numpy as np
# import scipy.sparse as sparse
# import scipy.sparse.linalg as splinalg
# import time
# from ..models import Grey_Scott

# class CrankNicolsonScheme:
#     """combination 2: Crank-Nicolson time stepping + second-order finite difference"""
#     def __init__(self, N, dx, dt, Du, Dv, steps,F,K,u,v):
#         self.N = N
#         self.dt = dt
#         self.steps = steps
#         self.Du = Du
#         self.Dv = Dv    
#         self.F = F
#         self.K = K
#         self.u = u
#         self.v = v
        
#         # construct the sparse Laplacian matrix L for 2D grid with periodic boundary conditions
#         I_1D = sparse.eye(N, format='csc')
#         e = np.ones(N)
#         D_1D = sparse.spdiags([e, e, -2*e, e, e], [-N+1, -1, 0, 1, N-1], N, N, format='csc')
#         L = (sparse.kron(I_1D, D_1D) + sparse.kron(D_1D, I_1D)) / (dx ** 2)
        
#         # construct the CN left-hand side matrix A and right-hand side matrix B
#         I_2D = sparse.eye(N**2, format='csc')
#         A_u = I_2D - (dt / 2.0) * Du * L
#         B_u = I_2D + (dt / 2.0) * Du * L
#         A_v = I_2D - (dt / 2.0) * Dv * L
#         B_v = I_2D + (dt / 2.0) * Dv * L
        
#         # pre-factorize the sparse matrices A_u and A_v for efficient solving during time stepping
#         self.B_u = B_u
#         self.B_v = B_v
#         self.solve_u = splinalg.factorized(A_u)
#         self.solve_v = splinalg.factorized(A_v)

#     def step(self, u, v, f, g):
#         """CN time stepping"""
#         dt = self.dt
        
#         u_vec = u.flatten()
#         v_vec = v.flatten()
#         f_vec = f.flatten()
#         g_vec = g.flatten()
        
#         # The RHS includes the diffusion term from the previous time step and the nonlinear reaction term from the current time step
#         rhs_u = self.B_u.dot(u_vec) + dt * f_vec
#         rhs_v = self.B_v.dot(v_vec) + dt * g_vec
        
#         u_next_vec = self.solve_u(rhs_u)
#         v_next_vec = self.solve_v(rhs_v)
        
#         return u_next_vec.reshape((self.N, self.N)), v_next_vec.reshape((self.N, self.N))

#     def run(self):
#         start_time = time.perf_counter()#record the start time for performance measurement
        
#         for i in range(self.steps):
#             f, g = Grey_Scott(self.F, self.K, self.u, self.v)#compute the reaction terms based on the current state of u and v
#             self.u, self.v = self.step(self.u, self.v, f, g)#update u and v using the explicit time stepping scheme
            
#             if i % 1000 == 0:
#                 print(f"[{CrankNicolsonScheme}] Progress: step {i}/{self.steps}")#periodically print progress every 1000 steps to monitor the simulation
                
#         end_time = time.perf_counter()#record the end time and calculate the total generation time for the simulation
#         gen_time = end_time - start_time
#         return self.v, gen_time


# import numpy as np
# import scipy.sparse.linalg as splinalg
# from scipy.sparse.linalg import LinearOperator
# import time
# from ..models import Grey_Scott

# class CrankNicolsonScheme:
#     """优化版：使用 Matrix-free (无矩阵) 线性求解器的 Crank-Nicolson 方案"""
#     def __init__(self, N, dx, dt, Du, Dv, steps, F, K, u, v):
#         self.N = N
#         self.dx = dx
#         self.dt = dt
#         self.steps = steps
#         self.Du = Du
#         self.Dv = Dv    
#         self.F = F
#         self.K = K
#         self.u = u
#         self.v = v
        
#         # 矩阵的维度是 N^2 x N^2，但我们永远不会真正生成它
#         self.shape = (N**2, N**2)
        
#         # 核心语法规则：定义如何计算 (A_u * x) 
#         def matvec_u(x_vec):
#             return self._apply_lhs_matrix(x_vec, self.Du)
            
#         # 核心语法规则：定义如何计算 (A_v * x)
#         def matvec_v(x_vec):
#             return self._apply_lhs_matrix(x_vec, self.Dv)

#         # 注册无矩阵的线性算子 (LinearOperator)
#         self.A_u_op = LinearOperator(self.shape, matvec=matvec_u)
#         self.A_v_op = LinearOperator(self.shape, matvec=matvec_v)

#     def _apply_laplacian(self, x_2d):
#         """利用循环移位直接在二维网格上现算拉普拉斯，避免生成稀疏矩阵"""
#         return (np.roll(x_2d, 1, axis=0) + np.roll(x_2d, -1, axis=0) +
#                 np.roll(x_2d, 1, axis=1) + np.roll(x_2d, -1, axis=1) -
#                 4.0 * x_2d) / (self.dx ** 2)

#     def _apply_lhs_matrix(self, x_vec, D):
#         """计算方程左侧： (I - dt/2 * D * Laplacian) * x"""
#         x_2d = x_vec.reshape((self.N, self.N))
#         lap_x = self._apply_laplacian(x_2d)
#         result_2d = x_2d - (self.dt / 2.0) * D * lap_x
#         return result_2d.flatten()

#     def _apply_rhs_matrix(self, x_vec, D):
#         """计算方程右侧： (I + dt/2 * D * Laplacian) * x"""
#         x_2d = x_vec.reshape((self.N, self.N))
#         lap_x = self._apply_laplacian(x_2d)
#         result_2d = x_2d + (self.dt / 2.0) * D * lap_x
#         return result_2d.flatten()

#     def step(self, u, v, f, g):
#         """CN 时间步进，使用共轭梯度法(CG)进行迭代求解"""
#         dt = self.dt
        
#         u_vec = u.flatten()
#         v_vec = v.flatten()
#         f_vec = f.flatten()
#         g_vec = g.flatten()
        
#         # 计算等式右端的已知向量： RHS = B * u_old + dt * f
#         rhs_u = self._apply_rhs_matrix(u_vec, self.Du) + dt * f_vec
#         rhs_v = self._apply_rhs_matrix(v_vec, self.Dv) + dt * g_vec
        
#         # 使用共轭梯度法求解。注意此处使用 rtol 适配最新版 SciPy 库
#         # 将上一时刻的解 (x0=u_vec) 作为初始猜想传递给 cg 求解器，极大加速收敛
#         u_next_vec, _ = splinalg.cg(self.A_u_op, rhs_u, x0=u_vec, rtol=1e-5)
#         v_next_vec, _ = splinalg.cg(self.A_v_op, rhs_v, x0=v_vec, rtol=1e-5)
        
#         return u_next_vec.reshape((self.N, self.N)), v_next_vec.reshape((self.N, self.N))

#     def run(self):
#         start_time = time.perf_counter()
        
#         for i in range(self.steps):
#             f, g = Grey_Scott(self.F, self.K, self.u, self.v)
#             self.u, self.v = self.step(self.u, self.v, f, g)
            
#             if i % 1000 == 0:
#                 print(f"[{CrankNicolsonScheme.__name__} (Matrix-Free)] Progress: step {i}/{self.steps}")
                
#         end_time = time.perf_counter()
#         gen_time = end_time - start_time
#         return self.v, gen_time
    
import numpy as np
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg import LinearOperator
import time
from ..models import Grey_Scott

class CrankNicolsonScheme:
    """Optimized version: Crank-Nicolson scheme using a Matrix-free linear solver"""
    def __init__(self, N, dx, dt, Du, Dv, steps, F, K, u, v):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.steps = steps
        self.Du = Du
        self.Dv = Dv    
        self.F = F
        self.K = K
        self.u = u
        self.v = v
        
        # The dimension of the matrix is N^2 x N^2, but we will never actually generate it
        self.shape = (N**2, N**2)
        
        # Core syntax rule: Define how to calculate (A_u * x) 
        def matvec_u(x_vec):
            return self._apply_lhs_matrix(x_vec, self.Du)
            
        # Core syntax rule: Define how to calculate (A_v * x)
        def matvec_v(x_vec):
            return self._apply_lhs_matrix(x_vec, self.Dv)

        # Register the matrix-free linear operator (LinearOperator)
        self.A_u_op = LinearOperator(self.shape, matvec=matvec_u)
        self.A_v_op = LinearOperator(self.shape, matvec=matvec_v)

    def _apply_laplacian(self, x_2d):
        """Calculate the Laplacian directly on the 2D grid using circular shifts to avoid generating sparse matrices"""
        return (np.roll(x_2d, 1, axis=0) + np.roll(x_2d, -1, axis=0) +
                np.roll(x_2d, 1, axis=1) + np.roll(x_2d, -1, axis=1) -
                4.0 * x_2d) / (self.dx ** 2)

    def _apply_lhs_matrix(self, x_vec, D):
        """Calculate the left side of the equation: (I - dt/2 * D * Laplacian) * x"""
        x_2d = x_vec.reshape((self.N, self.N))
        lap_x = self._apply_laplacian(x_2d)
        result_2d = x_2d - (self.dt / 2.0) * D * lap_x
        return result_2d.flatten()

    def _apply_rhs_matrix(self, x_vec, D):
        """Calculate the right side of the equation: (I + dt/2 * D * Laplacian) * x"""
        x_2d = x_vec.reshape((self.N, self.N))
        lap_x = self._apply_laplacian(x_2d)
        result_2d = x_2d + (self.dt / 2.0) * D * lap_x
        return result_2d.flatten()

    def step(self, u, v, f, g):
        """CN time stepping, iterative solving using Conjugate Gradient (CG) method"""
        dt = self.dt
        
        u_vec = u.flatten()
        v_vec = v.flatten()
        f_vec = f.flatten()
        g_vec = g.flatten()
        
        # Calculate the known vector on the right side of the equation: RHS = B * u_old + dt * f
        rhs_u = self._apply_rhs_matrix(u_vec, self.Du) + dt * f_vec
        rhs_v = self._apply_rhs_matrix(v_vec, self.Dv) + dt * g_vec
        
        # Solve using the Conjugate Gradient method. Note the use of rtol here to adapt to the latest SciPy library
        # Pass the solution from the previous time step (x0=u_vec) as the initial guess to the cg solver, greatly accelerating convergence
        u_next_vec, _ = splinalg.cg(self.A_u_op, rhs_u, x0=u_vec, rtol=1e-5)
        v_next_vec, _ = splinalg.cg(self.A_v_op, rhs_v, x0=v_vec, rtol=1e-5)
        
        return u_next_vec.reshape((self.N, self.N)), v_next_vec.reshape((self.N, self.N))

    def run(self):
        start_time = time.perf_counter()
        
        for i in range(self.steps):
            f, g = Grey_Scott(self.F, self.K, self.u, self.v)
            self.u, self.v = self.step(self.u, self.v, f, g)
            
            if i % 1000 == 0:
                print(f"[{CrankNicolsonScheme.__name__} (Matrix-Free)] Progress: step {i}/{self.steps}")
                
        end_time = time.perf_counter()
        gen_time = end_time - start_time
        return self.v, gen_time