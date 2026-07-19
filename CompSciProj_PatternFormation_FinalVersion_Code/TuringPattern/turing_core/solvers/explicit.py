# import numpy as np
# import time
# from ..models import Grey_Scott

# class ExplicitScheme:
#     """combination 1: Explicit time stepping + second-order finite difference"""
#     def __init__(self, dx, dt, Du, Dv, steps,F,K,u,v):
#         #dx is the spatial step size, dt is the time step size, Du and Dv are diffusion coefficients δ
#         self.dx2 = dx ** 2
#         self.dt = dt
#         self.Du = Du
#         self.Dv = Dv
#         self.steps = steps
#         self.F = F
#         self.K = K
#         self.u = u
#         self.v = v

#     def laplacian_2d(self, field):
#         """Second order finite difference approximation for the Laplacian"""
#         dx2 = self.dx2
#         #lap=(u_i+1,j + u_i-1,j+u_i,j+1 + u_i,j-1 - 4u_i,j) / dx^2；
#         #np.roll can also used to implement the periodic boundary conditions
#         lap = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
#                np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
#                4.0 * field) / dx2
#         return lap

#     def step(self, u, v, f, g):
#         """Explicit time stepping"""
       
#         dt = self.dt
#         Du = self.Du
#         Dv = self.Dv
        
#         lap_u = self.laplacian_2d(u)
#         lap_v = self.laplacian_2d(v)
        
#         # 完美对应白板推导形式：u^{k+1} = u^k + dt * (D * Laplacian(u) + f(u))
#         # This perfectly matches our derivation: u^{k+1} = u^k + dt * (D * Laplacian(u) + f(u))
#         u_next = u + dt * (Du * lap_u + f)
#         v_next = v + dt * (Dv * lap_v + g)
        
#         return u_next, v_next
    
#     def run(self):
#         start_time = time.perf_counter()#record the start time for performance measurement
#         print(type(self.u), type(self.v))
#         for i in range(self.steps):
#             f, g = Grey_Scott(self.F, self.K, self.u, self.v)#compute the reaction terms based on the current state of u and v
#             self.u, self.v = self.step(self.u, self.v, f, g)#update u and v using the explicit time stepping scheme
            
#             if i % 1000 == 0:
#                 print(f"[{ExplicitScheme}] Progress: step {i}/{self.steps}")#periodically print progress every 1000 steps to monitor the simulation
                
#         end_time = time.perf_counter()#record the end time and calculate the total generation time for the simulation
#         gen_time = end_time - start_time
#         return self.v, gen_time

# import numpy as np
# import cupy as cp
# import time
# from ..models import Grey_Scott

# class ExplicitScheme:
#     """使用 CuPy 驱动的 GPU 显式欧拉方案"""
#     def __init__(self, dx, dt, Du, Dv, steps, F, K, u, v):
#         self.dx2 = cp.float32(dx ** 2)
#         self.dt = cp.float32(dt)
#         self.Du = cp.float32(Du)
#         self.Dv = cp.float32(Dv)
#         self.steps = steps
#         self.F = cp.float32(F)
#         self.K = cp.float32(K)
        
#         # 确保网格大小为 N x N
#         self.N = cp.int32(u.shape[0])
        
#         # CuPy 大管家自动接管：将 CPU 内存 (NumPy) 直接转入 GPU 显存 (CuPy)
#         # 这一步就像把原材料直接用货车拉进显卡的超级工厂里
#         self.u_gpu = cp.asarray(u, dtype=cp.float32)
#         self.v_gpu = cp.asarray(v, dtype=cp.float32)
#         # 在显存中开辟两个空位用于存放下一秒的结果
#         self.u_next_gpu = cp.empty_like(self.u_gpu)
#         self.v_next_gpu = cp.empty_like(self.v_gpu)

#         # 使用 CuPy 的 RawKernel 注入 C 语言核心代码
#         # 注意：这里的 C 代码最前面加上了 extern "C"，这是 CuPy 编译所需的标准通行证
#         self.step_kernel = cp.RawKernel(r'''
#         extern "C" __global__ void reaction_diffusion_step(
#             const float* u, const float* v, float* u_next, float* v_next,
#             float Du, float Dv, float F, float K, float dt, float dx2, int N) 
#         {
#             // 获取当前线程在二维网格中的 x 和 y 坐标
#             int x = threadIdx.x + blockIdx.x * blockDim.x;
#             int y = threadIdx.y + blockIdx.y * blockDim.y;
            
#             // 如果坐标超出了网格范围，直接退出
#             if (x >= N || y >= N) return;
            
#             // 使用取余数运算 (%) 实现周期性边界条件
#             int x_up = (x - 1 + N) % N;
#             int x_down = (x + 1) % N;
#             int y_left = (y - 1 + N) % N;
#             int y_right = (y + 1) % N;
            
#             int idx = y * N + x;
            
#             float u_center = u[idx];
#             float v_center = v[idx];
            
#             // 计算拉普拉斯 (Laplacian)
#             float lap_u = (u[y * N + x_up] + u[y * N + x_down] + u[y_left * N + x] + u[y_right * N + x] - 4.0 * u_center) / dx2;
#             float lap_v = (v[y * N + x_up] + v[y * N + x_down] + v[y_left * N + x] + v[y_right * N + x] - 4.0 * v_center) / dx2;
            
#             // 计算 Grey-Scott 反应项
#             float uvv = u_center * v_center * v_center;
#             float f = -uvv + F * (1.0 - u_center);
#             float g = uvv - (F + K) * v_center;
            
#             // 显式欧拉步进
#             u_next[idx] = u_center + dt * (Du * lap_u + f);
#             v_next[idx] = v_center + dt * (Dv * lap_v + g);
#         }
#         ''', 'reaction_diffusion_step')
        
#         # 配置 GPU 的方阵划分 (16x16=256个线程组成一个小队)
#         self.block_size = (16, 16, 1)
#         grid_x = int(np.ceil(float(self.N) / self.block_size[0]))
#         grid_y = int(np.ceil(float(self.N) / self.block_size[1]))
#         self.grid_size = (grid_x, grid_y, 1)

#     def run(self):
#         start_time = time.perf_counter()
        
#         for i in range(self.steps):
#             # 将指挥棒交给 GPU Kernel，传参极其简洁
#             self.step_kernel(
#                 self.grid_size, 
#                 self.block_size,
#                 (self.u_gpu, self.v_gpu, self.u_next_gpu, self.v_next_gpu,
#                  self.Du, self.Dv, self.F, self.K, self.dt, self.dx2, self.N)
#             )
            
#             # Python 层面仅做引用的交换，不移动物理显存的数据，速度达到纳秒级
#             self.u_gpu, self.u_next_gpu = self.u_next_gpu, self.u_gpu
#             self.v_gpu, self.v_next_gpu = self.v_next_gpu, self.v_gpu
            
#             if i % 1000 == 0:
#                 print(f"[CuPy Explicit] Progress: step {i}/{self.steps}")
                
#         # 计算全部完成后，通知大管家将货物从显存装车运回主板 CPU (asnumpy)
#         final_v = cp.asnumpy(self.v_gpu)
        
#         end_time = time.perf_counter()
#         gen_time = end_time - start_time
        
#         return final_v.astype(np.float64), gen_time


import numpy as np
import cupy as cp
import time
from ..models import Grey_Scott

class ExplicitScheme:
    """GPU-accelerated explicit Euler scheme driven by CuPy"""
    def __init__(self, dx, dt, Du, Dv, steps, F, K, u, v):
        self.dx2 = cp.float32(dx ** 2)
        self.dt = cp.float32(dt)
        self.Du = cp.float32(Du)
        self.Dv = cp.float32(Dv)
        self.steps = steps
        self.F = cp.float32(F)
        self.K = cp.float32(K)
        
        # Ensure the grid size is N x N
        self.N = cp.int32(u.shape[0])
        
        # CuPy takes over: transfer data directly from CPU memory (NumPy) to GPU memory (CuPy)
        # This is like shipping raw materials directly to the GPU's super factory
        self.u_gpu = cp.asarray(u, dtype=cp.float32)
        self.v_gpu = cp.asarray(v, dtype=cp.float32)
        # Allocate two empty arrays in GPU memory for the next time step's results
        self.u_next_gpu = cp.empty_like(self.u_gpu)
        self.v_next_gpu = cp.empty_like(self.v_gpu)

        # Inject core C code using CuPy's RawKernel
        # Note: 'extern "C"' is added at the beginning, which is required for CuPy compilation
        self.step_kernel = cp.RawKernel(r'''
        extern "C" __global__ void reaction_diffusion_step(
            const float* u, const float* v, float* u_next, float* v_next,
            float Du, float Dv, float F, float K, float dt, float dx2, int N) 
        {
            // Get the x and y coordinates of the current thread in the 2D grid
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            
            // Return immediately if coordinates are out of grid bounds
            if (x >= N || y >= N) return;
            
            // Implement periodic boundary conditions using the modulo operator (%)
            int x_up = (x - 1 + N) % N;
            int x_down = (x + 1) % N;
            int y_left = (y - 1 + N) % N;
            int y_right = (y + 1) % N;
            
            int idx = y * N + x;
            
            float u_center = u[idx];
            float v_center = v[idx];
            
            // Calculate Laplacian
            float lap_u = (u[y * N + x_up] + u[y * N + x_down] + u[y_left * N + x] + u[y_right * N + x] - 4.0 * u_center) / dx2;
            float lap_v = (v[y * N + x_up] + v[y * N + x_down] + v[y_left * N + x] + v[y_right * N + x] - 4.0 * v_center) / dx2;
            
            // Calculate Grey-Scott reaction terms
            float uvv = u_center * v_center * v_center;
            float f = -uvv + F * (1.0 - u_center);
            float g = uvv - (F + K) * v_center;
            
            // Explicit Euler time stepping
            u_next[idx] = u_center + dt * (Du * lap_u + f);
            v_next[idx] = v_center + dt * (Dv * lap_v + g);
        }
        ''', 'reaction_diffusion_step')
        
        # Configure GPU block dimensions (16x16=256 threads per block)
        self.block_size = (16, 16, 1)
        grid_x = int(np.ceil(float(self.N) / self.block_size[0]))
        grid_y = int(np.ceil(float(self.N) / self.block_size[1]))
        self.grid_size = (grid_x, grid_y, 1)

    def run(self):
        start_time = time.perf_counter()
        
        for i in range(self.steps):
            # Execute the GPU Kernel with concise parameter passing
            self.step_kernel(
                self.grid_size, 
                self.block_size,
                (self.u_gpu, self.v_gpu, self.u_next_gpu, self.v_next_gpu,
                 self.Du, self.Dv, self.F, self.K, self.dt, self.dx2, self.N)
            )
            
            # Swap references at the Python level without moving physical data in GPU memory, achieving nanosecond speeds
            self.u_gpu, self.u_next_gpu = self.u_next_gpu, self.u_gpu
            self.v_gpu, self.v_next_gpu = self.v_next_gpu, self.v_gpu
            
            if i % 1000 == 0:
                print(f"[CuPy Explicit] Progress: step {i}/{self.steps}")
                
        # Once computation is complete, transfer the results from GPU memory back to CPU memory (asnumpy)
        final_v = cp.asnumpy(self.v_gpu)
        
        end_time = time.perf_counter()
        gen_time = end_time - start_time
        
        return final_v.astype(np.float64), gen_time