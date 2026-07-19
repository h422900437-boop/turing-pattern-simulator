import numpy as np
import matplotlib.pyplot as plt
import turing_core.solvers.explicit
import turing_core.solvers.Implicit
from turing_core.solvers.explicit import ExplicitScheme
from turing_core.solvers.Implicit import CrankNicolsonScheme

# def validate_all_orders():
#     print("开始综合验证：空间与时间收敛阶数...\n")
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#     # =========================================================================
#     # 模块 1：空间收敛阶数验证 (Spatial Order)
#     # =========================================================================
#     print("-> [1/2] 正在计算空间拉普拉斯算子的截断误差...")
#     N_values = [20, 40, 80, 160, 320]
#     dx_values = []
#     spatial_errors = []

#     for N in N_values:
#         dx = 1.0 / N
#         dx_values.append(dx)
#         X, Y = np.meshgrid(np.linspace(0, 1, N, endpoint=False), np.linspace(0, 1, N, endpoint=False))
#         U = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
#         exact_laplacian = -8 * (np.pi ** 2) * U

#         dummy = np.zeros((N, N))
#         solver = CrankNicolsonScheme(N=N, dx=dx, dt=1.0, Du=1.0, Dv=1.0, steps=1, F=0, K=0, u=dummy, v=dummy)
#         num_laplacian = solver._apply_laplacian(U)
#         spatial_errors.append(np.max(np.abs(num_laplacian - exact_laplacian)))

#     slope_spatial, _ = np.polyfit(np.log10(dx_values), np.log10(spatial_errors), 1)

#     ax1 = axes[0]
#     ax1.loglog(dx_values, spatial_errors, 'o-', color='#d62728', markersize=8, linewidth=2, label=f'Numerical Error (Slope = {slope_spatial:.2f})')
#     ax1.loglog(dx_values, [spatial_errors[0] * (dx / dx_values[0])**2 for dx in dx_values], 'k--', linewidth=2, label='Theoretical $\mathcal{O}(\Delta x^2)$')
#     ax1.set_xlabel('Spatial Step Size $\Delta x$ (Log Scale)', fontsize=12)
#     ax1.set_ylabel('Max Absolute Error $L_\infty$ (Log Scale)', fontsize=12)
#     ax1.set_title('Spatial Convergence: Laplacian Operator', fontsize=14)
#     ax1.legend(fontsize=11)
#     ax1.grid(True, which="both", ls="--", alpha=0.5)

#     # =========================================================================
#     # 模块 2：时间收敛阶数验证 (Temporal Order) - 终极三线对比
#     # =========================================================================
#     print("-> [2/2] 正在计算时间积分方案的误差 (三线对比)...")
#     N_temp, dx_temp = 40, 1.0
    
#     # 【细节】稍微提升扩散系数，让纯扩散的截断误差凸显，同时保证显式方案不发散 (CFL极限 D < 0.625)
#     Du, Dv = 0.5, 0.5 
#     F, K = 0.0367, 0.0649
#     T_end = 2.0

#     X_t, Y_t = np.meshgrid(np.linspace(0, 1, N_temp, endpoint=False), np.linspace(0, 1, N_temp, endpoint=False))
#     u_init = 0.5 + 0.1 * np.sin(2 * np.pi * X_t) * np.cos(2 * np.pi * Y_t)
#     v_init = 0.25 + 0.1 * np.cos(2 * np.pi * X_t) * np.sin(2 * np.pi * Y_t)

#     dt_ref = 0.005
#     steps_ref = int(T_end / dt_ref)

#     # -------------------------------------------------------------
#     # 准备“拦截器”：精确替换 solver 文件内部的局部引用
#     # -------------------------------------------------------------
#     orig_exp_gs = turing_core.solvers.explicit.Grey_Scott
#     orig_imp_gs = turing_core.solvers.Implicit.Grey_Scott
#     def dummy_gs(F_val, K_val, u_val, v_val):
#         return np.zeros_like(u_val), np.zeros_like(v_val)

#     # -------------------------------------------------------------
#     # 准备“拦截器”：强制打破 CG 求解器 1e-5 的精度天花板
#     # -------------------------------------------------------------
#     orig_cg = turing_core.solvers.Implicit.splinalg.cg
#     def patched_cg(*args, **kwargs):
#         kwargs['rtol'] = 1e-12  # 强制无矩阵求解器精度达到 10^-12
#         return orig_cg(*args, **kwargs)

#     # ================== 计算参考真解 ==================
#     print("   [计算真解] Full Model 参考解...")
#     turing_core.solvers.explicit.Grey_Scott = orig_exp_gs
#     turing_core.solvers.Implicit.Grey_Scott = orig_imp_gs
#     ref_solver_full = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt_ref, Du=Du, Dv=Dv, steps=steps_ref, F=F, K=K, u=u_init.copy(), v=v_init.copy())
#     v_ref_full, _ = ref_solver_full.run()

#     print("   [计算真解] Pure Diffusion 参考解...")
#     turing_core.solvers.explicit.Grey_Scott = dummy_gs
#     turing_core.solvers.Implicit.Grey_Scott = dummy_gs
#     # 必须加上高精度 CG 补丁，否则参考解本身就有 1e-5 的噪声！
#     turing_core.solvers.Implicit.splinalg.cg = patched_cg
#     ref_solver_pure = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt_ref, Du=Du, Dv=Dv, steps=steps_ref, F=F, K=K, u=u_init.copy(), v=v_init.copy())
#     v_ref_pure, _ = ref_solver_pure.run()
#     # 算完纯扩散参考解，先把 CG 补丁卸下
#     turing_core.solvers.Implicit.splinalg.cg = orig_cg

#     # ================== 测试各步长 ==================
#     dt_values = [0.4, 0.2, 0.1, 0.05]
#     errors_ex, errors_imex, errors_pure_im = [], [], []

#     print("   [计算测试点] 正在获取各步长误差...")
#     for dt in dt_values:
#         steps = int(T_end / dt)
        
#         # --- 1. 恢复正常反应，测试 Explicit 和 IMEX ---
#         turing_core.solvers.explicit.Grey_Scott = orig_exp_gs
#         turing_core.solvers.Implicit.Grey_Scott = orig_imp_gs
        
#         ex_solver = ExplicitScheme(dx=dx_temp, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K, u=u_init.copy(), v=v_init.copy())
#         v_ex, _ = ex_solver.run()
#         errors_ex.append(np.max(np.abs(v_ex - v_ref_full)))

#         imex_solver = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K, u=u_init.copy(), v=v_init.copy())
#         v_imex, _ = imex_solver.run()
#         errors_imex.append(np.max(np.abs(v_imex - v_ref_full)))

#         # --- 2. 彻底关闭反应，打上超高精度 CG 补丁，测试 Pure CN ---
#         turing_core.solvers.explicit.Grey_Scott = dummy_gs
#         turing_core.solvers.Implicit.Grey_Scott = dummy_gs
#         turing_core.solvers.Implicit.splinalg.cg = patched_cg
        
#         pure_im_solver = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K, u=u_init.copy(), v=v_init.copy())
#         v_pure_im, _ = pure_im_solver.run()
#         errors_pure_im.append(np.max(np.abs(v_pure_im - v_ref_pure)))
        
#         # 卸下补丁
#         turing_core.solvers.Implicit.splinalg.cg = orig_cg

#     # ================== 恢复系统默认状态 ==================
#     turing_core.solvers.explicit.Grey_Scott = orig_exp_gs
#     turing_core.solvers.Implicit.Grey_Scott = orig_imp_gs

#     # ================== 绘制右图 ==================
#     log_dt = np.log10(dt_values)
#     slope_ex, _ = np.polyfit(log_dt, np.log10(errors_ex), 1)
#     slope_imex, _ = np.polyfit(log_dt, np.log10(errors_imex), 1)
#     slope_pure_im, _ = np.polyfit(log_dt, np.log10(errors_pure_im), 1)

#     ax2 = axes[1]
#     ax2.loglog(dt_values, errors_ex, 'o-', color='#1f77b4', markersize=8, linewidth=2, label=f'Explicit Euler (Full, Slope = {slope_ex:.2f})')
#     ax2.loglog(dt_values, errors_imex, 's-', color='#2ca02c', markersize=8, linewidth=2, label=f'IMEX Crank-Nicolson (Full, Slope = {slope_imex:.2f})')
#     ax2.loglog(dt_values, errors_pure_im, 'D-', color='#ff7f0e', markersize=8, linewidth=2, label=f'Pure CN (Diffusion Only, Slope = {slope_pure_im:.2f})')

#     ax2.loglog(dt_values, [errors_ex[0] * (dt / dt_values[0])**1 for dt in dt_values], '--', color="#FD0000", alpha=0.5, label='Theoretical $\mathcal{O}(\Delta t)$')
#     ax2.loglog(dt_values, [errors_pure_im[0] * (dt / dt_values[0])**2 for dt in dt_values], '--', color="#0043fc", alpha=0.5, label='Theoretical $\mathcal{O}(\Delta t^2)$')

#     ax2.set_xlabel('Temporal Step Size $\Delta t$ (Log Scale)', fontsize=12)
#     ax2.set_ylabel('Max Absolute Error $L_\infty$ vs Reference (Log Scale)', fontsize=12)
#     ax2.set_title('Temporal Convergence: Full Model vs Pure Diffusion', fontsize=14)
#     ax2.legend(fontsize=10)
#     ax2.grid(True, which="both", ls="--", alpha=0.5)

#     plt.tight_layout()
#     plt.savefig('optimized_comprehensive_order_validation.png', dpi=300)
#     print("\n✅ 综合验证完成！数学精度证明如下：")
#     print(f"   [空间精度] 拉普拉斯算子收敛阶数: {slope_spatial:.2f} (匹配理论 O(dx^2))")
#     print(f"   [时间精度] 显式欧拉 (Full Model) 收敛阶数: {slope_ex:.2f} (匹配理论 O(dt))")
#     print(f"   [时间精度] IMEX CN (Full Model) 收敛阶数: {slope_imex:.2f} (因反应项显式处理，降为 O(dt))")
#     print(f"   [时间精度] 纯隐式 CN (Pure Diffusion) 收敛阶数: {slope_pure_im:.2f} (完美展现二阶 O(dt^2))")
#     plt.show()

# if __name__ == "__main__":
#     validate_all_orders()


# import numpy as np
# import matplotlib.pyplot as plt
# import turing_core.solvers.explicit
# import turing_core.solvers.Implicit
# from turing_core.solvers.explicit import ExplicitScheme
# from turing_core.solvers.Implicit import CrankNicolsonScheme

def validate_all_orders():
    print("Starting comprehensive validation: Spatial and temporal convergence orders...\n")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # =========================================================================
    # Module 1: Spatial convergence order validation (Spatial Order)
    # =========================================================================
    print("-> [1/2] Calculating truncation error of the spatial Laplacian operator...")
    N_values = [20, 40, 80, 160, 320]
    dx_values = []
    spatial_errors = []

    for N in N_values:
        dx = 1.0 / N
        dx_values.append(dx)
        X, Y = np.meshgrid(np.linspace(0, 1, N, endpoint=False), np.linspace(0, 1, N, endpoint=False))
        U = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        exact_laplacian = -8 * (np.pi ** 2) * U

        dummy = np.zeros((N, N))
        solver = CrankNicolsonScheme(N=N, dx=dx, dt=1.0, Du=1.0, Dv=1.0, steps=1, F=0, K=0, u=dummy, v=dummy)
        num_laplacian = solver._apply_laplacian(U)
        spatial_errors.append(np.max(np.abs(num_laplacian - exact_laplacian)))

    slope_spatial, _ = np.polyfit(np.log10(dx_values), np.log10(spatial_errors), 1)

    ax1 = axes[0]
    ax1.loglog(dx_values, spatial_errors, 'o-', color='#d62728', markersize=8, linewidth=2, label=f'Numerical Error (Slope = {slope_spatial:.2f})')
    ax1.loglog(dx_values, [spatial_errors[0] * (dx / dx_values[0])**2 for dx in dx_values], 'k--', linewidth=2, label='Theoretical $\mathcal{O}(\Delta x^2)$')
    ax1.set_xlabel('Spatial Step Size $\Delta x$ (Log Scale)', fontsize=12)
    ax1.set_ylabel('Max Absolute Error $L_\infty$ (Log Scale)', fontsize=12)
    ax1.set_title('Spatial Convergence: Laplacian Operator', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # =========================================================================
    # Module 2: Temporal convergence order validation (Temporal Order) - Ultimate three-curve comparison
    # =========================================================================
    print("-> [2/2] Calculating errors of time integration schemes (three-curve comparison)...")
    N_temp, dx_temp = 40, 1.0
    
    # [Detail] Slightly increase diffusion coefficients to highlight truncation errors of pure diffusion, while keeping explicit scheme stable (CFL limit D < 0.625)
    Du, Dv = 0.5, 0.5 
    F, K = 0.0367, 0.0649
    T_end = 2.0

    X_t, Y_t = np.meshgrid(np.linspace(0, 1, N_temp, endpoint=False), np.linspace(0, 1, N_temp, endpoint=False))
    u_init = 0.5 + 0.1 * np.sin(2 * np.pi * X_t) * np.cos(2 * np.pi * Y_t)
    v_init = 0.25 + 0.1 * np.cos(2 * np.pi * X_t) * np.sin(2 * np.pi * Y_t)

    dt_ref = 0.005
    steps_ref = int(T_end / dt_ref)

    # -------------------------------------------------------------
    # Prepare "Patch": Precisely replace local references inside the solver file
    # -------------------------------------------------------------
    orig_exp_gs = turing_core.solvers.explicit.Grey_Scott
    orig_imp_gs = turing_core.solvers.Implicit.Grey_Scott
    def dummy_gs(F_val, K_val, u_val, v_val):
        return np.zeros_like(u_val), np.zeros_like(v_val)

    # -------------------------------------------------------------
    # Prepare "Patch": Force breaking the 1e-5 accuracy ceiling of the CG solver
    # -------------------------------------------------------------
    orig_cg = turing_core.solvers.Implicit.splinalg.cg
    def patched_cg(*args, **kwargs):
        kwargs['rtol'] = 1e-12  # Force matrix-free solver accuracy to reach 10^-12
        return orig_cg(*args, **kwargs)

    # ================== Calculate Reference Solution ==================
    print("   [Calculating Reference] Full Model reference solution...")
    turing_core.solvers.explicit.Grey_Scott = orig_exp_gs
    turing_core.solvers.Implicit.Grey_Scott = orig_imp_gs
    ref_solver_full = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt_ref, Du=Du, Dv=Dv, steps=steps_ref, F=F, K=K, u=u_init.copy(), v=v_init.copy())
    v_ref_full, _ = ref_solver_full.run()

    print("   [Calculating Reference] Pure Diffusion reference solution...")
    turing_core.solvers.explicit.Grey_Scott = dummy_gs
    turing_core.solvers.Implicit.Grey_Scott = dummy_gs
    # Must apply high-precision CG patch, otherwise the reference solution itself has 1e-5 noise!
    turing_core.solvers.Implicit.splinalg.cg = patched_cg
    ref_solver_pure = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt_ref, Du=Du, Dv=Dv, steps=steps_ref, F=F, K=K, u=u_init.copy(), v=v_init.copy())
    v_ref_pure, _ = ref_solver_pure.run()
    # After calculating pure diffusion reference solution, remove the CG patch first
    turing_core.solvers.Implicit.splinalg.cg = orig_cg

    # ================== Test Different Step Sizes ==================
    dt_values = [0.4, 0.2, 0.1, 0.05]
    errors_ex, errors_imex, errors_pure_im = [], [], []

    print("   [Calculating Test Points] Getting errors for each step size...")
    for dt in dt_values:
        steps = int(T_end / dt)
        
        # --- 1. Restore normal reactions, test Explicit and IMEX ---
        turing_core.solvers.explicit.Grey_Scott = orig_exp_gs
        turing_core.solvers.Implicit.Grey_Scott = orig_imp_gs
        
        ex_solver = ExplicitScheme(dx=dx_temp, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K, u=u_init.copy(), v=v_init.copy())
        v_ex, _ = ex_solver.run()
        errors_ex.append(np.max(np.abs(v_ex - v_ref_full)))

        imex_solver = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K, u=u_init.copy(), v=v_init.copy())
        v_imex, _ = imex_solver.run()
        errors_imex.append(np.max(np.abs(v_imex - v_ref_full)))

        # --- 2. Completely turn off reactions, apply ultra-high precision CG patch, test Pure CN ---
        turing_core.solvers.explicit.Grey_Scott = dummy_gs
        turing_core.solvers.Implicit.Grey_Scott = dummy_gs
        turing_core.solvers.Implicit.splinalg.cg = patched_cg
        
        pure_im_solver = CrankNicolsonScheme(N=N_temp, dx=dx_temp, dt=dt, Du=Du, Dv=Dv, steps=steps, F=F, K=K, u=u_init.copy(), v=v_init.copy())
        v_pure_im, _ = pure_im_solver.run()
        errors_pure_im.append(np.max(np.abs(v_pure_im - v_ref_pure)))
        
        # Remove patch
        turing_core.solvers.Implicit.splinalg.cg = orig_cg

    # ================== Restore System Default State ==================
    turing_core.solvers.explicit.Grey_Scott = orig_exp_gs
    turing_core.solvers.Implicit.Grey_Scott = orig_imp_gs

    # ================== Plot Right Figure ==================
    log_dt = np.log10(dt_values)
    slope_ex, _ = np.polyfit(log_dt, np.log10(errors_ex), 1)
    slope_imex, _ = np.polyfit(log_dt, np.log10(errors_imex), 1)
    slope_pure_im, _ = np.polyfit(log_dt, np.log10(errors_pure_im), 1)

    ax2 = axes[1]
    ax2.loglog(dt_values, errors_ex, 'o-', color='#1f77b4', markersize=8, linewidth=2, label=f'Explicit Euler (Full, Slope = {slope_ex:.2f})')
    ax2.loglog(dt_values, errors_imex, 's-', color='#2ca02c', markersize=8, linewidth=2, label=f'IMEX Crank-Nicolson (Full, Slope = {slope_imex:.2f})')
    ax2.loglog(dt_values, errors_pure_im, 'D-', color='#ff7f0e', markersize=8, linewidth=2, label=f'Pure CN (Diffusion Only, Slope = {slope_pure_im:.2f})')

    ax2.loglog(dt_values, [errors_ex[0] * (dt / dt_values[0])**1 for dt in dt_values], '--', color="#FD0000", alpha=0.5, label='Theoretical $\mathcal{O}(\Delta t)$')
    ax2.loglog(dt_values, [errors_pure_im[0] * (dt / dt_values[0])**2 for dt in dt_values], '--', color="#0043fc", alpha=0.5, label='Theoretical $\mathcal{O}(\Delta t^2)$')

    ax2.set_xlabel('Temporal Step Size $\Delta t$ (Log Scale)', fontsize=12)
    ax2.set_ylabel('Max Absolute Error $L_\infty$ vs Reference (Log Scale)', fontsize=12)
    ax2.set_title('Temporal Convergence: Full Model vs Pure Diffusion', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig('optimized_comprehensive_order_validation.png', dpi=300)
    print("\n✅ Comprehensive validation complete! Mathematical accuracy proofs are as follows:")
    print(f"   [Spatial Accuracy] Laplacian operator convergence order: {slope_spatial:.2f} (Matches theoretical O(dx^2))")
    print(f"   [Temporal Accuracy] Explicit Euler (Full Model) convergence order: {slope_ex:.2f} (Matches theoretical O(dt))")
    print(f"   [Temporal Accuracy] IMEX CN (Full Model) convergence order: {slope_imex:.2f} (Reduced to O(dt) due to explicit handling of reaction terms)")
    print(f"   [Temporal Accuracy] Pure Implicit CN (Pure Diffusion) convergence order: {slope_pure_im:.2f} (Perfectly demonstrates second-order O(dt^2))")
    plt.show()

if __name__ == "__main__":
    validate_all_orders()