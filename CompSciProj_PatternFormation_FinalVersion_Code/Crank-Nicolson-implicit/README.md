# Crank-Nicolson Solver for 2D Reaction-Diffusion Systems

This solver implements the **Crank-Nicolson (CN)** method to simulate 2D Reaction-Diffusion equations. It strikes a balance between the high stability of implicit methods and the computational efficiency of optimized linear algebra.

---

## 1. Physical Model & Spatial Discretization

The system solves the partial differential equation (PDE) as derived in the physical model:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + f(u, v)$$

### Discrete Laplacian via Kronecker Product
In 2D space, the Laplacian operator $\nabla^2 u$ is approximated using the **5-point central difference stencil**:

$$\nabla^2 u \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}$$



To implement this efficiently without nested loops, we use the **Kronecker Product ($\otimes$)**. By combining a 1D second-derivative matrix $D_{1D}$ with an identity matrix $I$, we construct a global sparse matrix $L$:

$$L = \frac{1}{\Delta x^2} (I \otimes D_{1D} + D_{1D} \otimes I)$$

**Logic:**
* $I \otimes D_{1D}$ handles the derivatives in the $x$-direction.
* $D_{1D} \otimes I$ handles the derivatives in the $y$-direction.
* This creates a sparse matrix $L$ that maps the relationship between every pixel and its four neighbors in a flattened vector.

---

## 2. Temporal Discretization: The Crank-Nicolson Scheme

The solver uses the **Crank-Nicolson** method, which averages the diffusion term at the current time step ($n$) and the future time step ($n+1$):

$$\frac{u^{n+1} - u^n}{\Delta t} = \frac{D}{2} \left[ L u^{n+1} + L u^n \right] + f(u^n)$$



By rearranging terms to solve for the unknown $u^{n+1}$, we transform the PDE into a linear system:

$$\left( I - \frac{\Delta t D}{2} L \right) u^{n+1} = \left( I + \frac{\Delta t D}{2} L \right) u^n + \Delta t \cdot f^n$$

This is implemented in the code as: **$A \cdot u^{n+1} = B \cdot u^n + \Delta t \cdot f^n$**.

---

## 3. Code Implementation Logic

The solver is designed for high-performance real-time simulation using a pre-calculated linear algebra strategy.

### A. Matrix Initialization & Pre-Factorization
Since the diffusion coefficient $D$, grid spacing $\Delta x$, and time step $\Delta t$ are constant, the matrix $A = (I - \frac{\Delta t D}{2} L)$ remains unchanged throughout the simulation.
* **The "Master Key"**: During the `__init__` phase, we use `splinalg.factorized(A)` to perform an **LU Decomposition** of matrix $A$ once.
* **Efficiency**: The decomposed form is stored as `self.solve_u`, allowing for near-instantaneous solving in the main loop.



### B. The Time-Stepping Loop (`step`)
In every iteration, the program performs:
1.  **Calculate RHS**: Computes $rhs = B \cdot u^n + \Delta t \cdot f^n$.
2.  **Solve**: Executes `u_next = self.solve_u(rhs)`. This is a "back-substitution" step, which is computationally trivial compared to solving a full system from scratch.

### C. Why not the Jacobi Iterative Method?
While the **Jacobi Method** is a common introductory approach (splitting $A$ into $D+L+U$ and iterating), it was rejected for this implementation:
* **Efficiency**: Jacobi is an iterative solver; it requires hundreds of loops per frame to reach a solution, causing significant lag.
* **Convergence**: Jacobi may fail to converge if the time step $\Delta t$ is too large, leading to numerical instability (NaN results).
* **Direct Advantage**: Since our matrix $A$ is time-invariant, pre-factorization allows us to "unlock" the exact solution in a single step, maintaining "unconditional stability" while running at high frame rates.

---

## 4. Summary of Benefits
* **Accuracy**: Second-order accuracy in both time $O(\Delta t^2)$ and space $O(\Delta x^2)$.
* **Stability**: Avoids the strict time-step limits of explicit schemes.
* **Performance**: Sparse matrix operations and LU pre-factorization enable complex pattern formation (like Turing patterns) to emerge in real-time.

## 03-10-26
## Giraffe Spot Characteristics
Giraffe spots resemble a "polygonal grid." They typically consist of narrow, light-colored lines (low V concentration areas) and large, dark patches (high V concentration areas). The giraffe's fur pattern is formed during a specific stage of embryonic development, when melanocytes throughout the body undergo changes almost simultaneously.

## Initialization Seeding Strategy
Clearly, adding perturbations only at the center of the grid cannot produce a uniform fur pattern covering the entire grid. Using global random initialization more closely resembles the natural state and is more likely to generate patterns. 

Furthermore, through multiple parameter tunings, it was found that the traditional method of initial concentration $u=1, v=0$, plus a global initial perturbation, is unlikely to produce giraffe-like spots through long-term evolution. High-concentration areas cannot merge into large spots. Only when the initial concentration is set to **$u=0.5, v=0.3$**, supplemented by random perturbation, can ideal giraffe spots be stably obtained. 

* **Rationale:** The values of $u=0.5, v=0.3$ allow the system to directly cross the activation threshold and enter the steady-state diffusion region. 
* **Merging Mechanism:** At this initial concentration, the high concentration of substance $V$ in the system causes the generated Turing spots to expand rapidly and physically merge. This merging mechanism is the mathematical key to the formation of large, giraffe-like patches rather than small, leopard-like dots.

## Reaction Coefficients
We directly referenced the coefficients of formula (13) in the paper *"The Bifurcation Growth Rate for the Robust Pattern Formation in the Reaction-Diffusion System on the Growing Domain"* with:
* $d_u = 0.2, d_v = 0.1, F = 0.089, K = 0.06$

However, the resulting pattern was small and there was still some adhesion between the patterns. Therefore, we kept the ratio of $d_u/d_v = 2$ and increased it to the coefficients of **$d_u = 0.6, d_v = 0.3$**. We found that we could get a better result of giraffe pattern, but there were still many small patches. This was more obvious when the pattern was generated using the CN implicit model. When the coefficients were increased to **$d_u = 0.8, d_v = 0.4$**, a more uniform pattern was finally obtained.

## Reference
*"The Bifurcation Growth Rate for the Robust Pattern Formation in the Reaction-Diffusion System on the Growing Domain"*

Paper citation: arXiv:2407.17217 

Paper URL: https://arxiv.org/abs/2407.17217