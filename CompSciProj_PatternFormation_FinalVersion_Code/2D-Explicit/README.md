# 2D Reaction-Diffusion System Solver (explicit)

This code constructs a numerical solver based on an explicit Euler model in the time dimension and a two-dimensional **Reaction-Diffusion Systems** in space. It provides corresponding unit tests and interface definitions for the **Gray-Scott Model**, enabling visualization output.

## 1. Mathematical Background

### 1.1 The Governing Equations (PDE)
The evolution of concentrations $u$ and $v$ in a domain $\Omega = [0, 1]^2$ with periodic boundary conditions is described by:

$$\begin{cases}
\partial_{t}u = \delta_{1}\Delta u + f(u, v) \\
\partial_{t}v = \delta_{2}\Delta v + g(u, v)
\end{cases}$$

Where:
* $u, v$: Concentrations of chemical species.
* $\delta_1, \delta_2$: Diffusion coefficients.
* $\Delta = \partial_{xx} + \partial_{yy}$: The Laplacian operator representing diffusion.
* $f, g$: Non-linear reaction terms.

### 1.2 Gray-Scott Model Kinetics
The reaction terms $f$ and $g$ follow the Gray-Scott model:
* $f(u, v) = -uv^2 + F(1 - u)$
* $g(u, v) = uv^2 - (F + k)v$

---

## 2. Numerical Implementation

### 2.1 Spatial Discretization
We utilize the **Second-Order Central Difference** scheme on a uniform grid to approximate the Laplacian:
$$\Delta u_{i,j} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{\Delta x^2}$$

To satisfy the requirement for **periodic boundary conditions**, the implementation uses `numpy.roll` for efficient circular indexing.

### 2.2 Temporal Integration
The solver currently implements the **Explicit Euler (eE)** scheme:
$$u^{k+1} = u^k + \Delta t \cdot \left( \delta_1 \Delta u^k + f(u^k, v^k) \right)$$

---

## 3. Project Structure & Requirements

In alignment with the **PROJECT_4.pdf** assignments:

* **Software Design**: An Object-Oriented Programming (OOP) approach is used.
    * `ReactionModelBase`: Abstract interface for reaction kinetics.
    * `ExplicitScheme`: Handles numerical time-stepping.
    * `ReactionDiffusionSolver`: Manages the simulation loop and state.
* **Documentation**: This README provides the mathematical derivation and usage instructions.
* **Verification (Section 3.2)**: Integrated unit tests verify the accuracy of the Laplacian operator and the temporal update step.

---

## 4. Implementation Details

### 4.1 Global Seeding Strategy (Initialization)
To trigger the reaction-diffusion process, the system requires a departure from the unstable homogeneous equilibrium. Our solver implements a **Global Random Seeding** strategy:
1.  **Base State**: The field $u$ is initialized to $1.0$ and $v$ to $0.0$ across the entire grid.
2.  **Random Injection**: Approximately $1\%$ of the pixels are randomly selected as "reaction centers" where $v$ is injected (set to $0.5$) and $u$ is reduced.
3.  **Symmetry Breaking**: A small amount of Gaussian noise ($\sigma = 0.02$) is added to the entire field to break perfect numerical symmetry, allowing patterns to emerge naturally.

### 4.2 Spatial Discretization
We use a **Second-Order Central Difference** scheme. To satisfy **periodic boundary conditions**, we use `np.roll` for the Laplacian calculation, which effectively treats the edges of the grid as being connected.

### 4.3 Temporal Integration
The solver uses the **Explicit Euler (eE)** scheme:
$$u^{k+1} = u^k + \Delta t \cdot \left( \delta_1 \Delta u^k + f(u^k, v^k) \right)$$

---

## 5. Unit Testing & Verification

In accordance with Section 3.2 of the project assignments, the code includes a test suite to ensure numerical integrity:

### 5.1 Laplacian Accuracy Test (**test_laplacian_2d**)
* **Logic**: Creates a $3 \times 3$ grid with a single impulse (value of $1.0$) at the center.
* **Verification**: Checks if the discrete Laplacian returns $-4.0$ at the center and $+1.0$ at the adjacent neighbors, confirming the $\frac{1}{\Delta x^2}$ scaling and correct neighbor indexing.

### 5.2 Temporal Update Test (**test_explicit_euler_step**)
* **Logic**: Simulates a single time step on a uniform field where the Laplacian is zero. 
* **Verification**: By isolating the reaction terms, it confirms that the solver correctly calculates $u + \Delta t \cdot f(u,v)$, ensuring the Euler integration logic is mathematically sound.

---

## 5. References
Turing, A. M. (1952). "The chemical basis of morphogenesis." Phil. Trans. R. Soc. Lond. B.

Pearson, J. E. (1993). "Complex Patterns in a Simple System." Science.

Kondo, S. and Miura, T. (2010). "Reaction-Diffusion Model as a Framework for Understanding Biological Pattern Formation." Science.

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