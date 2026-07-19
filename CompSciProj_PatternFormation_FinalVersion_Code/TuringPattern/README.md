# Turing Pattern Simulator

A high-performance Python package for simulating Turing patterns using reaction-diffusion equations. This project generates natural patterns such as leopard spots and giraffe stripes through the Grey-Scott model, with support for both GPU-accelerated and high-precision numerical solving schemes.

## Features

- **Dual Solving Schemes**:
  - **Explicit Euler**: GPU-accelerated using CuPy with CUDA kernels (fast, O(Δt) accuracy)
  - **Crank-Nicolson (Implicit)**: Matrix-free solver with conjugate gradient method (O(Δt²) accuracy)

- **Pattern Presets**:
  - **Leopard Pattern**: Spot-like formations with fine-tuned diffusion coefficients
  - **Giraffe Pattern**: Stripe-like structures with different dynamics

- **Comprehensive Tools**:
  - GPU hardware diagnostics
  - Numerical convergence verification
  - Performance benchmarking and profiling
  - Unit tests for algorithm correctness

- **Optimized Implementation**:
  - CUDA kernel integration for explicit solver
  - Memory-efficient matrix-free implicit solver
  - Automatic memory management with garbage collection
  - Periodic boundary conditions

## Installation

### Prerequisites

- Python 3.7+
- CUDA Toolkit 11.x or 12.x (for GPU acceleration)
- pip

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd TuringPattern
```

### Step 2: Install Dependencies

#### For GPU Users (Recommended)

If you have CUDA 12.x:
```bash
pip install -e .
```

If you have CUDA 11.x, edit `requirements.txt` and change:
```
cupy-cuda11x>=12.9.0  # instead of cupy-cuda12x
```

Then install:
```bash
pip install -e .
```

#### For CPU-Only Users

If GPU is unavailable, you can use the implicit solver (which doesn't require CuPy):
```bash
# Comment out the cupy line in requirements.txt
pip install -e .
```

**Note**: The `-e` flag enables editable mode, so code changes take effect immediately without reinstallation.

## Quick Start

### Run All Experiments

Execute all four combinations (2 patterns × 2 methods):

```bash
python run_sim.py
```

This will generate and display:
- Leopard + Explicit (GPU)
- Leopard + Implicit (high precision)
- Giraffe + Explicit (GPU)
- Giraffe + Implicit (high precision)

### Use in Python Code

```python
from turing_core.interface import run_simulation
from turing_core.visual import SolverVisualizer

# Run simulation
v_matrix, gen_time = run_simulation(pattern="leopard", method="explicit")

# Display result
SolverVisualizer.show_result(v_matrix, gen_time, "Leopard Pattern - Explicit Solver")
```

### Command-Line Interface

After installation, you can also run simulations via console (note: requires entry point fix):
```bash
turing-sim
```

## Detailed Usage

### Pattern Selection

```python
# Available patterns
pattern = "leopard"   # Spot-like formations
# or
pattern = "giraffe"   # Stripe-like formations
```

### Solver Selection

```python
# Available methods
method = "explicit"   # GPU-accelerated, fast but lower accuracy (O(Δt))
# or
method = "implicit"   # CPU-based, higher accuracy (O(Δt²))
```

### Parameter Customization

Modify parameters in `turing_core/models.py`:

```python
def leopard_model():
    return {
        'N': 200,           # Grid size (200×200)
        'dx': 1.0,          # Spatial step size
        'dt': 1.0,          # Time step size
        'steps': 10000,     # Number of iterations
        'Du': 0.16,         # Diffusion coefficient for u
        'Dv': 0.08,         # Diffusion coefficient for v
        'F': 0.0367,        # Feed rate
        'K': 0.0649,        # Kill rate
    }
```

## Project Structure

```text
TuringPattern/
├── turing_core/
│   ├── __init__.py                    # Package initialization
│   ├── interface.py                   # Main simulation interface (run_simulation)
│   ├── models.py                      # Grey-Scott model & parameters
│   ├── visual.py                      # Visualization (SolverVisualizer)
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── explicit.py                # GPU-accelerated explicit Euler solver
│   │   └── Implicit.py                # Matrix-free Crank-Nicolson solver
│   └── seeding/
│       ├── __init__.py
│       ├── Leo_seeding.py             # Leopard pattern initialization
│       └── Giraffe_seeding.py         # Giraffe pattern initialization
├── run_sim.py                         # Main entry point (runs all 4 experiments)
├── test_cupy.py                       # GPU hardware diagnostics
├── benchmark_profile.py               # Performance benchmarking & profiling
├── validate_order.py                  # Numerical convergence verification
├── unittest.py                        # Unit tests for solvers
├── setup.py                           # Package configuration
├── requirements.txt                   # Dependency list
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

## Algorithm Details

### Explicit Scheme (GPU-Accelerated)

- **Time Integration**: Explicit Euler
- **Spatial Discretization**: 2nd-order finite differences
- **Implementation**: CUDA kernels via CuPy
- **Grid Configuration**: 16×16 thread blocks
- **Accuracy**: O(Δx²) spatial, O(Δt) temporal
- **Advantage**: Fast execution on modern GPUs
- **Limitation**: Requires CFL stability condition

### Implicit Scheme (High-Precision)

- **Time Integration**: Crank-Nicolson (IMEX)
- **Spatial Discretization**: 2nd-order finite differences
- **Linear Solver**: Conjugate Gradient (CG) method
- **Matrix Type**: Matrix-free (no explicit matrix construction)
- **Accuracy**: O(Δx²) spatial, O(Δt²) temporal (diffusion)
- **Advantage**: Unconditionally stable, higher accuracy
- **Limitation**: Slower computation due to iterative solving

### Boundary Conditions

Both solvers use periodic boundary conditions (toroidal topology), implemented via modulo arithmetic:
```python
x_up = (x - 1 + N) % N
x_down = (x + 1) % N
```

## Tools & Utilities

### GPU Diagnostics

Check if CuPy and CUDA are properly installed:

```bash
python test_cupy.py
```

Output example:
```
Attempting to wake up the GPU...
✅ CuPy installed successfully! GPU communication is fully open!
Device in use: NVIDIA GeForce RTX 3080
```

### Performance Benchmarking

Generate a 2×2 comparison plot of all four method combinations with runtimes:

```bash
python benchmark_profile.py
```

This will:
- Run all 4 experiments
- Generate `benchmark_results.png`
- Print profiling statistics for top 15 time-consuming functions
- Identify performance bottlenecks

### Numerical Validation

Verify spatial and temporal convergence orders match theoretical predictions:

```bash
python validate_order.py
```

Generates `optimized_comprehensive_order_validation.png` showing:
- Spatial convergence: Laplacian operator (should be O(Δx²))
- Temporal convergence: Three solver comparisons
  - Explicit Euler (Full model): O(Δt)
  - IMEX Crank-Nicolson (Full model): O(Δt)
  - Pure CN (Diffusion only): O(Δt²)

### Unit Tests

Run algorithm correctness tests:

```bash
python -m unittest unittest.py
```

Tests cover:
- 2nd-order finite difference approximation
- Explicit Euler time stepping correctness
- Sparse matrix Laplacian operator
- Periodic boundary condition implementation

## Performance Characteristics

### Typical Execution Times (on RTX 3080)

| Pattern | Method | Grid | Steps | Time |
|---------|--------|------|-------|------|
| Leopard | Explicit | 200×200 | 10,000 | ~2-3s |
| Leopard | Implicit | 200×200 | 10,000 | ~30-40s |
| Giraffe | Explicit | 200×200 | 50,000 | ~10-15s |
| Giraffe | Implicit | 200×200 | 50,000 | ~150-180s |

*Note: Times vary depending on hardware. Use `benchmark_profile.py` to measure your system.*

## Dependencies

- **numpy** ≥ 1.20.0 - Numerical computing
- **scipy** ≥ 1.7.0 - Scientific algorithms (sparse matrices, linear solvers)
- **matplotlib** ≥ 3.4.0 - Visualization
- **cupy-cuda12x** ≥ 12.9.0 - GPU acceleration (optional, for explicit solver)

See `requirements.txt` for full specifications.

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'cupy'`

**Solution**: Install CuPy for your CUDA version:
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# Or use CPU-only (implicit solver only)
# Comment out cupy in requirements.txt
```

### Issue: GPU memory error (OOM)

**Solution**: Reduce grid size in models.py:
```python
'N': 100,  # Reduce from 200 to 100
```

### Issue: Slow implicit solver

**Solution**: The Crank-Nicolson solver is inherently slower due to iterative linear solving. Use explicit solver if speed is critical.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{turing_pattern_2025,
  title={Turing Pattern Simulator: GPU-Accelerated Reaction-Diffusion Simulation},
  author={Merten, Julius and Lim, Seongmook and Kim, Soonjung and Li, Weiqi and Huang, Guowei and Elqamel, Ahmed Hamed and Guo, Ziheng},
  year={2025},
  institution={Free University Berlin},
  url={https://github.com/your-org/turing-pattern-simulator}
}
```

## References

1. Turing, A. M. (1952). "The Chemical Basis of Morphogenesis". Philosophical Transactions of the Royal Society B, 237(641).
2. Grey-Scott Model: https://en.wikipedia.org/wiki/Gray%E2%80%93Scott_model
3. CuPy Documentation: https://docs.cupy.dev/

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributors

- Julius Merten
- Seongmook Lim
- Soonjung Kim
- Weiqi Li
- Guowei Huang
- Ahmed Hamed Elqamel
- Ziheng Guo

Department of Mathematics, Free University Berlin (2025)
