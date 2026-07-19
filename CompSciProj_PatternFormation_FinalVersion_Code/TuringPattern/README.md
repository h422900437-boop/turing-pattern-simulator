
# Turing Pattern Simulator

A Python-based simulation project for Turing Patterns. This project solves reaction-diffusion equations to simulate the generation of natural patterns such as leopard spots and giraffe stripes.

## Installation

This project is modularized and supports installation in editable mode. Please ensure you are in the project's root directory when running these commands.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd TuringPattern
```

### 2. Install the Package
Install the project and all its dependencies using `pip`:
```bash
pip install -e .
```
*Note: The `-e` flag installs the package in editable mode, meaning any changes you make to the source code will immediately take effect without needing reinstallation.*

## Running the Simulation

Because the project uses a standard Python package structure, **you must run the scripts from the root directory** to avoid relative import errors.

**Option A: Run the main script**
```bash
python run_sim.py
```

**Option B: Use the console command**
If you have installed the package via `setup.py`, you can launch the simulation directly from your terminal:
```bash
turing-sim
```

## Project Structure

```text
TuringPattern/
├── turing_core/          # Core algorithm package
│   ├── models/           # Model definitions (e.g., Grey-Scott)
│   └── solvers/          # Numerical solvers (e.g., GPU Explicit scheme)
├── run_sim.py            # Main entry point for the simulation
├── setup.py              # Packaging and distribution setup
├── test_cupy.py          # A simple hardware diagnostic script
├── benchmark_profile.py  # Benchmark and analyze the time consumption
├── validate_order.py     # Verify the mathematical accuracy of the model
├── .gitignore            # Git ignore rules
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```
