from turing_core.interface import run_simulation
from turing_core.visual import SolverVisualizer

# experiment 1: Leopard + Explicit
v1,gen_time1 = run_simulation(pattern="leopard", method="explicit")
# experiment 2: Leopard + Implicit
v2,gen_time2 = run_simulation(pattern="leopard", method="implicit")
# experiment 3: Giraffe + Explicit
v3,gen_time3 = run_simulation(pattern="giraffe", method="explicit")   
# experiment 4: Giraffe + Implicit
v4,gen_time4 = run_simulation(pattern="giraffe", method="implicit")
# show results
SolverVisualizer.show_result(v_field=v1, gen_time=gen_time1,scheme_name="Leopard Pattern - Explicit")
SolverVisualizer.show_result(v_field=v2, gen_time=gen_time2,scheme_name="Leopard Pattern - Implicit")
SolverVisualizer.show_result(v_field=v3, gen_time=gen_time3,scheme_name="Giraffe Pattern - Explicit")
SolverVisualizer.show_result(v_field=v4, gen_time=gen_time4,scheme_name="Giraffe Pattern - Implicit")  
