from .models import leopard_model, giraffe_model
from .solvers.explicit import ExplicitScheme
from .solvers.Implicit import CrankNicolsonScheme
from .seeding.Giraffe_seeding import giraffe_seeding
from .seeding.Leo_seeding import leo_seeding
import time

def run_simulation(pattern="leopard", method="explicit"):
    # start the timer to measure the total generation time for the simulation, which includes both the setup and the execution of the solver
    start = time.time()
     
    if pattern == "leopard" :
        params = leopard_model()
        u,v = leo_seeding( N=params['N'] )

    else:
        params = giraffe_model()
        u,v = giraffe_seeding( N=params['N']) 
 
    # set up the solver based on the chosen method (explicit or implicit) and the parameters defined for the selected pattern (leopard or giraffe). The solver will be responsible for running the reaction-diffusion simulation and generating the resulting pattern.
    if method == "explicit":
        solver = ExplicitScheme( dx=params['dx'], dt=params['dt'], steps=params['steps'],
         F=params['F'], K=params['K'], Du=params['Du'], Dv=params['Dv'],u=u,v=v)
        
    else:  
        solver = CrankNicolsonScheme(N=params['N'],dx=params['dx'], dt=params['dt'], steps=params['steps'],
         F=params['F'], K=params['K'], Du=params['Du'], Dv=params['Dv'],u=u,v=v)
        
    v, gen_time = solver.run()

    return v, gen_time
