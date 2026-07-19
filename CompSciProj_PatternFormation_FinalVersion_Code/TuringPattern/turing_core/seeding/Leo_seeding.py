import numpy as np

def leo_seeding( N):
    #global uniform initialization strategy:
    # We start with a uniform state where U fills the space and V is zero, representing the unreacted state of the system
    u = np.ones((N, N))
    v = np.zeros((N, N))

    #global seeding strategy: we randomly inject a small amount of V at random locations across the entire grid to serve as seeds that trigger the reaction, creating initial perturbations in the system
    # We randomly select about 1% of the pixels as "reaction centers" where we inject a small amount of V to trigger the reaction, creating initial perturbations in the system
    seed_mask = np.random.rand(N, N) > 0.98
    v[seed_mask] = 0.5
    u[seed_mask] = 0.5

    # 3. Add small Gaussian noise to break perfect symmetry and promote pattern formation
    u += np.random.normal(scale=0.02, size=(N, N))
    v += np.random.normal(scale=0.02, size=(N, N))

    return u,v
  
   