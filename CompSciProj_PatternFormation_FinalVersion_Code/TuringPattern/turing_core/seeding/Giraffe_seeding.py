import numpy as np

def giraffe_seeding( N):
    
        np.random.seed(42)
        
        u = 0.5+ np.random.uniform(-0.1, 0.1, (N, N))
        v = 0.3+ np.random.uniform(-0.1, 0.1, (N, N))
            
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        return u,v
    