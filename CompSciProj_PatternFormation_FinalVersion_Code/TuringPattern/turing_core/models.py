
def Grey_Scott( F, K, u, v):
    uvv = u * v**2
    f = -uvv + F * (1.0 - u)
    g = uvv - (F + K) * v
    return f,g

def leopard_model():
    return {
        'N': 200, 'dx': 1.0, 'dt': 1.0, 'steps': 10000,
        'Du': 0.16, 'Dv': 0.08, 'F': 0.0367, 'K': 0.0649,
    }

def giraffe_model():
    return {
        'N': 200, 'dx': 1.0, 'dt': 0.02, 'steps': 50000,
        'Du': 0.8, 'Dv': 0.4, 'F': 0.089, 'K': 0.060,
    }