import numpy as np
import matplotlib.pyplot as plt

class SolverVisualizer:
    @staticmethod
    def show_result(v_field, gen_time, scheme_name):
        plt.figure(figsize=(6, 6))
        plt.imshow(v_field, cmap='Greys_r', vmin=0, vmax=1)
        plt.title(f"{scheme_name}\nTime: {gen_time:.2f}s")
        plt.axis('off')
        plt.show()