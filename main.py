from hdcpy import *
import matplotlib.pyplot as plt

dimensionality  = 10000
successes       = 0
experiments     = 3000

for _ in range(experiments):
    u = generate_hypervector(dimensionality)
    v = generate_hypervector(dimensionality)

    if (hamming_distance(u, v, dimensionality) >= 0.5):
        successes += 1

probability = successes / experiments

print(f'Probability over {experiments} trials is: {probability}.')
    
