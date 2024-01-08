import numpy as np
import matplotlib.pyplot as plt

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA
def generate_hypervector(dimensionality):
    return np.random.randint(0, 2, dimensionality, np.byte)

# Compute Hamming distance between hypervectors,
# generated via the 'generate_hypervector' function.
def hamming_distance(u, v):
    return np.count_nonzero(np.logical_xor(u, v))

# Basic HDC Operations
def bind(u, v):
    return np.logical_xor(u, v)

def bundle(u, v):
    return np.where((u + v) >= 2, 1, 0)

def permute(u, amount):
    return np.roll(u, amount)

experiments = 100

delta_array             = []
dimensionality_array    = []

for dimensionality in range(start, stop, step):
    for experiment in range(experiments):
        u = generate_hypervector(dimensionality)
        v = generate_hypervector(dimensionality)

        delta_array.append(hamming_distance(u, v))

plt.plot(delta_array)
plt.xlabel('Dimensionality')
plt.ylabel('Hamming Distance')
plt.show()