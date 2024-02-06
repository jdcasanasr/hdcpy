import numpy as np

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA's.
def generate_hypervector(dimensionality):
    return np.random.choice([True, False], size = dimensionality, p = [0.5, 0.5])

# Compute Hamming distance between two hypervectors,
# generated via the 'generate_hypervector' function.
# ToDo: Eliminate 'dimensionality' parameter.
def hamming_distance(u, v):
    return float(np.count_nonzero(np.logical_xor(u, v))) / u.size

dimensionality = 10000

u = generate_hypervector(dimensionality)
v = generate_hypervector(dimensionality)

delta = hamming_distance(u, v)

#print(u)
#print(v)
print("Hamming Distance: %0.4f" % delta)