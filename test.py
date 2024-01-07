import numpy as np
import matplotlib.pyplot as plt

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA
#def generate_hypervector(dimensionality):
#    unpacked_hypervector    = np.random.randint(low = 0, high = 2, size = dimensionality, dtype = np.bool_)
#    packed_hypervector      = np.packbits(unpacked_hypervector, axis = None, bitorder = 'big')

#    return packed_hypervector

def generate_hypervector(dimensionality):
    rng = np.random.default_rng()
    unpacked_hypervector    = rng.integers(2, size = dimensionality)
    #unpacked_hypervector    = np.random.Generator.integers(low = 0, high = 2, size = dimensionality, dtype = np.bool_)
    packed_hypervector      = np.packbits(unpacked_hypervector, axis = None, bitorder = 'big')

    return packed_hypervector

# Compute Hamming distance between two packed hypervectors,
# generated via the 'generate_hypervector' function.
def hamming_distance(u, v):
    return np.count_nonzero(np.bitwise_xor(u, v))

start   = 100
stop    = 10000
step    = 100

experiments = 100

delta_array             = []
dimensionality_array    = []

for dimensionality in range(start, stop, step):
    for experiment in range(experiments):
        u = generate_hypervector(dimensionality)
        v = generate_hypervector(dimensionality)

        delta_array.append(hamming_distance(u, v))
        #dimensionality_array.append(dimensionality)

plt.plot(delta_array)
plt.xlabel('Dimensionality')
plt.ylabel('Hamming Distance')
plt.show()