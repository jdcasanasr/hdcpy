import numpy as np

# ToDo: Generate 'level' and 'ID (position)' hypervectors.

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA's.
def generate_hypervector(dimensionality):
    return np.random.randint(0, 2, dimensionality, np.byte)

# Compute Hamming distance between two hypervectors,
# generated via the 'generate_hypervector' function.
# ToDo: Eliminate 'dimensionality' parameter.
def hamming_distance(u, v, dimensionality):
    return np.count_nonzero(np.logical_xor(u, v)) / dimensionality

# Basic HDC operations.
def bind(u, v):
    return np.logical_xor(u, v)

def bundle(u, v):
    return np.where((u + v) >= 2, 1, 0)

def permute(u, amount):
    return np.roll(u, amount)

# Find the class hypervector matching the
# minimum distance with a query vector.
def find_class(associative_memory, query_hypervector, dimensionality):
    minimum_distance = 1.0
    
    for class_hypervector in associative_memory:
        delta = hamming_distance(query_hypervector, class_hypervector, dimensionality)

        if (delta < minimum_distance):
            minimum_distance = delta

    return minimum_distance