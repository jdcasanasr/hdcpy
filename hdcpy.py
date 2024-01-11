import numpy as np

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA's.
def generate_hypervector(dimensionality):
    return np.random.randint(0, 2, dimensionality, np.byte)

# Copy the contents of one hypervector instance into another.
def copy_hypervector(source_hypervector):
    destination_hypervector = []

    for element in source_hypervector:
        destination_hypervector.append(element)

    return destination_hypervector

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

# Negate random positions in a source hypervector. Return a different hypervector.
def flip_random_positions(source_hypervector, dimensionality, number_positions):
    flip_positions      = []
    flip_hypervector    = copy_hypervector(source_hypervector)

    # Generate random positions without substitution.
    rng             = np.random.default_rng()
    flip_positions  = rng.choice(dimensionality, size = number_positions, replace = False)

    for position in flip_positions:
        flip_hypervector[position] = np.logical_not(source_hypervector[position])

    return flip_hypervector

# Generate a set of level hypervectors from a seed hypervector (L1).
def generate_level_hypervectors(seed_hypervector, dimensionality, levels):
    level_hypervectors  = []
    number_positions    = int(dimensionality / levels)
    dummy_hypervector   = copy_hypervector(seed_hypervector)

    level_hypervectors.append(seed_hypervector)

    for _ in range(1, levels):
        dummy_hypervector = flip_random_positions(dummy_hypervector, dimensionality, number_positions)
        level_hypervectors.append(dummy_hypervector)

    return level_hypervectors

# Divide a given [lower_limit, upper_limit] range into even chunks.
def quantize_range(lower_limit, upper_limit, number_levels):
    return np.linspace(lower_limit, upper_limit, number_levels)

# Return the level hypervector matching a given sample.
def quantize_sample(sample, quantized_range, level_hypervector_array):
    return level_hypervector_array[np.digitize(sample, quantized_range, True)]

# Find the class hypervector matching the
# minimum distance with a query vector.
def find_class(associative_memory, query_hypervector, dimensionality):
    minimum_distance = 1.0
    
    for class_hypervector in associative_memory:
        delta = hamming_distance(query_hypervector, class_hypervector, dimensionality)

        if (delta < minimum_distance):
            minimum_distance = delta

    return minimum_distance