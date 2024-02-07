import numpy as np

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA's.
def generate_hypervector(dimensionality):
    return np.random.choice([True, False], size = dimensionality, p = [0.5, 0.5])

# Compute Hamming distance between two hypervectors,
# generated via the 'generate_hypervector' function.
def hamming_distance(u, v):
    return np.count_nonzero(np.logical_xor(u, v)) / u.size

# Basic HDC operations.
def bind(u, v):
    return np.logical_xor(u, v)

def bundle(u, v):
    w = generate_hypervector(u.size)

    return np.logical_or(np.logical_and(w, np.logical_xor(u, v)), np.logical_and(u, v))

def permute(u, amount):
    return np.roll(u, amount)

# Negate random positions in a source hypervector. Return a different hypervector.
def flip(u, number_positions):
    flip_u = np.array([None] * u.size)
    flip_positions = np.random.choice(u.size, size = number_positions, replace = False)

    for index in range(u.size):
        if (index in flip_positions):
            flip_u[index] = np.logical_not(u[index])
        
        else:
            flip_u[index] = u[index]

    return flip_u

# Generate a set of level hypervectors from a seed hypervector (L1).
def generate_level_hypervectors(u_seed, number_levels):
    level_hypervectors  = np.array([None] * number_levels)
    number_positions    = int(u_seed.size / number_levels)

    level_hypervectors[0] = u_seed

    for index in range(1, number_levels):
        level_hypervectors[index] = flip(level_hypervectors[index - 1], number_positions)

    return level_hypervectors

dimensionality = 10
number_levels = 5

u_level_array = generate_level_hypervectors(generate_hypervector(dimensionality), number_levels)

print(u_level_array)