from hdcpy import *
import numpy as np

def flip_random_positions(hypervector, dimensionality, number_positions):
    flip_positions      = []
    flip_hypervector    = hypervector.copy()

    #for _ in range(number_positions):
    rng             = np.random.default_rng()
    flip_positions  = rng.choice(dimensionality, size = number_positions, replace = False)

        #flip_positions.append(np.random.randint(0, dimensionality))

    for position in flip_positions:
        flip_hypervector[position] = np.logical_not(hypervector[position])

    return flip_hypervector

def generate_level_hypervectors(seed_hypervector, dimensionality, levels):
    level_hypervectors  = []
    number_positions    = int(dimensionality / levels)
    dummy_hypervector   = seed_hypervector

    level_hypervectors.append(seed_hypervector)

    for _ in range(1, levels):
        dummy_hypervector = flip_random_positions(dummy_hypervector, dimensionality, number_positions)
        level_hypervectors.append(dummy_hypervector)

    return level_hypervectors

dimensionality      = 10
levels              = 20
seed_hypervector    = generate_hypervector(dimensionality)
number_positions    = 5

#level_array = generate_level_hypervectors(seed_hypervector, dimensionality, levels)

#print(str(seed_hypervector))
#print(str(flip_random_positions(seed_hypervector, dimensionality, 5)))

for _ in range(levels):
    u = generate_hypervector(dimensionality)
    v = flip_random_positions(u, dimensionality, number_positions)

    print('%0.5f' % hamming_distance(u, v, dimensionality))