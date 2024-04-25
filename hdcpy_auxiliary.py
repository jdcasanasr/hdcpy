import numpy as np
from hdcpy_v2 import *

def flip(u:np.array, number_of_positions:np.uint, vsa:np.str_) -> np.array:
    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    dimensionality      = np.shape(u)[0]
    flip_hypervector    = np.empty(dimensionality, dtype = np.int_)
    flip_positions      = np.random.choice(dimensionality, size = number_of_positions, replace = False)

    match vsa:
        case 'BSC':
            for index in range(dimensionality):
                if index in flip_positions:
                    flip_hypervector[index] = 1 if u[index] == 0 else 0

                else:
                    flip_hypervector[index] = u[index]

            return flip_hypervector

        case 'MAP':
            for index in range(dimensionality):
                if index in flip_positions:
                    flip_hypervector[index] = 1 if u[index] == -1 else -1

                else:
                    flip_hypervector[index] = u[index]

            return flip_hypervector

        case _:
            print('Warning: Returning a non-VSA hypervector.')

            return np.empty(dimensionality)