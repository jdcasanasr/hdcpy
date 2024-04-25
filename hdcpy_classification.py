from hdcpy_v2           import *
from hdcpy_auxiliary    import *

def get_level_hypermatrix(number_of_levels:np.uint, dimensionality:np.uint, vsa:np.str_) -> np.array:
    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    number_of_flip_positions    = np.uint(np.ceil((dimensionality / (number_of_levels - 1))))
    level_hypermatrix           = np.empty((number_of_levels, dimensionality), np.int_)
    
    level_hypermatrix[0][:]     = random_hypervector(dimensionality, vsa)

    for index in range(1, number_of_levels):
        level_hypermatrix[index] = flip(level_hypermatrix[index - 1], number_of_flip_positions, vsa)

    return level_hypermatrix

def get_id_hypermatrix(number_of_ids:np.uint, dimensionality:np.uint, vsa:np.str_) -> np.array:
    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    id_hypermatrix    = np.empty((number_of_ids, dimensionality), np.int_)

    for index in range(number_of_ids):
        id_hypermatrix[index] = random_hypervector(dimensionality, vsa)

    return id_hypermatrix