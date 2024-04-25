import numpy as np

def random_hypervector(dimensionality:np.uint, vsa:np.str_) -> np.array:
    supported_vsas = ['BSC', 'MAP']

    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    match vsa:
        case 'BSC':
            return np.random.choice([0, 1], size = dimensionality, p = [0.5, 0.5])
        
        case 'MAP':
            return np.random.choice([-1, 1], size = dimensionality, p = [0.5, 0.5])
        
        case _:
            print('Warning: Returning a non-VSA hypervector.')

            return np.empty(dimensionality)

def hamming_distance(u:np.array, v:np.array) -> np.double:
    u_dimensionality = np.shape(u)[0]
    v_dimensionality = np.shape(v)[0]

    if u_dimensionality != v_dimensionality:
        raise ValueError(f'Hypervectors must have the same dimensions.')

    return  np.sum(u != v) / u_dimensionality

def cosine_similarity(u:np.array, v:np.array) -> np.double:
    if np.shape(u)[0] != np.shape(v)[0]:
        raise ValueError(f'Hypervectors must have the same dimensions.')

    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def bind(u:np.array, v:np.array, vsa:np.str_) -> np.array:
    if np.shape(u)[0] != np.shape(v)[0]:
        raise ValueError(f'Hypervectors must have the same dimensions.')
    
    supported_vsas = ['BSC', 'MAP']

    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    match vsa:
        case 'BSC':
            return np.bitwise_xor(u, v)
        
        case 'MAP':
            return np.multiply(u, v, dtype = np.int_)
        
        case _:
            print('Warning: Returning a non-VSA hypervector.')

            return np.empty(np.shape(u)[0])

# HERE!        
def bundle(hypervector_u:np.array, hypervector_v:np.array, vsa:np.str_) -> np.array:
    if np.shape(hypervector_u) != np.shape(hypervector_v):
        raise ValueError(f'Shapes do not match: {np.shape(hypervector_u)} != {np.shape(hypervector_v)}')

    else:
        supported_vsas = ['BSC', 'MAP']

        if vsa not in supported_vsas:
            raise ValueError(f'Invalid VSA: Expected one of the following: {supported_vsas}')

        else:
            match vsa:
                case 'BSC':
                    number_of_dimensions    = np.shape(hypervector_u)[0]
                    hypervector_w           = random_hypervector(number_of_dimensions, vsa)

                    return np.logical_or(np.logical_and(hypervector_w, np.logical_xor(hypervector_u, hypervector_v, dtype = np.uint), dtype = np.uint), np.logical_and(hypervector_u, hypervector_v, dtype = np.uint), dtype = np.uint)
                
                case 'MAP':
                    number_of_dimensions    = hypervector_u.size
                    hypervector_w           = random_hypervector(number_of_dimensions, vsa)

                    return np.sign(np.add(hypervector_u, hypervector_v, hypervector_w))
                
                case _:
                    number_of_dimensions    = np.shape(hypervector_u)[0]
                    hypervector_w           = random_hypervector(number_of_dimensions, vsa)

                    return np.logical_or(np.logical_and(hypervector_w, np.logical_xor(hypervector_u, hypervector_v, dtype = np.uint), dtype = np.uint), np.logical_and(hypervector_u, hypervector_v, dtype = np.uint), dtype = np.uint)