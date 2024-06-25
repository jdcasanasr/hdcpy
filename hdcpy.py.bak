import numpy as np

supported_vsas = ['BSC', 'MAP']

def random_hypervector(dimensionality:np.uint, vsa:np.str_) -> np.array:
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
     
def bundle(u:np.array, v:np.array, vsa:np.str_) -> np.array:
    dimensionality_u = np.shape(u)[0]
    dimensionality_v = np.shape(v)[0]

    if dimensionality_u != dimensionality_v:
        raise ValueError(f'Hypervectors must have the same dimensions.')

    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    match vsa:
        case 'BSC':
            w = random_hypervector(dimensionality_u, vsa)

            return np.where(np.add(u, v, w) >= 2, 1, 0)
        
        case 'MAP':
            w = random_hypervector(dimensionality_u, vsa)

            return np.sign(np.add(u, v, w))
        
        case _:
            print('Warning: Returning a non-VSA hypervector.')

            return np.empty(dimensionality_u)
        
def permute(u:np.array, shift_amount:np.int_) -> np.array:
    return np.roll(u, shift_amount)

def multibundle(hypermatrix:np.array, vsa:np.str_) -> np.array:
    number_of_hypervectors  = np.shape(hypermatrix)[0]
    dimensionality          = np.shape(hypermatrix)[1]

    if number_of_hypervectors < 2:
        raise ValueError(f'Expected at least two hypervectors.')

    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    match vsa:
        case 'BSC':
            if number_of_hypervectors % 2 == 0:
                tie_break = random_hypervector(dimensionality, vsa)

                return np.where(np.sum(np.vstack((hypermatrix, tie_break)), axis = 0) >= (number_of_hypervectors / 2), 1, 0)

            else:
                return np.where(np.sum(hypermatrix, axis = 0) >= np.uint(np.ceil((number_of_hypervectors / 2))), 1, 0)
                
        
        case 'MAP':
            if number_of_hypervectors % 2 == 0:
                tie_break = random_hypervector(dimensionality, vsa)

                return np.sign(np.sum(np.vstack((hypermatrix, tie_break)), axis = 0))

            else:
                return np.sign(np.sum(hypermatrix, axis = 0))
            
        case _:
            print('Warning: Returning a non-VSA hypervector.')

            return np.empty(dimensionality)