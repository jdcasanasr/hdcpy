import numpy    as np
import os

from hdcpy                      import *
from sklearn.datasets           import fetch_openml
from sklearn.model_selection    import train_test_split

def substitute_characters(arr):
    # Define the mapping from characters to integers
    char_to_int = {'A': 0, 'G': 1, 'T': 2, 'C': 3, 'D': 4, 'N': 5, 'S': 6, 'R': 7}
    
    # Vectorized substitution using numpy
    vectorized_substitution = np.vectorize(lambda x: char_to_int[x])
    
    # Apply the substitution
    int_array = vectorized_substitution(arr)
    
    return int_array

def sub_2(arr):
    # Define the mapping from characters to integers
    char_to_int = {'N': 0, 'EI': 1, 'IE': 2}
    
    # Vectorized substitution using numpy
    vectorized_substitution = np.vectorize(lambda x: char_to_int[x])
    
    # Apply the substitution
    int_array = vectorized_substitution(arr)
    
    return int_array

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

def binarize(u:np.array, vsa:np.str_) -> np.array:
    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')
    
    match vsa:
        case 'BSC':
            return np.where(u >= 1, 1, 0)

        case 'MAP':
            return np.where(u > 0, 1, -1)
        
        case _:
            print('Warning: hypervector not binarized')

            return u

def get_dataset(dataset_name: str, save_directory:str, test_proportion:float):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_path = os.path.join(save_directory, f'{dataset_name}.csv')

    if not os.path.exists(file_path):
        dataset         = fetch_openml(name = dataset_name, version = 1, parser = 'auto')
        data            = np.array(dataset.data)
        target          = np.array(dataset.target)
        dataset_array   = np.column_stack((data, target))

        np.savetxt(file_path, dataset_array, delimiter = ',', fmt = '%s')

        return train_test_split(data, target, test_size = test_proportion)

    else:
        dataset_array   = np.genfromtxt(file_path, delimiter = ',', dtype = np.str_)
        data            = np.array(dataset_array[:, :-1])
        target          = np.array(dataset_array[:, -1])

        return train_test_split(data, target, test_size = test_proportion)