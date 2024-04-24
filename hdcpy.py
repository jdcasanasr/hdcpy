import numpy as np
import os

from sklearn.datasets           import fetch_openml
from sklearn.model_selection    import train_test_split

def fetch_dataset(dataset_name: str, save_directory:str, test_proportion:float):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_path = os.path.join(save_directory, f'{dataset_name}.csv')

    # Check if dataset already exists, and if not,
    # fetch it from the internet.
    if not os.path.exists(file_path):
        dataset = fetch_openml(name=dataset_name, version=1, parser='auto')
        data, target = np.array(dataset.data).astype(np.float_), np.array(dataset.target).astype(np.int_)

        # Normalize feature values between 0 and 1, if needed.
        if not np.all((data >= 0) & (data <= 1)):
            data = data / np.max(data)

        training_features, testing_features, training_labels, testing_labels = train_test_split(data, target, test_size=test_proportion)

        # Combine data and target into a single array, and
        # store it in .csv format.
        dataset_array = np.column_stack((data, target))

        np.savetxt(file_path, dataset_array, delimiter=',', fmt='%s')

        return training_features, testing_features, training_labels - 1, testing_labels - 1

    else:
        # Read pre-existing ".csv" file, split into features and labels
        # and return both as numpy arrays.
        dataset_array = np.genfromtxt(file_path, delimiter=',', dtype=np.float_)
        data = dataset_array[:, :-1]
        target = dataset_array[:, -1].astype(np.int_)

        training_features, testing_features, training_labels, testing_labels = train_test_split(data, target, test_size=test_proportion)

        return training_features, testing_features, training_labels - 1, testing_labels - 1


# Returns a random hypervector with a BSC or MAP Vector-Symbolic Architecture scheme.
def random_hypervector(number_of_dimensions:np.uint, vsa:np.str_) -> np.array:
    supported_vsas = ['BSC', 'MAP']

    if vsa not in supported_vsas:
        raise ValueError(f'Invalid VSA: Expected one of the following: {supported_vsas}')

    else:
        match vsa:
            case 'BSC':
                return np.random.choice([True, False], size = number_of_dimensions, p = [0.5, 0.5])
            
            case 'MAP':
                return np.random.choice([-1, 1], size = number_of_dimensions, p = [0.5, 0.5])
            
            case _:
                return np.random.choice([True, False], size = number_of_dimensions, p = [0.5, 0.5])

def hamming_distance(hypervector_u:np.array, hypervector_v:np.array) -> np.double:
    if np.shape(hypervector_u) != np.shape(hypervector_v):
        raise ValueError(f'Shapes do not match: {np.shape(hypervector_u)} != {np.shape(hypervector_v)}')

    else:
        number_of_dimensions = hypervector_u.size

        return np.count_nonzero(np.logical_xor(hypervector_u, hypervector_v, dtype = np.bool_)) / number_of_dimensions

def cosine_similarity(hypervector_u:np.array, hypervector_v:np.array) -> np.double:
    if np.shape(hypervector_u) != np.shape(hypervector_v):
        raise ValueError(f'Shapes do not match: {np.shape(hypervector_u)} != {np.shape(hypervector_v)}')

    else:
        dot_product = np.dot(hypervector_u, hypervector_v)
        magnitude_u = np.linalg.norm(hypervector_u)
        magnitude_v = np.linalg.norm(hypervector_v)
    
        return dot_product / (magnitude_u * magnitude_v)

def bind(hypervector_u:np.array, hypervector_v:np.array, vsa:np.str_) -> np.array:
    if np.shape(hypervector_u) != np.shape(hypervector_v):
        raise ValueError(f'Shapes do not match: {np.shape(hypervector_u)} != {np.shape(hypervector_v)}')

    else: 
        supported_vsas = ['BSC', 'MAP']

        if vsa not in supported_vsas:
            raise ValueError(f'Invalid VSA: Expected one of the following: {supported_vsas}')

        else:
            match vsa:
                case 'BSC':
                    return np.logical_xor(hypervector_u, hypervector_v, dtype = np.bool_)
                
                case 'MAP':
                    return np.multiply(hypervector_u, hypervector_v, dtype = np.int_)
                
                case _:
                    return np.multiply(hypervector_u, hypervector_v, dtype = np.int_)

def bundle(hypervector_u:np.array, hypervector_v:np.array, vsa:np.str_) -> np.array:
    if np.shape(hypervector_u) != np.shape(hypervector_v):
        raise ValueError(f'Shapes do not match: {np.shape(hypervector_u)} != {np.shape(hypervector_v)}')

    else:
        supported_vsas = ['BSC', 'MAP']

        if vsa not in supported_vsas:
            raise ValueError(f'Invalid VSA: Expected one of the following: {supported_vsas}')

        else:
            match vsa:
                case 'BSC', _:
                    number_of_dimensions    = hypervector_u.size
                    hypervector_w           = random_hypervector(number_of_dimensions, vsa)

                    return np.logical_or(np.logical_and(hypervector_w, np.logical_xor(hypervector_u, hypervector_v, dtype = np.bool_), dtype = np.bool_), np.logical_and(hypervector_u, hypervector_v, dtype = np.bool_), dtype = np.bool_)
                
                case 'MAP':
                    number_of_dimensions    = hypervector_u.size
                    hypervector_w           = random_hypervector(number_of_dimensions, vsa)

                    return np.sign(np.add(hypervector_u, hypervector_v, hypervector_w))

def multibundle(hypermatrix: np.array, vsa:np.str_) -> np.array:
    if np.shape(hypermatrix)[0] < 2:
        raise ValueError(f'Expected at least two hypervectors, found {np.shape(hypermatrix)[0]}')
    
    else:
        supported_vsas = ['BSC', 'MAP']

        if vsa not in supported_vsas:
            raise ValueError(f'Invalid VSA: Expected one of the following: {supported_vsas}')

        else:
            match vsa:
                case 'BSC', _:
                    number_of_rows, number_of_columns   = np.shape(hypermatrix)
                    number_of_dimensions                = number_of_columns
                    tie_breaking_hypervector            = random_hypervector(number_of_dimensions, vsa)

                    number_of_true  = np.sum(hypermatrix, axis = 0)
                    number_of_false = np.subtract(number_of_rows, number_of_true)

                    bundle_hypervector = bundle_hypervector = np.where(number_of_true > number_of_false, True,
                                                np.where(number_of_true < number_of_false, False, tie_breaking_hypervector))

                    return bundle_hypervector
                
                case 'MAP':
                    number_of_rows, number_of_columns   = np.shape(hypermatrix)
                    number_of_dimensions                = number_of_columns
                    
                    # If dealing with an even number of hypervectors,
                    # yield a tie-breaking hypervector.
                    if number_of_rows % 2 == 0:
                        tie_breaking_hypervector = random_hypervector(number_of_dimensions, vsa)

                        return np.sign(np.sum(np.vstack((hypermatrix, tie_breaking_hypervector)), axis = 0))

                    else:
                        return np.sign(np.sum(hypermatrix, axis = 0))

def permute(hypervector:np.array, shift_amount:np.int_) -> np.array:
    return np.roll(hypervector, shift_amount)

def flip(hypervector_u:np.array, number_of_positions:np.uint) -> np.array:
    #number_of_dimensions = hypervector_u.size
    number_of_dimensions = np.shape(hypervector_u)

    flip_hypervector    = np.empty(number_of_dimensions, np.bool_)
    flip_positions      = np.random.choice(number_of_dimensions, size = number_of_positions, replace = False)

    for element_index in range(number_of_dimensions):
        if element_index in flip_positions:
            flip_hypervector[element_index] = np.logical_not(hypervector_u[element_index])
        
        else:
            flip_hypervector[element_index] = hypervector_u[element_index]

    return flip_hypervector

def get_level_hypermatrix(number_of_levels:np.uint, number_of_dimensions:np.uint) -> np.array:
    number_of_rows              = number_of_levels
    number_of_columns           = number_of_dimensions
    number_of_flip_positions    = int(np.ceil((number_of_dimensions / (number_of_levels - 1))))
    level_hypermatrix           = np.empty((number_of_rows, number_of_columns), np.bool_)
    
    level_hypermatrix[0]        = random_hypervector(number_of_dimensions)

    for level_index in range(1, number_of_levels):
        level_hypermatrix[level_index] = flip(level_hypermatrix[level_index - 1], number_of_flip_positions)

    return level_hypermatrix

def get_position_hypermatrix(number_of_positions:np.uint, number_of_dimensions:np.uint) -> np.array:
    number_of_rows          = number_of_positions
    number_of_columns       = number_of_dimensions
    
    position_hypermatrix    = np.empty((number_of_rows, number_of_columns), np.bool_)

    for position_index in range(number_of_positions):
        position_hypermatrix[position_index] = random_hypervector(number_of_dimensions)

    return position_hypermatrix

def get_quantization_range(lower_limit:np.double, upper_limit:np.double, number_of_levels:np.uint) -> np.array:
    return np.linspace(lower_limit, upper_limit, number_of_levels)

# Return the level hypervector matching a given sample.
def get_level_hypervector(feature:np.double, quantization_range:np.array, level_hypermatrix:np.array) -> np.array:
    index = np.digitize(feature, quantization_range, True)
    return level_hypermatrix[index]

# Renamed 'encode' to 'encode_analog'
def encode_analog(feature_vector:np.array, quantization_range:np.array, level_hypermatrix:np.array, position_hypermatrix:np.array) -> np.array:
    number_of_rows, number_of_columns   = np.shape(position_hypermatrix)
    # ToDo: Add a check for number_of_features != number_of_rows
    number_of_features                  = number_of_rows
    # Hypermatrix for 'level_hypervector BIND position_hypervector'
    bind_hypermatrix                    = np.empty((number_of_rows, number_of_columns), np.bool_)

    for feature_index in range(number_of_features):
        feature                             = feature_vector[feature_index]
        position_hypervector                = position_hypermatrix[feature_index]
        level_hypervector                   = get_level_hypervector(feature, quantization_range, level_hypermatrix)
        bind_hypermatrix[feature_index]     = bind(position_hypervector, level_hypervector)

    return multibundle(bind_hypermatrix)

# Find the class hypervector matching the
# minimum distance with a query vector.
def classify(associative_memory:np.array, query_hypervector:np.array):
    number_of_classes = np.shape(associative_memory)[0]
    similarity_vector = np.empty(number_of_classes, np.double)

    for class_index in range(number_of_classes):
        similarity_vector[class_index] = hamming_distance(associative_memory[class_index], query_hypervector)

    # (class, similarity)
    return np.argmin(similarity_vector) # + 1

# Retrains the model by adding and subtracting hypervectors.
def retrain_analog(
    associative_memory:np.array,
    training_data:np.array,
    training_labels:np.array,
    quantization_range:np.array,
    level_hypermatrix:np.array,
    position_hypermatrix:np.array
):
    number_of_queries = np.shape(training_data)[0]

    for index in range(number_of_queries):
        query_hypervector   = encode_analog(training_data[index][:], quantization_range, level_hypermatrix, position_hypermatrix)
        predicted_class     = classify(associative_memory, query_hypervector)
        actual_class        = training_labels[index]

        # Caution: We assume training labels start from zero!
        if predicted_class != actual_class:
            # We binarize via a unit-step function.
            associative_memory[predicted_class] = np.heaviside(np.subtract(associative_memory[predicted_class], query_hypervector), 0)
            associative_memory[actual_class]    = np.heaviside(np.add(associative_memory[actual_class], query_hypervector), 0)