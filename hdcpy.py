import numpy as np

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA's.
def generate_hypervector(dimensionality) -> np.array:
    return np.random.choice([True, False], size = dimensionality, p = [0.5, 0.5])

# Compute Hamming distance between two hypervectors,
# generated via the 'generate_hypervector' function.
def hamming_distance(u, v) -> np.single:
    return np.count_nonzero(np.logical_xor(u, v, dtype =np.bool_)) / u.size

# Basic HDC operations.
def bind(u, v) -> np.array:
    return np.logical_xor(u, v, dtype = np.bool_)

def bundle(u, v) -> np.array:
    w = generate_hypervector(u.size)

    return np.logical_or(np.logical_and(w, np.logical_xor(u, v, dtype = np.bool_), dtype = np.bool_), np.logical_and(u, v, dtype = np.bool_), dtype = np.bool_)

def permute(u, amount) -> np.array:
    return np.roll(u, amount, dtype = np.bool_)

# Negate random positions in a source hypervector. Return a different hypervector.
def flip(u, number_positions) -> np.array:
    flip_u          = np.array([None] * u.size)
    flip_positions  = np.random.choice(u.size, size = number_positions, replace = False)

    for index in range(u.size):
        if (index in flip_positions):
            flip_u[index] = np.logical_not(u[index])
        
        else:
            flip_u[index] = u[index]

    return flip_u

# Generate a set of level hypervectors from a seed hypervector (L1).
def generate_level_hypervectors(u_seed, number_levels) -> np.array:
    level_hypervectors  = np.array([None] * number_levels)
    number_positions    = int(u_seed.size / number_levels)

    level_hypervectors[0] = u_seed

    for index in range(1, number_levels):
        level_hypervectors[index] = flip(level_hypervectors[index - 1], number_positions)

    return level_hypervectors

# Generate a set of ID (position) hypervectors.
def generate_id_hypervectors (dimensionality, number_id_hypervectors) -> np.array:
    id_hypervectors_array = np.array([None] * number_id_hypervectors)

    for index in range(number_id_hypervectors):
        id_hypervectors_array[index] = generate_hypervector(dimensionality)

    return id_hypervectors_array

# Divide a given [lower_limit, upper_limit] range into even chunks.
def quantize_range(lower_limit, upper_limit, number_levels):
    return np.linspace(lower_limit, upper_limit, number_levels)

# Return the level hypervector matching a given sample.
def quantize_sample(sample, quantized_range, level_hypervector_array) -> np.array:
    return level_hypervector_array[np.digitize(sample, quantized_range, True)]

# Cram the whole dataset into a single array, stripping blanks and
# commas, and convertig values into floating-point numbers.
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = [float(val) for val in line.strip().split(',')]
            data.append(values)
    return data

# Load data from a dataset into two separate 'feature' and 'class'
# vectors (ISOLET only so far).
# ToDo: Generalize for any dataset in .csv format.
def load_dataset(dataset_path):
    data_matrix     = load_data(dataset_path)
    feature_matrix  = []
    class_vector    = []

    for data_vector in data_matrix:
        feature_matrix.append(data_vector[0:616])
        class_vector.append(data_vector[617])

    return feature_matrix, class_vector

# Embed a feature vector into binary hyperspace.
# ToDo: Generalize for other VSA's.
def transform (feature_vector, dimensionality, quantized_range, level_hypervector_array, id_hypervector_array) -> np.array:
    transformed_hypervector = np.array([None] * dimensionality)

    for feature_index in range(len(feature_vector)):

                feature                 = feature_vector[feature_index]
                position_hypervector    = id_hypervector_array[feature_index]
                level_hypervector       = quantize_sample(feature, quantized_range, level_hypervector_array)

                bind_hypervector        = bind(position_hypervector, level_hypervector)

                if (None in transformed_hypervector):
                    transformed_hypervector = bind_hypervector

                else:
                    transformed_hypervector = bundle(transformed_hypervector, bind_hypervector)

    return transformed_hypervector

# Find the class hypervector matching the
# minimum distance with a query vector.
def classify(associative_memory, query_hypervector):
    distance_array = np.array([None] * associative_memory.size)

    for index in range(associative_memory.size):
        distance_array[index] = hamming_distance(associative_memory[index], query_hypervector)

    return (np.argmin(distance_array) + 1, np.min(distance_array))
