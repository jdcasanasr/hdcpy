import numpy as np

# ToDo: Generalize for other VSA's.
def random_hypervector(number_of_dimensions:np.uint) -> np.array:
    return np.random.choice([True, False], size = number_of_dimensions, p = [0.5, 0.5])

# ToDo: Add check for different distances.
def hamming_distance(hypervector_u:np.array, hypervector_v:np.array) -> np.double:
    number_of_dimensions = hypervector_u.size

    return np.count_nonzero(np.logical_xor(hypervector_u, hypervector_v, dtype = np.bool_)) / number_of_dimensions

def bind(hypervector_u:np.array, hypervector_v:np.array) -> np.array:
    return np.logical_xor(hypervector_u, hypervector_v, dtype = np.bool_)

# ToDo: Add check for different dimensionalities.
def bundle(hypervector_u:np.array, hypervector_v:np.array) -> np.array:
    number_of_dimensions    = hypervector_u.size
    hypervector_w           = random_hypervector(number_of_dimensions)

    return np.logical_or(np.logical_and(hypervector_w, np.logical_xor(hypervector_u, hypervector_v, dtype = np.bool_), dtype = np.bool_), np.logical_and(hypervector_u, hypervector_v, dtype = np.bool_), dtype = np.bool_)

# ToDo: Add check for different dimensionalities.
def multibundle(hypermatrix:np.array) -> np.array:
    number_of_rows, number_of_columns   = np.shape(hypermatrix)
    number_of_dimensions                = number_of_columns
    bundle_hypervector                  = np.empty(number_of_dimensions, np.bool_)

    # Even number of hypervectors case.
    if number_of_rows % 2 == 0:
        # Append a random hypervector to break any possible ties.
        new_hypermatrix         = np.empty((number_of_rows + 1, number_of_columns), np.bool_)
        # Check this part!
        new_hypermatrix[:-1]    = hypermatrix
        new_hypermatrix[-1]     = random_hypervector(number_of_dimensions)
        
        for column_index in range(number_of_columns):
            number_of_true      = 0
            number_of_false     = 0

            for row_index in range(number_of_rows + 1):
                if new_hypermatrix[row_index][column_index] == True:
                    number_of_true += 1
                else:
                    number_of_false += 1

            if number_of_true > number_of_false:
                bundle_hypervector[column_index] = True

            else:
                bundle_hypervector[column_index] = False

    # Odd number of hypervectors case.
    else:
        for column_index in range(number_of_columns):
            number_of_true      = 0
            number_of_false     = 0

            for row_index in range(number_of_rows):
                if hypermatrix[row_index][column_index] == True:
                    number_of_true += 1
                else:
                    number_of_false += 1

            if number_of_true > number_of_false:
                bundle_hypervector[column_index] = True

            else:
                bundle_hypervector[column_index] = False

    return bundle_hypervector

def flip(hypervector_u:np.array, number_of_positions:np.uint) -> np.array:
    number_of_dimensions = hypervector_u.size

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

# Cram the whole dataset into a single array, stripping blanks and
# commas, and convertig values into floating-point numbers.
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = [float(val) for val in line.strip().split(',')]
            data.append(values)
    return data

# ToDo: Generalize for any dataset in .csv format.
def load_dataset(dataset_path):
    data_matrix     = load_data(dataset_path)
    feature_matrix  = []
    class_vector    = []

    for data_vector in data_matrix:
        feature_matrix.append(data_vector[0:616])
        class_vector.append(data_vector[617])

    return feature_matrix, class_vector

# Return the level hypervector matching a given sample.
def get_level_hypervector(feature:np.double, quantization_range:np.array, level_hypermatrix:np.array) -> np.array:
    index = np.digitize(feature, quantization_range, True)
    return level_hypermatrix[index]

def encode(feature_vector:np.array, quantization_range:np.array, level_hypermatrix:np.array, position_hypermatrix:np.array) -> np.array:
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
    similarity_vector = np.empty(number_of_classes, np.uint)

    for class_index in range(number_of_classes):
        similarity_vector[class_index] = hamming_similarity(associative_memory[class_index], query_hypervector)

    # (class, similarity)
    return (np.argmax(similarity_vector) + 1, np.max(similarity_vector))
