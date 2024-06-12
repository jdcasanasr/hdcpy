from hdcpy              import *
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

def get_base_hypermatrix(number_of_bases:np.uint, dimensionality:np.uint, vsa:np.str_) -> np.array:
    base_hypermatrix = np.empty((number_of_bases, dimensionality), np.int_)

    for index in range(number_of_bases):
        base_hypermatrix[index] = random_hypervector(dimensionality, vsa)

    return base_hypermatrix

def get_id_hypermatrix(number_of_ids:np.uint, dimensionality:np.uint, vsa:np.str_) -> np.array:
    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    id_hypermatrix    = np.empty((number_of_ids, dimensionality), np.int_)

    for index in range(number_of_ids):
        id_hypermatrix[index] = random_hypervector(dimensionality, vsa)

    return id_hypermatrix

def get_level_hypervector(feature:np.double, level_item_memory:np.array) -> np.array:
    number_of_levels    = np.shape(level_item_memory)[0]
    quantization_range  = np.linspace(-1, 1, number_of_levels)

    index = np.digitize(feature, quantization_range, False)

    if index > number_of_levels - 1:
        return level_item_memory[index - 1]
    
    else:
        return level_item_memory[index]
    
def get_base_hypervector(base:np.str_, base_dictionary:dict, base_item_memory:np.array) -> np.array:
    return base_item_memory[base_dictionary[base]]

def get_id_hypervector(id:np.uint, id_item_memory:np.array) -> np.array:
    return id_item_memory[id]

def encode_analog(bin_vector:np.array, level_item_memory:np.array, id_item_memory:np.array, vsa:np.str_) -> np.array:
    number_of_bins      = np.shape(bin_vector)[0]

    if number_of_bins != np.shape(id_item_memory)[0]:
        raise ValueError(f'Number of bins ({number_of_bins}) and number of IDs ({np.shape(id_item_memory[0])}) do not match.')

    bind_hypermatrix    = np.empty(np.shape(id_item_memory), np.int_)

    for index in range(number_of_bins):
        bind_hypermatrix[index] = bind(get_id_hypervector(index, id_item_memory), get_level_hypervector(bin_vector[index], level_item_memory), vsa)

    return multibundle(bind_hypermatrix, vsa)

def encode_discrete(base_vector:np.array, base_dictionary:dict, base_item_memory:np.array, id_item_memory:np.array, vsa:np.str_) -> np.array:
    number_of_bases     = np.shape(base_vector)[0]
    bind_hypermatrix    = np.empty(np.shape(id_item_memory), np.int_)

    for index in range(number_of_bases):
        bind_hypermatrix[index] = bind(get_id_hypervector(index, id_item_memory),
                                       get_base_hypervector(base_vector[index], base_dictionary, base_item_memory), vsa)

    return multibundle(bind_hypermatrix, vsa)

def classify(query_hypervector:np.array, associative_memory:np.array, vsa:np.str_) -> np.uint:
    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    number_of_classes = np.shape(associative_memory)[0]
    similarity_vector = np.empty(number_of_classes, np.double)

    match vsa:
        case 'BSC':
            for index in range(number_of_classes):
                similarity_vector[index] = hamming_distance(associative_memory[index], query_hypervector)

            return np.argmin(similarity_vector)

        case 'MAP':
            for index in range(number_of_classes):
                similarity_vector[index] = cosine_similarity(associative_memory[index], query_hypervector)

            return np.argmax(similarity_vector)
        
        case _:
            print('Warning: Returning a meaningless value.')

            return -1

# Version 1: Retrain All, Then Binarize.
# Version 2: Retrain And Binarize For Each Misprediction.
# Version 3: All Is Non-Binary, Except For Distance Calculation.
def retrain_analog(associative_memory:np.array, training_dataset:np.array, training_labels:np.array, level_item_memory:np.array, id_item_memory:np.array, vsa:np.str_):

    predicted_labels                = np.empty(np.shape(training_dataset)[0], np.int_)
    number_of_feature_vectors       = np.shape(training_dataset)[0]
    number_of_dimensions            = np.shape(associative_memory)[1]
    encoded_dataset                 = np.empty((number_of_feature_vectors, number_of_dimensions), np.int_)
    non_binary_associative_memory   = np.empty(np.shape(associative_memory), np.int_)

    # Pre-Encode Dataset
    for index, feature_vector in enumerate(training_dataset):
        encoded_dataset[index] = encode_analog(feature_vector, level_item_memory, id_item_memory, vsa)

    # Classify And Store Predicted Labels.
    for index, encoded_feature_vector in enumerate(encoded_dataset):
        predicted_labels[index] = classify(encoded_feature_vector, associative_memory, vsa)

    scoreboard           = True if (predicted_labels == training_labels) else False
    miss_indeces         = np.where(scoreboard == False)

    # Get Non-Binarized, Retrained Associative Memory.

    non_binary_associative_memory = np.copy(associative_memory)

    for miss_index in miss_indeces:
        correct_class       = training_labels[miss_index]
        mispredicted_class  = predicted_labels[miss_index]
        bad_query           = encoded_dataset[miss_index]

        non_binary_associative_memory[correct_class]       = np.add(associative_memory[correct_class], bad_query)
        non_binary_associative_memory[mispredicted_class]  = np.subtract(associative_memory[mispredicted_class], bad_query)

    
    return binarize(non_binary_associative_memory, vsa)

def train_analog(
    training_features:np.array,
    training_labels:np.array,
    label_array:np.array,
    label_dictionary:dict,
    id_item_memory:np.array,
    level_item_memory:np.array,
    vsa:np.str_
) -> np.array:
    dimensionality          = np.shape(id_item_memory)[1]
    number_of_classes       = np.shape(label_array)[0]
    associative_memory      = np.empty((number_of_classes, dimensionality), np.int_)

    for current_label in label_array:
        prototype_hypermatrix = np.empty(dimensionality, dtype = np.int_)

        for index, training_label in enumerate(training_labels):
            if training_label == current_label:
                prototype_hypermatrix = np.vstack((prototype_hypermatrix, encode_analog(training_features[index], level_item_memory, id_item_memory, vsa)))

        associative_memory[label_dictionary[current_label]] = multibundle(prototype_hypermatrix[1:][:], vsa)

    return associative_memory

def train_discrete(
    training_features:np.array,
    training_labels:np.array,
    label_array:np.array,
    label_dictionary:dict,
    id_item_memory:np.array,
    base_dictionary:dict,
    base_item_memory:np.array,
    vsa:np.str_
) -> np.array:
    dimensionality          = np.shape(id_item_memory)[1]
    number_of_classes       = np.shape(label_array)[0]
    associative_memory      = np.empty((number_of_classes, dimensionality), np.int_)

    for current_label in label_array:
        prototype_hypermatrix = np.empty(dimensionality, dtype = np.int_)

        for index, training_label in enumerate(training_labels):
            if training_label == current_label:
                prototype_hypermatrix = np.vstack((prototype_hypermatrix,
                                                   encode_discrete(training_features[index], base_dictionary,
                                                                   base_item_memory, id_item_memory, vsa)))

        associative_memory[label_dictionary[current_label]] = multibundle(prototype_hypermatrix[1:][:], vsa)

    return associative_memory

def test_analog(
    testing_features:np.array,
    testing_labels:np.array,
    label_dictionary:dict,
    associative_memory:np.array,
    id_item_memory:np.array,
    level_item_memory:np.array,
    vsa:np.str_
) -> np.double:
    number_of_hits  = 0
    number_of_tests = np.shape(testing_labels)[0]

    for index, query_vector in enumerate(testing_features):
        query_hypervector   = encode_analog(query_vector, level_item_memory, id_item_memory, vsa)
        predicted_class     = classify(query_hypervector, associative_memory, vsa)
        actual_class        = label_dictionary[testing_labels[index]]

        if predicted_class == actual_class:
            number_of_hits += 1

    return number_of_hits / number_of_tests * 100

def test_discrete(
    testing_features:np.array,
    testing_labels:np.array,
    label_dictionary:dict,
    associative_memory:np.array,
    id_item_memory:np.array,
    base_dictionary:dict,
    base_item_memory:np.array,
    vsa:np.str_
) -> np.double:
    number_of_hits  = 0
    number_of_tests = np.shape(testing_labels)[0]

    for index, query_vector in enumerate(testing_features):
        query_hypervector   = encode_discrete(query_vector, base_dictionary, base_item_memory, id_item_memory, vsa)
        predicted_class     = classify(query_hypervector, associative_memory, vsa)
        actual_class        = label_dictionary[testing_labels[index]]

        if predicted_class == actual_class:
            number_of_hits += 1

    return number_of_hits / number_of_tests * 100