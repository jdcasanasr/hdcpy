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

# Note: An 'equivalence dictionary' is required to transform
# any label (a str-type object) into an integer number 
# between [0, number_of_classes).
def retrain_analog(
    associative_memory:np.array,
    training_data:np.array,
    training_labels:np.array,
    level_item_memory:np.array,
    id_item_memory:np.array,
    equivalence_dictionary:dict,
    vsa:np.str_
) -> np.array:
    number_of_queries   = np.shape(training_data)[0]
    number_of_labels    = np.shape(training_labels)[0]
    number_of_classes   = np.shape(associative_memory)[0]

    retrained_memory    = associative_memory

    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    if number_of_queries != number_of_labels:
        raise ValueError(f'Number of queries ({number_of_queries}) and labels ({number_of_labels}) do not match.')

    for index in range(number_of_queries):
        query_hypervector   = encode_analog(training_data[index][:], level_item_memory, id_item_memory, vsa)
        predicted_class     = classify(query_hypervector, associative_memory, vsa)
        actual_class        = equivalence_dictionary[training_labels[index]]

        # Caution: We assume training labels start from zero!
        if predicted_class != actual_class:
            retrained_memory[predicted_class] = np.subtract(associative_memory[predicted_class], query_hypervector)
            retrained_memory[actual_class]    = np.add(associative_memory[actual_class], query_hypervector)

    # Once retraining is done, re-binarize the associative memory.
    for index in range(number_of_classes):
        retrained_memory[index] = binarize(retrained_memory[index], vsa)

    return retrained_memory

def retrain_discrete(
    associative_memory:np.array,
    training_data:np.array,
    training_labels:np.array,
    base_item_memory:np.array,
    base_dictionary:dict,
    id_item_memory:np.array,
    equivalence_dictionary:dict,
    vsa:np.str_
) -> np.array:
    
    number_of_queries   = np.shape(training_data)[0]
    number_of_labels    = np.shape(training_labels)[0]
    number_of_classes   = np.shape(associative_memory)[0]

    retrained_memory    = associative_memory

    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    if number_of_queries != number_of_labels:
        raise ValueError(f'Number of queries ({number_of_queries}) and labels ({number_of_labels}) do not match.')

    for index, training_sample in enumerate(training_data):
        query_hypervector   = encode_discrete(training_sample, base_dictionary, base_item_memory, id_item_memory, vsa)
        predicted_class     = classify(query_hypervector, associative_memory, vsa)
        actual_class        = equivalence_dictionary[training_labels[index]]

        # Caution: We assume training labels start from zero!
        if predicted_class != actual_class:
            retrained_memory[predicted_class] = np.subtract(associative_memory[predicted_class], query_hypervector)
            retrained_memory[actual_class]    = np.add(associative_memory[actual_class], query_hypervector)

    # Once retraining is done, re-binarize the associative memory.
    for index in range(number_of_classes):
        retrained_memory[index] = binarize(retrained_memory[index], vsa)

    return retrained_memory