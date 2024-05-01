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

def get_id_hypermatrix(number_of_ids:np.uint, dimensionality:np.uint, vsa:np.str_) -> np.array:
    if vsa not in supported_vsas:
        raise ValueError(f'{vsa} is not a supported VSA.')

    id_hypermatrix    = np.empty((number_of_ids, dimensionality), np.int_)

    for index in range(number_of_ids):
        id_hypermatrix[index] = random_hypervector(dimensionality, vsa)

    return id_hypermatrix

def get_level_hypervector(feature:np.double, level_item_memory:np.array) -> np.array:
    quantization_range = np.linspace(-1, 1, np.shape(level_item_memory)[0])

    return level_item_memory[np.digitize(feature, quantization_range, True)]

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
):
    number_of_queries   = np.shape(training_data)[0]
    number_of_labels    = np.shape(training_labels)[0]
    number_of_classes   = np.shape(associative_memory)[0]

    retrained_memory    = associative_memory

    if number_of_queries != number_of_labels:
        raise ValueError(f'Number of queries ({number_of_queries}) and labels ({number_of_labels}) do not match.')

    for index in range(number_of_queries):
        query_hypervector   = encode_analog(training_data[index][:], level_item_memory, id_item_memory, vsa)
        predicted_class     = classify(associative_memory, query_hypervector, vsa)
        actual_class        = equivalence_dictionary(training_labels[index])

        # Caution: We assume training labels start from zero!
        if predicted_class != actual_class:
            retrained_memory[predicted_class] = np.subtract(associative_memory[predicted_class], query_hypervector)
            retrained_memory[actual_class]    = np.add(associative_memory[actual_class], query_hypervector)

    # Once retraining is done, re-binarize the associative memory.
    match vsa:
        case 'BSC':
            for index in range(number_of_classes):
                retrained_memory[index] = np.where(retrained_memory[index] >= 2, 1, 0)

            return retrained_memory

        case 'MAP':
            for index in range(number_of_classes):
                retrained_memory[index] = np.sign(retrained_memory[index])

            return retrained_memory

        case _:
            print('Warning: Retraining not performed')

            return associative_memory