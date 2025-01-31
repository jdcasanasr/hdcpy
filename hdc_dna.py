from hdcpy import *

def encode_dna_sequence(dna_sequence:np.str_, nucleotid_dictionary:dict, id_hypermatrix:np.array, dimensionality:np.int_, vsa:np.str_) -> np.array:
    
    sequence_length     = np.shape(dna_sequence)[0]
    string_hypermatrix  = np.empty((sequence_length, dimensionality), np.int_)

    for index, nucleotid in enumerate(dna_sequence):
        string_hypermatrix[index] = bind(id_hypermatrix[index], nucleotid_dictionary[nucleotid], vsa)

    return multibundle(string_hypermatrix, vsa)

def train_dna(encoded_X_train:np.str_, y_train:np.str_, dimensionality:np.int_, vsa:np.str_) -> np.array:
    number_of_classes       = len(np.unique(y_train))
    associative_memory      = np.empty((number_of_classes, dimensionality), np.int_)

    for current_label in range(number_of_classes):
        # Select all vectors corresponding to the current label
        prototype_hypermatrix   = encoded_X_train[(y_train == current_label)]

        # Bundle the prototype_hypermatrix to form the associative memory for the current label
        associative_memory[current_label] = multibundle(prototype_hypermatrix, vsa)

    return associative_memory

def classify_dna(query_hypervector:np.array, associative_memory:np.array) -> np.int_:
    distances = []

    for class_hypervector in associative_memory:
        distances.append(hamming_distance(query_hypervector, class_hypervector))

    return np.argmin(distances)

def test_dna(X_test:np.array, y_test:np.array, associative_memory:np.array, nucleotid_dictionary:dict, dimensionality:np.int_, vsa:np.str_)-> float:
    hits    = 0
    tests   = np.shape(y_test)[0]

    for index, label in enumerate(y_test):
        if (label == classify_dna(X_test[index], associative_memory)):
            hits += 1

    return hits / tests