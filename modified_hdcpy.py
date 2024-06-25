import numpy    as np
import os

from sklearn.datasets           import fetch_openml
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder
from sklearn.preprocessing      import MinMaxScaler

supported_vsas = ['BSC', 'MAP']

def random_hypervector(dimensionality:np.int_, vsa:np.str_) -> np.array:
    match vsa:
        case 'BSC':
            return np.random.choice([0, 1], size = dimensionality, p = [0.5, 0.5])
        
        case 'MAP':
            return np.random.choice([-1, 1], size = dimensionality, p = [0.5, 0.5])

def hamming_distance(u:np.array, v:np.array) -> np.float_:
    return  np.sum(u != v) / np.shape(u)[0]

def cosine_similarity(u:np.array, v:np.array) -> np.float_:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def bind(u:np.array, v:np.array, vsa:np.str_) -> np.array:
    match vsa:
        case 'BSC':
            return np.bitwise_xor(u, v)
    
        case 'MAP':
            return np.multiply(u, v, dtype = np.int_)
     
def bundle(u:np.array, v:np.array, vsa:np.str_) -> np.array:
    match vsa:
        case 'BSC':
            w = random_hypervector(np.shape(u)[0], vsa)

            return np.where(np.add(u, v, w) >= 2, 1, 0)
        
        case 'MAP':
            w = random_hypervector(np.shape(u)[0], vsa)

            return np.sign(np.add(u, v, w))
        
def permute(u:np.array, shift_amount:np.int_) -> np.array:
    return np.roll(u, shift_amount)
    
def multibundle(hypermatrix: np.array, vsa: np.str_) -> np.array:
    number_of_hypervectors = hypermatrix.shape[0]
    dimensionality = hypermatrix.shape[1]
    half_number_of_hypervectors = number_of_hypervectors / 2

    if number_of_hypervectors % 2 == 0:
        tie_break = random_hypervector(dimensionality, vsa)
        summed_matrix = np.sum(hypermatrix, axis=0) + tie_break
    else:
        summed_matrix = np.sum(hypermatrix, axis=0)
    
    if vsa == 'BSC':
        threshold = half_number_of_hypervectors
        return np.where(summed_matrix >= threshold, 1, 0)
    
    if vsa == 'MAP':
        return np.sign(summed_matrix)
        
def flip(u: np.ndarray, number_of_positions: int, vsa: str) -> np.ndarray:
    dimensionality = u.shape[0]
    flip_positions = np.random.choice(dimensionality, size=number_of_positions, replace=False)
    
    flip_hypervector = u.copy()
    if vsa == 'BSC':
        flip_hypervector[flip_positions] = 1 - flip_hypervector[flip_positions]
    elif vsa == 'MAP':
        flip_hypervector[flip_positions] = -flip_hypervector[flip_positions]
    else:
        raise ValueError(f"Unsupported VSA type: {vsa}")
    
    return flip_hypervector

def binarize(u:np.array, vsa:np.str_) -> np.array:
    match vsa:
        case 'BSC':
            return np.where(u >= 1, 1, 0)

        case 'MAP':
            return np.where(u > 0, 1, -1)

def get_dataset(dataset_name:np.str_, save_directory:np.str_, test_proportion:np.float_):
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    file_path = os.path.join(save_directory, f'{dataset_name}.csv')
    label_encoder = LabelEncoder()
    feature_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))

    if not os.path.exists(file_path):
        # Fetch and save the dataset if it doesn't exist
        dataset = fetch_openml(name=dataset_name, version=1, as_frame=False)
        data = dataset.data
        target = dataset.target

        dataset_array = np.column_stack((data, target))
        np.savetxt(file_path, dataset_array, delimiter=',', fmt='%s')

    # Load the dataset from the file
    dataset_array = np.genfromtxt(file_path, delimiter=',', dtype=np.str_)
    data = dataset_array[:, :-1].astype(np.float_)
    target = dataset_array[:, -1]
    target_encoded = label_encoder.fit_transform(target)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target_encoded, test_size=test_proportion, random_state=42)

    # Scale the features
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def get_number_of_classes(training_features:np.array) -> np.int_:
    return np.size(np.unique(training_features))


def get_level_hypermatrix(number_of_levels: int, dimensionality: int, vsa: str) -> np.ndarray:
    number_of_flip_positions = np.ceil(dimensionality / (number_of_levels - 1)).astype(int)
    level_hypermatrix = np.empty((number_of_levels, dimensionality), dtype=int)
    
    # Generate the initial random hypervector
    level_hypermatrix[0] = random_hypervector(dimensionality, vsa)
    
    # Generate subsequent levels by flipping positions in the previous level
    for index in range(1, number_of_levels):
        level_hypermatrix[index] = flip(level_hypermatrix[index - 1], number_of_flip_positions, vsa)
    
    return level_hypermatrix

def get_id_hypermatrix(number_of_ids: np.uint, dimensionality: np.uint, vsa: np.str_) -> np.array:
    if vsa == 'BSC':
        id_hypermatrix = np.random.choice([0, 1], size=(number_of_ids, dimensionality), p=[0.5, 0.5])
    elif vsa == 'MAP':
        id_hypermatrix = np.random.choice([-1, 1], size=(number_of_ids, dimensionality), p=[0.5, 0.5])
    else:
        raise ValueError(f"Unsupported VSA type: {vsa}")
    
    return id_hypermatrix

def get_level_hypervector(feature: np.double, level_item_memory: np.array) -> np.array:
    number_of_levels = level_item_memory.shape[0]
    quantization_range = np.linspace(-1, 1, number_of_levels, endpoint=False)

    index = np.digitize(feature, quantization_range)

    if index >= number_of_levels:
        return level_item_memory[number_of_levels - 1]
    else:
        return level_item_memory[index]

def get_id_hypervector(id:np.uint, id_item_memory:np.array) -> np.array:
    return id_item_memory[id]

def encode_analog(feature_vector:np.array, number_of_features:np.int_, level_item_memory:np.array, id_item_memory:np.array, vsa:np.str_) -> np.array:
    bind_hypermatrix = np.empty(np.shape(id_item_memory), np.int_)

    for index in range(number_of_features):
        bind_hypermatrix[index] = bind(get_id_hypervector(index, id_item_memory), get_level_hypervector(feature_vector[index], level_item_memory), vsa)

    return multibundle(bind_hypermatrix, vsa)

def encode_dataset(dataset:np.array, number_of_dimensions:np.int_, level_item_memory:np.array, id_item_memory:np.array, vsa:np.str_) -> np.array:
    number_of_feature_vectors   = np.shape(dataset)[0]
    number_of_features          = np.shape(dataset)[1]
    encoded_dataset             = np.empty((number_of_feature_vectors, number_of_dimensions), np.int_)

    for index, feature_vector in enumerate(dataset):
        encoded_dataset[index] = encode_analog(feature_vector, number_of_features, level_item_memory, id_item_memory, vsa)

    return encoded_dataset

def classify(query_hypervector:np.array, associative_memory:np.array, vsa:np.str_) -> np.uint:
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

def train_analog(encoded_training_dataset: np.array, training_labels: np.array, number_of_classes: int, number_of_dimensions: int, vsa: str) -> np.array:
    associative_memory = np.empty((number_of_classes, number_of_dimensions), np.int_)

    for current_label in range(number_of_classes):
        # Select all vectors corresponding to the current label
        label_mask = (training_labels == current_label)
        prototype_hypermatrix = encoded_training_dataset[label_mask]

        # Bundle the prototype_hypermatrix to form the associative memory for the current label
        associative_memory[current_label] = multibundle(prototype_hypermatrix, vsa)

    return associative_memory

def test_analog(encoded_testing_dataset:np.array, testing_labels:np.array, associative_memory:np.array, vsa:np.str_) -> np.float_:
    number_of_hits  = 0
    number_of_tests = np.shape(testing_labels)[0]

    for index, query_hypervector in enumerate(encoded_testing_dataset):
        predicted_class     = classify(query_hypervector, associative_memory, vsa)
        actual_class        = testing_labels[index]

        if predicted_class == actual_class:
            number_of_hits += 1

    return number_of_hits / number_of_tests * 100