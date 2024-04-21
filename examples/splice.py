import numpy as np
import time as tm
import os

from hdcpy import *

from sklearn.datasets           import fetch_openml
from sklearn.model_selection    import train_test_split

def fetch_dataset(dataset_name: str, save_directory:str, test_proportion:float):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_path = os.path.join(save_directory, f'{dataset_name}.csv')

    # Check if the dataset already exists, and if not,
    # fetch it from the internet.
    if not os.path.exists(file_path):
        dataset         = fetch_openml(name = dataset_name, version = 1, parser = 'auto')
        data, target    = np.array(dataset.data), np.array(dataset.target)

        training_features, testing_features, training_labels, testing_labels = train_test_split(data, target, test_size = test_proportion)

        # Combine data and target into a single array, and
        # store it in .csv format.
        dataset_array = np.column_stack((data, target))

        np.savetxt(file_path, dataset_array, delimiter = ',', fmt = '%s')

        return training_features, testing_features, training_labels, testing_labels
    
    else:
        # Read pre-existing ".csv" file, split it into features and labels
        # and return both as numpy arrays.
        dataset_array   = np.genfromtxt(file_path, delimiter = ',', dtype = np.str_)
        data            = dataset_array[:, :-1]
        target          = dataset_array[:, -1]

        training_features, testing_features, training_labels, testing_labels = train_test_split(data, target, test_size = test_proportion)

        return training_features, testing_features, training_labels, testing_labels

def encode_dna_sequence (input_sequence:np.array, nucleotides:dict, item_memory:np.array) -> np.array:
    # 1. Read every element in 'input_sequence'.
    # 2. Depending on its position, bind the proper nucleotide[element] and item_memory[index].
    # 3. Stack each encoded element.
    # 4. Once done, multibundle the stack.
    input_sequence_iterator = np.nditer(input_sequence, flags = ['c_index', 'refs_ok'])
    hypermatrix             = np.empty(np.shape(item_memory)[1], dtype = np.bool_)

    for nucleotide in input_sequence_iterator:
        bind_hypervector    = bind(nucleotides[str(nucleotide)], item_memory[input_sequence_iterator.index])
        hypermatrix         = np.vstack((hypermatrix, bind_hypervector))

    return multibundle(hypermatrix[1:][:])

number_of_dimensions    = 10000

nucleotides = {
    # Nucleotides.
    'A' : random_hypervector(number_of_dimensions),
    'C' : random_hypervector(number_of_dimensions),
    'G' : random_hypervector(number_of_dimensions),
    'T' : random_hypervector(number_of_dimensions),

    # "Ambiguous" nucleotide placeholders.
    'D' : random_hypervector(number_of_dimensions),
    'N' : random_hypervector(number_of_dimensions),
    'S' : random_hypervector(number_of_dimensions),
    'R' : random_hypervector(number_of_dimensions)
}

labels = {'EI', 'IE', 'N'}

label_dictionary = {
    'EI'    : 0,
    'IE'    : 1,
    'N'     : 2
}

# Fetch and split dataset.
dataset_name    = 'splice'
dataset_path    = '/home/jdcasanasr/Development/hdcpy/data'
test_proportion = 0.2

training_features, testing_features, training_labels, testing_labels = fetch_dataset(dataset_name, dataset_path, test_proportion)

# Populate item memory with ID hypervectors.
item_memory_size    = np.shape(training_features)[0]
item_memory         = np.empty((item_memory_size, number_of_dimensions))

associative_memory = np.empty((3, number_of_dimensions), np.bool_)

for index in range(item_memory_size):
    item_memory[index] = random_hypervector(number_of_dimensions)

testing_times                   = []

# Train the model in label order.
training_time_begin = tm.time()
for label in labels:
    prototype_hypermatrix       = np.empty(number_of_dimensions, dtype = np.bool_)
    training_labels_iterator    = np.nditer(training_labels, flags = ['c_index', 'refs_ok'])

    # 1. Traverse the training_label array.
    # 2. Check if the label matches our current label.
    # 3. If so, transform the vector in the label's index.
    # 4. Store it in the prototype hypermatrix.
    for training_label in training_labels_iterator:
        if training_label == label:
            prototype_hypervector = encode_dna_sequence(training_features[training_labels_iterator.index], nucleotides, item_memory)
            prototype_hypermatrix = np.vstack((prototype_hypermatrix, prototype_hypervector))

    # Build the class hypervector and store it.
    associative_memory[label_dictionary[label]] = multibundle(prototype_hypermatrix[1:][:])
training_time_end = tm.time()

# Inference stage.
number_of_correct_predictions   = 0
number_of_tests                 = 0
testing_features_iterator       = np.nditer(training_labels, flags = ['c_index', 'refs_ok'])

# 1. Encode the current feature vector.
# 2. Classify the encoded vector.
# 3. Compare both the predicted and the actual label.
for feature_index in range(np.shape(testing_features)[0]):
    testing_time_begin  = tm.time()
    query_hypervector   = encode_dna_sequence(testing_features[feature_index], nucleotides, item_memory)
    predicted_class     = classify(associative_memory, query_hypervector)
    number_of_tests     += 1

    if predicted_class == label_dictionary[testing_labels[feature_index]]:
        number_of_correct_predictions += 1

    testing_time_end = tm.time()
    testing_times.append(testing_time_end - testing_time_begin)

 # Compute performance metrics and output to console.
    accuracy                        = number_of_correct_predictions / number_of_tests * 100
    training_time_per_dataset       = training_time_end - training_time_begin
    average_testing_time_per_query  = sum(testing_times) / len(testing_times)

print(f'{number_of_dimensions},{8},{accuracy:0.2f},{training_time_per_dataset:0.6f},{average_testing_time_per_query:0.6f}')