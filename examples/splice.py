import numpy as np
import sys
import time as tm

from hdcpy import *

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

    return multibundle(hypermatrix)

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
dataset_path    = '../data'
test_proportion = 0.2

dataset         = fetch_openml(name=dataset_name, version=1, parser='auto')
data, target    = dataset.data, dataset.target
training_features, testing_features, training_labels, testing_labels = train_test_split(data, target, test_size=test_proportion)

# Extract values from dataframes.
training_features   = training_features.values
testing_features    = testing_features.values
training_labels     = training_labels.values
testing_labels      = testing_labels.values

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