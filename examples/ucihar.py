from hdcpy import *
import numpy as np
import sys

def load_feature_data(file_path):
    with open(file_path, 'r') as f:
        data = [line.strip().split() for line in f]
    return np.array(data, dtype = np.double)

def load_class_data(file_path):
    with open(file_path, 'r') as f:
        data = [line.strip().split()[0] for line in f]
    return np.array(data, dtype = np.uint)

training_feature_data   = load_feature_data('/home/jdcasanasr/Development/hdcpy/data/UCIHAR/X_train.txt')
training_class_data     = load_class_data('/home/jdcasanasr/Development/hdcpy/data/UCIHAR/y_train.txt')

testing_feature_data    = load_feature_data('/home/jdcasanasr/Development/hdcpy/data/UCIHAR/X_test.txt')
testing_class_data      = load_class_data('/home/jdcasanasr/Development/hdcpy/data/UCIHAR/y_test.txt')

# Dataset characteristics.
number_of_classes                   = 6
number_of_features_per_instance     = 561
number_of_total_instances           = 7352
number_of_instances_per_class       = 1407 # Some data is missing.

# Training parameters.
number_of_dimensions            = int(sys.argv[1])  # 8000
number_of_quantization_levels   = int(sys.argv[2])  # 9
quantization_lower_limit        = -1.0
quantization_upper_limit        = 1.0

number_of_data_elements         = np.shape(testing_feature_data)[0]
number_of_correct_predictions   = 0
number_of_class_instances       = 0

# Trainig preparation.
number_of_positions             = number_of_features_per_instance

level_hypermatrix               = get_level_hypermatrix(number_of_quantization_levels, number_of_dimensions)
position_hypermatrix            = get_position_hypermatrix(number_of_positions, number_of_dimensions)
quantization_range              = get_quantization_range(quantization_lower_limit, quantization_upper_limit, number_of_quantization_levels)

associative_memory              = np.empty((number_of_classes, number_of_dimensions), np.bool_)

# Training test.
for class_label in range(1, number_of_classes + 1): # Range = [1, 6]
    prototype_hypermatrix           = np.empty(number_of_dimensions, dtype = np.bool_)
    training_class_data_iterator    = np.nditer(training_class_data, flags = ['c_index'])
    number_of_class_instances       = 0

    # Iterate over all classes in the training dataset.
    for class_item in training_class_data_iterator:
        if class_item == class_label:
            prototype_hypervector = encode(training_feature_data[training_class_data_iterator.index], quantization_range, level_hypermatrix, position_hypermatrix)
            prototype_hypermatrix = np.vstack((prototype_hypermatrix, prototype_hypervector))

            number_of_class_instances += 1

            #print(f'Training For Class {class_label} | Found {number_of_class_instances} Instances So Far', end = '\r')

    # Build the class hypervector.
    associative_memory[class_label - 1] = multibundle(prototype_hypermatrix[1:][:]) # Range = [0, 25]

# Inference test.
number_of_class_instances = 0
for index in range(number_of_data_elements):
    number_of_class_instances   += 1
    feature_vector              = testing_feature_data[index][:]
    actual_class                = testing_class_data[index]
    query_hypervector           = encode(feature_vector, quantization_range, level_hypermatrix, position_hypermatrix)
    predicted_class             = classify(associative_memory, query_hypervector)

    if predicted_class == actual_class:
        number_of_correct_predictions += 1

    #print(f'Instances: {number_of_class_instances} of {number_of_data_elements} | Correct Predictions: {number_of_correct_predictions} | Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%', end = '\r')

#print(f'Instances: {number_of_class_instances} of {number_of_data_elements} | Correct Predictions: {number_of_correct_predictions} | Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%')
print(f'{(number_of_correct_predictions / number_of_data_elements) * 100:0.2f}')