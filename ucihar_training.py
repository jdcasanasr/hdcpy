from hdcpy import *
import numpy as np

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

# Dataset characteristics.
number_of_classes                   = 6
number_of_features_per_instance     = 516
number_of_total_instances           = 7352
number_of_instances_per_class       = 1407 # Some data is missing.

# Training parameters.
number_of_dimensions            = 10000
number_of_quantization_levels   = 10
quantization_lower_limit        = -1.0
quantization_upper_limit        = 1.0

# Trainig preparation.
number_of_positions             = number_of_features_per_instance

level_hypermatrix               = get_level_hypermatrix(number_of_quantization_levels, number_of_dimensions)
position_hypermatrix            = get_position_hypermatrix(number_of_positions, number_of_dimensions)
quantization_range              = get_quantization_range(quantization_lower_limit, quantization_upper_limit, number_of_quantization_levels)

associative_memory              = np.empty((number_of_classes, number_of_dimensions), np.bool_)

# Train the model in ascending order.
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

            print(f'Training For Class {class_label} | Found {number_of_class_instances} Instances So Far', end = '\r')

    # Build the class hypervector.
    associative_memory[class_label - 1] = multibundle(prototype_hypermatrix[1:][:]) # Range = [0, 25]

# Backup data for ucihar_inference_test.py.
np.save("level_hypermatrix", level_hypermatrix)
np.save("position_hypermatrix", position_hypermatrix)
np.save("quantization_range", quantization_range)
np.save("associative_memory", associative_memory)
