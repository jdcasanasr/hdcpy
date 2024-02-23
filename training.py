from hdcpy import *
import pandas as pd
import numpy as np

feature_matrix  = pd.read_csv("data/X_train.txt", delim_whitespace=True, header=None).values
class_vector    = pd.read_csv("data/y_train.txt", delim_whitespace=True, header=None).values

# Get useful parameters.
number_of_features  = np.shape(feature_matrix)[1]
number_of_classes   = np.max(class_vector)

# Training parameters.
number_of_dimensions            = 10000
number_of_levels                = 10
level_lower_limit               = -1.0
level_upper_limit               = 1.0
number_of_positions             = number_of_features

class_instance_vector           = np.zeros(number_of_classes, np.uint)

# Compute the number of instances per class.
for class_index in range(1, number_of_classes + 1):
    for class_instance in class_vector:
        if class_instance == class_index:
            class_instance_vector[class_index - 1] += 1

number_of_instances_per_class = np.max(class_instance_vector)

print(class_instance_vector)

#level_hypermatrix               = get_level_hypermatrix(number_of_levels, number_of_dimensions)
#position_hypermatrix            = get_position_hypermatrix(number_of_positions, number_of_dimensions)
#quantization_range              = get_quantization_range(level_lower_limit, level_upper_limit, number_of_levels)
#
#associative_memory              = np.empty((number_of_classes, number_of_dimensions), np.bool_)
#prototype_hypermatrix           = np.empty((number_of_instances_per_class, number_of_dimensions), np.bool_)
#
## Iterate over each class.
#for class_key in range(1, number_of_classes + 1): # Range = [1, 26]
#    prototype_index = 0
#
#    # Iterate over 'class_vector'.
#    for class_vector_index in range(len(class_vector)): # Range = [0, 6237]
#        # Train the model for each class in order.
#        if (alphabet[class_key] == alphabet[int(class_vector[class_vector_index])]) and (prototype_index < number_of_instances_per_class):
#            prototype_hypermatrix[prototype_index] = encode(feature_matrix[class_vector_index], quantization_range, level_hypermatrix, position_hypermatrix)
#            prototype_index += 1 # Range = [0, 239]
#            print(f'Training for class {alphabet[class_key]} | Found {prototype_index} of 240 Instances.', end = '\r')
#
#    # Build the class hypervector.
#    associative_memory[class_key - 1] = multibundle(prototype_hypermatrix) # Range = [0, 25]
#
## Backup data for inference_test.py.
#np.save("level_hypermatrix", level_hypermatrix)
#np.save("position_hypermatrix", position_hypermatrix)
#np.save("quantization_range", quantization_range)
#np.save("associative_memory", associative_memory)