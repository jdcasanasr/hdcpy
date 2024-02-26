from hdcpy import *

alphabet = {

    1   : 'a',
    2   : 'b',
    3   : 'c',
    4   : 'd',
    5   : 'e',
    6   : 'f',
    7   : 'g',
    8   : 'h',
    9   : 'i',
    10  : 'j',
    11  : 'k',
    12  : 'l',
    13  : 'm',
    14  : 'n',
    15  : 'o',
    16  : 'p',
    17  : 'q',
    18  : 'r',
    19  : 's',
    20  : 't',
    21  : 'u',
    22  : 'v',
    23  : 'w',
    24  : 'x',
    25  : 'y',
    26  : 'z'
}

training_data_path              = 'data/ISOLET/isolet1+2+3+4.data'
feature_matrix, class_vector    = load_dataset(training_data_path)

# ISOLET dataset parameters.
number_of_data_elements         = 6238
number_of_classes               = 26
number_of_instances_per_class   = 240
number_of_features              = 616

# Training parameters.
number_of_dimensions            = 9000
number_of_levels                = 10
level_lower_limit               = -1.0
level_upper_limit               = 1.0
number_of_positions             = number_of_features

level_hypermatrix               = get_level_hypermatrix(number_of_levels, number_of_dimensions)
position_hypermatrix            = get_position_hypermatrix(number_of_positions, number_of_dimensions)
quantization_range              = get_quantization_range(level_lower_limit, level_upper_limit, number_of_levels)

associative_memory              = np.empty((number_of_classes, number_of_dimensions), np.bool_)
prototype_hypermatrix           = np.empty((number_of_instances_per_class, number_of_dimensions), np.bool_)

# Iterate over each class.
for class_key in range(1, number_of_classes + 1): # Range = [1, 26]
    prototype_index = 0

    # Iterate over 'class_vector'.
    for class_vector_index in range(len(class_vector)): # Range = [0, 6237]
        # Train the model for each class in order.
        if (alphabet[class_key] == alphabet[int(class_vector[class_vector_index])]) and (prototype_index < number_of_instances_per_class):
            prototype_hypermatrix[prototype_index] = encode(feature_matrix[class_vector_index], quantization_range, level_hypermatrix, position_hypermatrix)
            prototype_index += 1 # Range = [0, 239]
            print(f'Training for class {alphabet[class_key]} | Found {prototype_index} of 240 Instances.', end = '\r')

    # Build the class hypervector.
    associative_memory[class_key - 1] = multibundle(prototype_hypermatrix) # Range = [0, 25]

# Backup data for inference_test.py.
np.save("level_hypermatrix", level_hypermatrix)
np.save("position_hypermatrix", position_hypermatrix)
np.save("quantization_range", quantization_range)
np.save("associative_memory", associative_memory)