from hdcpy import *
import sys
import time

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


training_feature_data, training_class_data  = load_dataset('/home/jdcasanasr/Development/hdc_profiling/data_dir/data/isolet/isolet1+2+3+4.data')
testing_feature_data, testing_class_data    = load_dataset('/home/jdcasanasr/Development/hdc_profiling/data_dir/data/isolet/isolet5.data')

# ISOLET dataset parameters.
number_of_data_elements         = 6238
number_of_classes               = 26
number_of_instances_per_class   = 240
number_of_features              = 616

# Model parameters.
number_of_dimensions            = int(sys.argv[1])
number_of_levels                = int(sys.argv[2])
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
    for class_vector_index in range(len(training_class_data)): # Range = [0, 6237]
        # Train the model for each class in order.
        if (alphabet[class_key] == alphabet[int(training_class_data[class_vector_index])]) and (prototype_index < number_of_instances_per_class):
            prototype_hypermatrix[prototype_index] = encode(training_feature_data[class_vector_index], quantization_range, level_hypermatrix, position_hypermatrix)
            prototype_index += 1 # Range = [0, 239]
            #print(f'Training for class {alphabet[class_key]} | Found {prototype_index} of 240 Instances.', end = '\r')

    # Build the class hypervector.
    associative_memory[class_key - 1] = multibundle(prototype_hypermatrix) # Range = [0, 25]

number_of_data_elements         = len(testing_class_data)
number_of_classes               = 26
number_of_correct_predictions   = 0
number_of_class_instances       = 0
times                           = []

# Find all instances of letter a' in feature_matrix, and test classification.
for test_class in range(1, number_of_classes + 1):
    for class_index in range(len(testing_class_data)):
        if alphabet[int(testing_class_data[class_index])] == alphabet[test_class]:
            start = time.time()
            feature_vector = testing_feature_data[class_index]
            number_of_class_instances += 1
            #print(f'Instances: {number_of_class_instances} of {number_of_data_elements} | Correct Predictions: {number_of_correct_predictions} | Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%', end = '\r')

            # We all know it belongs to class 'a'!.
            query_hypervector = encode(feature_vector, quantization_range, level_hypermatrix, position_hypermatrix)

            # Classify the unknown hypervector.
            predicted_class = classify(associative_memory, query_hypervector)
            if (alphabet[predicted_class] == alphabet[test_class]):
                number_of_correct_predictions += 1
            end = time.time()
            times.append(end - start)
            #print(f'Elapsed Time per Query: {sum(times) / len(times):0.6f}')


print(f'Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%')
#print(f'{(number_of_correct_predictions / number_of_data_elements) * 100:0.2f}')