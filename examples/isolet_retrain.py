import numpy as np
import sys
import time as tm

from hdcpy import *

if __name__ == "__main__":
    dataset_name    = 'isolet'
    dataset_path    = '../data'
    test_proportion = 0.2

    training_features, testing_features, training_labels, testing_labels = fetch_dataset(dataset_name, dataset_path, test_proportion)

    # Dataset characteristics.
    number_of_classes           = np.max(training_labels) + 1
    number_of_features          = np.shape(training_features)[1]
    number_of_testing_vectors   = np.shape(testing_features)[0]
    signal_minimum_level        = -1.0
    signal_maximum_level        = 1.0

    # Training preparation.
    number_of_dimensions            = int(sys.argv[1])
    number_of_quantization_levels   = int(sys.argv[2])

    level_hypermatrix       = get_level_hypermatrix(number_of_quantization_levels, number_of_dimensions)
    position_hypermatrix    = get_position_hypermatrix(number_of_features, number_of_dimensions)
    quantization_range      = get_quantization_range(signal_minimum_level, signal_maximum_level, number_of_quantization_levels)

    associative_memory      = np.empty((number_of_classes, number_of_dimensions), np.bool_)

    # Testing preparation.
    number_of_correct_predictions = 0

    # Performance Metrics.
    accuracy                        = None
    training_time_per_dataset       = None
    average_testing_time_per_query  = None
    testing_times                   = []

    # Training stage.
    training_time_begin = tm.time()

    # Train the model in class order (1, 2, 3, ...).
    for class_index in range(number_of_classes):
        prototype_hypermatrix       = np.empty(number_of_dimensions, dtype = np.bool_)
        training_labels_iterator    = np.nditer(training_labels, flags = ['c_index'])

        for label in training_labels_iterator:
            if class_index == label:
                prototype_hypervector = encode_analog(training_features[training_labels_iterator.index], quantization_range, level_hypermatrix, position_hypermatrix)
                prototype_hypermatrix = np.vstack((prototype_hypermatrix, prototype_hypervector))

        # Build the class hypervector and store it.
        associative_memory[class_index] = multibundle(prototype_hypermatrix[1:][:])

    training_time_end = tm.time()

    # Retraining stage.
    retrain_analog(associative_memory, training_features, training_labels, quantization_range, level_hypermatrix, position_hypermatrix)

    # Inference stage.
    for test_index in range(number_of_testing_vectors):
        testing_time_begin = tm.time()

        feature_vector              = testing_features[test_index][:]
        actual_class                = testing_labels[test_index]
        query_hypervector           = encode_analog(feature_vector, quantization_range, level_hypermatrix, position_hypermatrix)
        predicted_class             = classify(associative_memory, query_hypervector)

        if predicted_class == actual_class:
            number_of_correct_predictions += 1

        testing_time_end = tm.time()
        testing_times.append(testing_time_end - testing_time_begin)

    # Compute performance metrics and output to console.
    accuracy                        = number_of_correct_predictions / number_of_testing_vectors * 100
    training_time_per_dataset       = training_time_end - training_time_begin
    average_testing_time_per_query  = sum(testing_times) / len(testing_times)

    print(f'{number_of_dimensions},{number_of_quantization_levels},{accuracy:0.2f},{training_time_per_dataset:0.6f},{average_testing_time_per_query:0.6f}')