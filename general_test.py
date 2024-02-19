from hdcpy import *
import numpy as np

training_data_path              = 'data/isolet1+2+3+4.data'
feature_matrix, class_vector    = load_dataset(training_data_path)
number_of_features              = len(feature_matrix[0])
number_of_positions             = number_of_features
feature_vector                  = np.empty(number_of_features, np.double)
feature_vector                  = feature_matrix[0]

lower_limit                     = -1.0
upper_limit                     = 1.0
number_of_levels                = 10
number_of_dimensions            = 10000
quantization_range              = get_quantization_range(lower_limit, upper_limit, number_of_levels)
level_hypermatrix               = get_level_hypermatrix(number_of_levels, number_of_dimensions)
position_hypermatrix            = get_position_hypermatrix(number_of_features, number_of_dimensions)

for level_index in range(1, number_of_levels):
    delta = hamming_distance(level_hypermatrix[0], level_hypermatrix[level_index])
    print(f'Hamming Distance: {delta:0.4f}.')

#for level_index in range(number_of_levels - 1):
#    delta = hamming_distance(level_hypermatrix[level_index], level_hypermatrix[level_index + 1])
#    print(f'Hamming Distance: {delta:0.4f}.')

#for position_index in range(number_of_positions - 1):
#    delta = hamming_distance(position_hypermatrix[position_index], position_hypermatrix[position_index + 1])
#    print(f'Hamming Distance: {delta:0.4f}.')