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

training_data_path              = 'data/isolet1+2+3+4.data'
feature_matrix, class_vector    = load_dataset(training_data_path)

# Training parameters for the ISOLET dataset.
number_of_dimensions        = 10000
number_of_classes           = 26
number_of_levels            = 10
number_of_positions         = 616
number_of_features          = number_of_positions
level_lower_limit           = -1.0
level_upper_limit           = 1.0
number_of_class_instances   = 240

level_hypermatrix           = get_level_hypermatrix(number_of_levels, number_of_dimensions)
position_hypermatrix        = get_position_hypermatrix(number_of_positions, number_of_dimensions)
quantization_range          = get_quantization_range(level_lower_limit, level_upper_limit, number_of_levels)

associative_memory          = np.empty((number_of_classes, number_of_dimensions), np.bool_)
prototype_tensor            = np.empty((number_of_class_instances, number_of_dimensions, number_of_classes), np.bool_)

print('Hello!')