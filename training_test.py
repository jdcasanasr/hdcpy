from hdcpy import *

# Step 0: Preparation
# Define a dictionary for better human readibility.
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

#1 Load data from ISOLET dataset.
training_data_path              = 'data/isolet1+2+3+4.data'
feature_matrix, class_vector    = load_dataset(training_data_path)

#2 Train the model.
number_of_dimensions        = 10000
number_of_levels            = 100
level_lower_limit           = -1
level_upper_limit           = 1
number_of_features          = len(feature_matrix[0])
number_of_classes           = int(np.max(class_vector))

level_hypermatrix           = get_level_hypermatrix(number_of_dimensions, number_of_levels)
position_hypermatrix        = get_position_hypermatrix(number_of_dimensions, number_of_features)
quantization_range          = get_quantization_range(level_lower_limit, level_upper_limit, number_of_levels)

# Backup data for inference test.
np.save("level_hypermatrix", level_hypermatrix)
np.save("position_hypermatrix", position_hypermatrix)
np.save("quantization_range", quantization_range)

class_hypervector           = np.empty(number_of_classes, np.bool_)
associative_memory          = np.empty((number_of_classes, number_of_dimensions), np.bool_)
bind_hypermatrix            = np.empty((number_of_features, number_of_dimensions), np.bool_)

# Traverse the entire class_vector array in search for letters.
for key in range(1, number_of_classes):
    number_of_instances = 0

    for class_index in range(len(class_vector)):
        if alphabet[int(class_vector[class_index])] == alphabet[key]:
            # Auxiliary code to check the program hasn't hung.
            number_of_instances += 1
            print(f'Found {number_of_instances} instances of class {alphabet[key]}', end = '\r')

            # Generate the class hypervector for the letter 'a'.
            feature_vector = feature_matrix[class_index]

            for feature_index in range(len(feature_vector)):
                feature                                 = feature_vector[feature_index]
                position_hypervector                    = position_hypermatrix[feature_index]
                level_hypervector                       = quantize_feature(feature, quantization_range, level_hypermatrix)

                bind_hypermatrix[feature_index]         = bind(position_hypervector, level_hypervector)

    associative_memory[key] = multibundle(bind_hypermatrix)

# Save vector to a file.
np.save("associative_memory", associative_memory)