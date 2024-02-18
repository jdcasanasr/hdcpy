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
dimensionality              = 10000
quantization_levels         = 100
quantization_lower_limit    = -1
quantization_upper_limit    = 1
seed_hypervector            = generate_hypervector(dimensionality)
number_of_features          = len(feature_matrix[0])
number_of_classes           = int(np.max(class_vector))
number_of_instances         = 0
number_of_features          = len(feature_matrix[0])

level_hypervector_array     = generate_level_hypervectors(seed_hypervector, quantization_levels)
id_hypervector_array        = generate_id_hypervectors(dimensionality, number_of_features)
quantized_range             = quantize_range(quantization_lower_limit, quantization_upper_limit, quantization_levels)

# Backup data for inference test.
np.save("level_hypervector_array", level_hypervector_array)
np.save("id_hypervector_array", id_hypervector_array)
np.save("quantized_range", quantized_range)

class_hypervector           = np.array([None] * dimensionality)
associative_memory          = np.array([None] * dimensionality)
bind_hypermatrix            = np.empty((number_of_features, dimensionality), np.bool_)

# Traverse the entire class_vector array in search for letters.
for index in range(len(class_vector)):
    if alphabet[int(class_vector[index])] == 'a':
        # Auxiliary code to check the program hasn't hung.
        number_of_instances += 1
        print(f'Found {number_of_instances} instances of letter a', end = '\r')

        # Generate the class hypervector for the letter 'a'.
        feature_vector = feature_matrix[index]

        for feature_index in range(len(feature_vector)):

            feature                                 = feature_vector[feature_index]
            position_hypervector                    = id_hypervector_array[feature_index]
            level_hypervector                       = quantize_sample(feature, quantized_range, level_hypervector_array)

            bind_hypermatrix[feature_index]         = bind(position_hypervector, level_hypervector)

associative_memory = multibundle(bind_hypermatrix)

# Save vector to a file.
np.save("associative_memory", associative_memory)