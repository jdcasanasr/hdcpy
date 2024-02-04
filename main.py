from hdcpy import *

#1 Load data from ISOLET dataset.
training_data_path  = 'data/isolet1+2+3+4.data'
testing_data_path   = 'data/isolet5.data'

feature_matrix, class_vector = load_dataset(training_data_path)

#2 Train the model.
dimensionality              = 10000
quantization_levels         = 50
quantization_lower_limit    = -1
quantization_upper_limit    = 1
seed_hypervector            = generate_hypervector(dimensionality)

level_hypervector_array     = generate_level_hypervectors(seed_hypervector, dimensionality, quantization_levels)
quantized_range             = quantize_range(quantization_lower_limit, quantization_upper_limit, quantization_levels)

associative_memory = []

# Iterate through each feature vector
for class_index in range(1, 26):            #Iterate over each class (a, b, c, ...).
    class_hypervector = []

    for index in range(len(class_vector)):  # Iterate over each element in 'class_vector'.
        if (class_vector[index] == class_index):
            # Use 'index' to access 'feature_matrix'
            feature_vector = feature_matrix[index]

            for feature in feature_vector:
                # Generate 'Level' and 'Position (ID)' hypervectors.
                position_hypervector    = generate_hypervector(dimensionality)
                level_hypervector       = quantize_sample(feature_vector[feature], quantized_range, level_hypervector_array)
                dummy_hypervector       = bind(position_hypervector, level_hypervector)

                if not class_hypervector: # Check if 'class_hypervector' is empty.
                    class_hypervector = dummy_hypervector
                else:
                    aux_hypervector     = copy_hypervector(class_hypervector)
                    class_hypervector   = bundle(aux_hypervector, dummy_hypervector)

    associative_memory.append([class_index, index])


print("Hello, Dummy!")