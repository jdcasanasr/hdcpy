from hdcpy import *

#1 Load data from ISOLET dataset.
training_data_path  = 'data/isolet1+2+3+4.data'
testing_data_path   = 'data/isolet5.data'

feature_matrix, class_vector = load_dataset(training_data_path)

#2 Train the model.
dimensionality              = 100
quantization_levels         = 10
quantization_lower_limit    = -1
quantization_upper_limit    = 1
seed_hypervector            = generate_hypervector(dimensionality)

level_hypervector_array     = generate_level_hypervectors(seed_hypervector, quantization_levels)
quantized_range             = quantize_range(quantization_lower_limit, quantization_upper_limit, quantization_levels)

associative_memory = []

# ToDo: Debug classification

# Iterate through each feature vector
for letter in range(1, 26):            #Iterate over each class (a, b, c, ...).
    class_hypervector   = []
    feature_vector      = []

    for class_index in range(len(class_vector)):  # Iterate over each element in 'class_vector'.
        class_element = int(class_vector[class_index])
        if (class_element == letter): # If current training data belongs to the same class...
            feature_vector = feature_matrix[class_index] # Fetch feature vector that belongs to the current letter.

            for feature in feature_vector:
                # Generate 'Level' and 'Position (ID)' hypervectors and bind them.
                position_hypervector    = generate_hypervector(dimensionality) # ID hypervector
                level_hypervector       = quantize_sample(feature, quantized_range, level_hypervector_array) # Level hypervector.
                bind_hypervector        = bind(position_hypervector, level_hypervector)

                # Form the class prototype hypervector by bundling all subsequent hypervectors
                # belonging to the same class.
                if (len(class_hypervector) == 0): # Check if 'class_hypervector' is empty.
                    #class_hypervector = copy_hypervector(bind_hypervector)
                    #class_hypervector = copy.deepcopy(bind_hypervector)
                    #class_hypervector = bind(position_hypervector, level_hypervector)
                    class_hypervector = bind_hypervector

                else:
                    #aux_hypervector     = copy_hypervector(class_hypervector)
                    #aux_hypervector     = copy.deepcopy(class_hypervector)
                    #class_hypervector   = bundle(aux_hypervector, bind_hypervector)
                    class_hypervector   = bundle(class_hypervector, bind_hypervector)

                    #if (any(class_hypervector)):
                    #    print("Something's Wrong!")

    associative_memory.append([class_hypervector, letter]) # In the end, this has to have 26 letter hypervectors.


print("Hello, Dummy!")