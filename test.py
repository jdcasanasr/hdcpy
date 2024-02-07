from hdcpy import *

#1 Load data from ISOLET dataset.
training_data_path  = 'data/isolet1+2+3+4.data'
testing_data_path   = 'data/isolet5.data'

feature_matrix, class_vector = load_dataset(training_data_path)

#2 Train the model.
dimensionality              = 10000
quantization_levels         = 10
quantization_lower_limit    = -1
quantization_upper_limit    = 1
seed_hypervector            = generate_hypervector(dimensionality)

level_hypervector_array     = generate_level_hypervectors(seed_hypervector, quantization_levels)
quantized_range             = quantize_range(quantization_lower_limit, quantization_upper_limit, quantization_levels)

feature_vector              = feature_matrix[0]
class_hypervector           = np.array([None] * dimensionality)
iteration                   = 0

for feature in feature_vector:
    position_hypervector    = generate_hypervector(dimensionality)                               # ID hypervector
    level_hypervector       = quantize_sample(feature, quantized_range, level_hypervector_array) # Level

    bind_hypervector        = bind(position_hypervector, level_hypervector)

    if (None in class_hypervector):
        class_hypervector = bind_hypervector

    else:
        class_hypervector = bundle(class_hypervector, bind_hypervector)