from hdcpy import *
import numpy as np

dimensionality      = 50
levels              = 13
seed_hypervector = generate_hypervector(dimensionality)

upper_limit     = 1
lower_limit     = -1
number_levels   = 12
sample          = 0.34

# Assume upper_limit = 1 and lower_limit = -1 for VoiceHD.
def quantize(sample, upper_limit, lower_limit, number_levels):
    quantized_range = np.linspace(lower_limit, upper_limit, number_levels)
    quantized_index = np.digitize(sample, quantized_range, True)

    return quantized_index

level_array     = generate_level_hypervectors(seed_hypervector, dimensionality, levels)
quantized_index = quantize(sample, upper_limit, lower_limit, number_levels)

print(level_array[quantized_index])