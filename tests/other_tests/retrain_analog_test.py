from hdcpy                  import random_hypervector
from hdcpy_auxiliary        import binarize
from hdcpy_classification   import *

import numpy as np

dimensionality      = 10000
number_of_classes   = 5
number_of_features  = 100
number_of_levels    = 10
vsa                 = 'BSC'

associative_memory  = np.empty((number_of_classes, dimensionality), np.int_)
id_item_memory      = get_id_hypermatrix(number_of_features, dimensionality, vsa)
level_item_memory   = get_level_hypermatrix(number_of_levels, dimensionality, vsa)

# Populate dummy memories