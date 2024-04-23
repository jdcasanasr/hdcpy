import numpy as np
from hdcpy import *

number_of_hypervectors  = number_of_rows    = 2
number_of_dimensions    = number_of_columns = 10000

random_hypermatrix      = np.random.choice([-1, 1], (number_of_rows, number_of_columns), p = [0.5, 0.5])
bundle_hypervector      = multibundle(random_hypermatrix, 'MAP')

for row_index in range(number_of_rows):
    similarity = cosine_similarity(bundle_hypervector, random_hypermatrix[row_index][:])

    print(f'Similarity: {similarity:0.4f}')