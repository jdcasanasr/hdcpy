import numpy as np

from hdcpy import random_hypervector

number_of_tests = 10000
dimensionality  = 10000

for _ in range(number_of_tests):
    u = random_hypervector(dimensionality, 'BSC')

    if np.any((u != 0) & (u != 1)):
        print('Test for BSC FAILED')

print('Test for BSC PASSED')

for _ in range(number_of_tests):
    u = random_hypervector(dimensionality, 'MAP')

    if np.any((u != -1) & (u != 1)):
        print('Test for MAP FAILED')

print('Test for MAP PASSED')