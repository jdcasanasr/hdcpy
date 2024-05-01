from hdcpy                  import *
from hdcpy_auxiliary        import *
from hdcpy_classification   import *

dimensionality  = 10
number_of_tests = 10

for _ in range(number_of_tests):
    u           = np.random.randint(-10, 11, size = dimensionality)
    binarized_u = binarize(u, 'MAP')


