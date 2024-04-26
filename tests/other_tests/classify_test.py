from hdcpy_v2               import *
from hdcpy_classification   import *
from hdcpy_auxiliary        import *

print('BSC Case')
dimensionality      = 10000
number_of_classes   = 10
vsa                 = 'BSC'
associative_memory  = np.uint(np.ceil(np.random.randint(0, 2, size = (number_of_classes, dimensionality))))
query_hypervector   = random_hypervector(dimensionality, vsa)

print(f'Predicted class: {classify(query_hypervector, associative_memory, vsa)}')

print('MAP Case')
dimensionality      = 10000
number_of_classes   = 10
vsa                 = 'MAP'
associative_memory  = np.int_(np.ceil(np.random.choice([-1, 1], size = (number_of_classes, dimensionality))))
query_hypervector   = random_hypervector(dimensionality, vsa)

print(f'Predicted class: {classify(query_hypervector, associative_memory, vsa)}')