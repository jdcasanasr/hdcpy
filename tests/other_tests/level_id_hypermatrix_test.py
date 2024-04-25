from hdcpy_v2               import *
from hdcpy_auxiliary        import *
from hdcpy_classification   import *

print('BSC Test')
level_item_memory   = get_level_hypermatrix(10, 10000, 'BSC')
id_item_memory      = get_id_hypermatrix(10, 10000, 'BSC')

for index in range(1, 10):
    print(f'{hamming_distance(level_item_memory[0], level_item_memory[index])}')
    
for index in range(1, 10):
    print(f'{hamming_distance(id_item_memory[0], id_item_memory[index])}')

print('MAP Test')
level_item_memory   = get_level_hypermatrix(10, 10000, 'MAP')
id_item_memory      = get_id_hypermatrix(10, 10000, 'MAP')

for index in range(1, 10):
    print(f'{cosine_similarity(level_item_memory[0], level_item_memory[index])}')
    
for index in range(1, 10):
    print(f'{cosine_similarity(id_item_memory[0], id_item_memory[index])}')