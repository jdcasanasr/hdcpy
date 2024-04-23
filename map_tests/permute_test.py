import numpy as np
from hdcpy import *

for _ in range(10):
    vsa         = 'BSC'
    u           = random_hypervector(10000, vsa)
    permute_u   = permute(u, 1)

    if vsa == 'BSC':
        print(f'Distance: {hamming_distance(u, permute_u)}')
    
    else:
        print(f'Similarity: {cosine_similarity(u, permute_u)}')