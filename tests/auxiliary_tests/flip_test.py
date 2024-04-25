from hdcpy_v2 import *
from hdcpy_auxiliary import *

for _ in range(10):
    u       = random_hypervector(10000, 'BSC')
    flip_hv = flip(u, 2500, 'BSC')

    print(f'{hamming_distance(u, flip_hv):0.4f}')

for _ in range(10):
    u       = random_hypervector(10000, 'MAP')
    flip_hv = flip(u, 2500, 'MAP')

    print(f'{cosine_similarity(u, flip_hv):0.4f}')