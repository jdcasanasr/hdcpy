from hdcpy_v2 import *

for _ in range(10):
    u       = random_hypervector(10000, 'BSC')
    v       = random_hypervector(10000, 'BSC')
    bind_hv = bind(u, v, 'BSC')

    print(f'{hamming_distance(u, bind_hv)}, {hamming_distance(v, bind_hv)}')

for _ in range(10):
    u       = random_hypervector(10000, 'MAP')
    v       = random_hypervector(10000, 'MAP')
    bind_hv = bind(u, v, 'MAP')

    print(f'{cosine_similarity(u, bind_hv)}, {cosine_similarity(v, bind_hv)}')