from hdcpy_v2 import *

for _ in range(10):
    u           = random_hypervector(10000, 'BSC')
    permute_u   = permute(u, 100)

    print(f'{hamming_distance(u, permute_u)}')

for _ in range(10):
    u           = random_hypervector(10000, 'MAP')
    permute_u   = permute(u, 100)

    print(f'{cosine_similarity(u, permute_u)}')