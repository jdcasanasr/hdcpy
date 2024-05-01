from hdcpy import *

for _ in range(10):
    u = random_hypervector(10000, 'BSC')
    v = random_hypervector(10000, 'BSC')

    print(f'{hamming_distance(u, v)}')