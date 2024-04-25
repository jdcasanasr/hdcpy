from hdcpy_v2 import *

for _ in range(10):
    u = random_hypervector(10000, 'MAP')
    v = random_hypervector(10000, 'MAP')

    print(f'{cosine_similarity(u, v)}')