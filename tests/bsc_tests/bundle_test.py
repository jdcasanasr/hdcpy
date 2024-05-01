from hdcpy import *

for _ in range(10):
    u           = random_hypervector(10000, 'BSC')
    v           = random_hypervector(10000, 'BSC')
    bundle_hv   = bundle(u, v, 'BSC')

    print(f'{hamming_distance(u, bundle_hv):0.4f}, {hamming_distance(v, bundle_hv):0.4f}')

for _ in range(10):
    u           = random_hypervector(10000, 'MAP')
    v           = random_hypervector(10000, 'MAP')
    bundle_hv   = bundle(u, v, 'MAP')

    print(f'{cosine_similarity(u, bundle_hv):0.4f}, {cosine_similarity(v, bundle_hv):0.4f}')