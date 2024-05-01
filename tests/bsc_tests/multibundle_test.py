from hdcpy import *

print('BSC, Even Case')
for _ in range(10):
    u           = random_hypervector(10000, 'BSC')
    v           = random_hypervector(10000, 'BSC')
    bundle_hv   = multibundle(np.vstack((u, v)), 'BSC')

    print(f'{hamming_distance(u, bundle_hv):0.4f}, {hamming_distance(v, bundle_hv):0.4f}')

print('BSC, Odd Case')
for _ in range(10):
    u           = random_hypervector(10000, 'BSC')
    v           = random_hypervector(10000, 'BSC')
    w           = random_hypervector(10000, 'BSC')

    bundle_hv   = multibundle(np.vstack((u, v, w)), 'BSC')

    print(f'{hamming_distance(u, bundle_hv):0.4f}, {hamming_distance(v, bundle_hv):0.4f}, {hamming_distance(w, bundle_hv):0.4f}')

print('MAP, Even Case')
for _ in range(10):
    u           = random_hypervector(10000, 'MAP')
    v           = random_hypervector(10000, 'MAP')
    bundle_hv   = multibundle(np.vstack((u, v)), 'MAP')

    print(f'{cosine_similarity(u, bundle_hv):0.4f}, {cosine_similarity(v, bundle_hv):0.4f}')

print('MAP, Odd Case')
for _ in range(10):
    u           = random_hypervector(10000, 'MAP')
    v           = random_hypervector(10000, 'MAP')
    w           = random_hypervector(10000, 'MAP')
    bundle_hv   = multibundle(np.vstack((u, v, w)), 'MAP')

    print(f'{cosine_similarity(u, bundle_hv):0.4f}, {cosine_similarity(v, bundle_hv):0.4f}, {cosine_similarity(w, bundle_hv):0.4f}')