from hdcpy import *

# We expect values close to 0 (dissimilar or quasi-orthogonal vectors).
for _ in range(10):
    u           = random_hypervector(10000, 'MAP')
    v           = random_hypervector(10000, 'MAP')
    bundle_hv   = bundle(u, v, 'MAP')

    similarity_u = cosine_similarity(u, bundle_hv)
    similarity_v = cosine_similarity(v, bundle_hv)

    print(f'Similarities: {similarity_u:0.4f}, {similarity_v:0.4f}')