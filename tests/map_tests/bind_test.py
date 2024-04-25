from hdcpy import *

# We expect values close to 0 (dissimilar or quasi-orthogonal vectors).
for _ in range(10):
    u       = random_hypervector(10000, 'MAP')
    v       = random_hypervector(10000, 'MAP')
    bind_hv = bind(u, v, 'MAP')

    similarity_u = cosine_similarity(u, bind_hv)
    similarity_v = cosine_similarity(v, bind_hv)

    print(f'Similarities: {similarity_u:0.4f}, {similarity_v:0.4f}')