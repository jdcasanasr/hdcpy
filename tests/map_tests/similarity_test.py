from hdcpy import *

# We expect values close to 0 (dissimilar or quasi-orthogonal vectors).
for _ in range(10):
    u = random_hypervector(10000, 'MAP')
    v = random_hypervector(10000, 'MAP')

    similarity = cosine_similarity(u, v)

    print(f'Similarity: {similarity:0.4f}')