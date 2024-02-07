import numpy as np

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA's.
def generate_hypervector(dimensionality):
    return np.random.choice([True, False], size = dimensionality, p = [0.5, 0.5])

# Basic HDC operations.
def bind(u, v):
    return np.logical_xor(u, v)

dim = 10

u = generate_hypervector(dim)
v = generate_hypervector(dim)

print(u)
print(v)
print(bind(u, v))