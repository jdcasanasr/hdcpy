import numpy as np

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA's.
def generate_hypervector(dimensionality):
    return np.random.choice([True, False], size = dimensionality, p = [0.5, 0.5])

# Negate random positions in a source hypervector. Return a different hypervector.
def flip (u, number_positions):
    flip_u = np.array([None] * u.size)
    flip_positions = np.random.choice(u.size, size = number_positions, replace = False)

    for index in range(u.size):
        if (index in flip_positions):
            flip_u[index] = np.logical_not(u[index])
        
        else:
            flip_u[index] = u[index]

    return flip_u

dimensionality = 10

u = generate_hypervector(dimensionality)
m = 5

print(type(u))
print(type(flip(u, m)))
print(type(u))