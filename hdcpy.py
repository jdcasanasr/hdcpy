import numpy as np
#import matplotlib.pyplot as plt

# Generate "binary", random hypervector.
# ToDo: Generalize for other VSA
def generate_hypervector(dimensionality):
    return np.random.randint(0, 2, dimensionality, np.byte)

# Compute Hamming distance between hypervectors,
# generated via the 'generate_hypervector' function.
def hamming_distance(u, v, dimensionality):
    return np.count_nonzero(np.logical_xor(u, v)) / dimensionality

# Basic HDC Operations
def bind(u, v):
    return np.logical_xor(u, v)

def bundle(u, v):
    return np.where((u + v) >= 2, 1, 0)

def permute(u, amount):
    return np.roll(u, amount)

# Take two hypervectors at random and compute
# their Hamming distance.
def experiment(dimensionality):
    u = generate_hypervector(dimensionality)
    v = generate_hypervector(dimensionality)

    return hamming_distance(u, v, dimensionality)

def binding_experiment(dimensionality):
    u = generate_hypervector(dimensionality)
    v = generate_hypervector(dimensionality)

    bound_hypervector = bind(u, v)

    delta_u = hamming_distance(u, bound_hypervector, dimensionality)
    delta_v = hamming_distance(v, bound_hypervector, dimensionality)

    return delta_u, delta_v

def bundling_experiment(dimensionality):
    u = generate_hypervector(dimensionality)
    v = generate_hypervector(dimensionality)

    bundled_hypervector = bundle(u, v)

    delta_u = hamming_distance(u, bundled_hypervector, dimensionality)
    delta_v = hamming_distance(v, bundled_hypervector, dimensionality)

    return delta_u, delta_v

# Find the class hypervector matching the
# minimum distance with a query vector.
def find_class(associative_memory, query_hypervector, dimensionality):
    minimum_distance = 1.0
    
    for class_hypervector in associative_memory:
        delta = hamming_distance(query_hypervector, class_hypervector, dimensionality)

        if (delta < minimum_distance):
            minimum_distance = delta

    return minimum_distance

for _ in range(10):
    #print(binding_experiment(10000))
    #print(bundling_experiment(10000))
    print('Distance (Binding):  (%0.4f, %0.4f)' %(binding_experiment(10000)))
    print('Distance (Bundling): (%0.4f, %0.4f)' %(bundling_experiment(10000)))
    

#dimensionality      = 10000
#number_classes      = 26
#associative_memory  = []
#query_hypervector   = generate_hypervector(dimensionality)

# Populate associative memory with dummy hypervectors.
#for _ in range(number_classes):
#    associative_memory.append(generate_hypervector(dimensionality))

#print("Minimum Distance: %0.5f" % find_class(associative_memory, query_hypervector, dimensionality))


#upper_limit     = 1000
#dimensionality  = 10000
#distance_array  = []

#for i in range(1, upper_limit + 1):
#    experiments = i
#    successes   = 0
#
#    for j in range(1, i + 1):
#        distance = experiment(dimensionality)
#        successes += 1 if distance >= 0.5 else 0
#
#    probability = successes / experiments
#
#    print("Experiments: %d | Successes: %d | Probability: %0.2f" % (experiments, successes, probability))
