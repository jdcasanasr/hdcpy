from hdcpy import *

number_of_dimensions    = 10
#number_of_hypervectors  = 100

#hypervector_array       = np.array([None] * number_of_hypervectors)
#bundled_hypervector_1   = np.array([None] * number_of_dimensions)
#bundled_hypervector_2   = np.array([None] * number_of_dimensions)

u = generate_hypervector(number_of_dimensions)
v = generate_hypervector(number_of_dimensions)
#hypervector_array[0] = u
#hypervector_array[1] = v

#for index in range(number_of_hypervectors):
#    hypervector_array[index] = generate_hypervector(number_of_dimensions)

bundled_hypervector_1 = bundle_3(u, v)
#bundled_hypervector_2 = bundle_2(hypervector_array)

delta_1 = hamming_distance(u, bundled_hypervector_1)
delta_2 = hamming_distance(v, bundled_hypervector_1)
#delta_3 = hamming_distance(u, bundled_hypervector_2)
#delta_4 = hamming_distance(v, bundled_hypervector_2)

print(f'Hamming Distance 1: {delta_1:0.4f}')
print(f'Hamming Distance 2: {delta_2:0.4f}')
#print(f'Hamming Distance 3: {delta_3:0.4f}')
#print(f'Hamming Distance 4: {delta_4:0.4f}')

#for index in range(number_of_hypervectors - 1):
#    if None in bundled_hypervector:
#        bundled_hypervector = bundle_2(hypervector_array[index], hypervector_array[index + 1])
#
#    else:
#        bundled_hypervector = bundle_2(bundled_hypervector, hypervector_array[index + 1])

#for index in range(number_of_hypervectors):
#    delta = hamming_distance(hypervector_array[index], bundled_hypervector_2)
#    print(f'Hamming distance: {delta:0.4f}')

#for _ in range(number_of_tests):
#    u                   = generate_hypervector(dimensionality)
#    v                   = generate_hypervector(dimensionality)
#    bound_hypervector   = bind(u, v)
#    bundled_hypervector = bundle(u, v)
#    distance_u_bound    = hamming_distance(u, bound_hypervector)
#    distance_v_bound    = hamming_distance(v, bound_hypervector)
#    distance_u_bundled  = hamming_distance(u, bundled_hypervector)
#    distance_v_bundled  = hamming_distance(v, bundled_hypervector)
#
#    print('Bound distance (u): %0.4f | Bundled distance (u): %0.4f.' % (distance_u_bound, distance_u_bundled))
#    print('Bound distance (v): %0.4f | Bundled distance (v): %0.4f.' % (distance_v_bound, distance_v_bundled))