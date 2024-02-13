from hdcpy import *

number_of_dimensions    = 10
#number_of_tests         = 10
number_of_hypervectors  = 2

# 0.5000 - Bind
# 0.2500 - Bundle

hypervector_array   = np.array([None] * number_of_hypervectors)
#bundled_hypervector = np.array([None] * number_of_dimensions)

for index in range(number_of_hypervectors):
    hypervector_array[index] = generate_hypervector(number_of_dimensions)

bundled_hypervector = bundle(hypervector_array)

#for index in range(number_of_hypervectors - 1):
#    if None in bundled_hypervector:
#        bundled_hypervector = bundle(hypervector_array[index], hypervector_array[index + 1])
#
#    else:
#        bundled_hypervector = bundle(bundled_hypervector, hypervector_array[index + 1])

for index in range(number_of_hypervectors):
    delta = hamming_distance(hypervector_array[index], bundled_hypervector)
    print('hamming_distance(hypervector_array[%d], bundled_hypervector) = %0.4f' % (index, delta))

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