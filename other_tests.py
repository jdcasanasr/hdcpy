from hdcpy import *
from torchhd import *

number_of_hypervectors  = 20
number_of_dimensions    = 10000
vsa                     = 'BSC'

torchhd_uv_tensor   = random(number_of_hypervectors, number_of_dimensions, vsa)
#hypervector_array   = np.empty((number_of_hypervectors, number_of_dimensions), np.bool_)
delta_array         = []

#for tensor_index in range(len(torchhd_uv_tensor)):
#    for index in range(len(torchhd_uv_tensor[tensor_index])):
#        hypervector_array[tensor_index][index] = np.bool_(torchhd_uv_tensor[tensor_index][index].item())

#hdcpy_bundle_hypervector    = bundle_2(hypervector_array)
torchhd_bundle_hypervector  = multibundle(torchhd_uv_tensor)

#hdcpy_u_delta = hamming_distance(hdcpy_u, hdcpy_bundle_hypervector)
#hdcpy_v_delta = hamming_distance(hdcpy_v, hdcpy_bundle_hypervector)
#torchhd_u_delta = (number_of_dimensions - int(hamming_similarity(torchhd_uv_tensor[0], torchhd_bundle_hypervector).item())) / number_of_dimensions
#torchhd_v_delta = (number_of_dimensions - int(hamming_similarity(torchhd_uv_tensor[1], torchhd_bundle_hypervector).item())) / number_of_dimensions

for index in range(number_of_hypervectors):
    delta_array.append((number_of_dimensions - int(hamming_similarity(torchhd_uv_tensor[index], torchhd_bundle_hypervector).item())) / number_of_dimensions)

for index in range(number_of_hypervectors):
    print(f'Hamming(hypervector, bundle) = {delta_array[index]:0.4f}')
