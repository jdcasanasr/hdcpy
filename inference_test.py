from hdcpy import *

alphabet = {

    1   : 'a',
    2   : 'b',
    3   : 'c',
    4   : 'd',
    5   : 'e',
    6   : 'f',
    7   : 'g',
    8   : 'h',
    9   : 'i',
    10  : 'j',
    11  : 'k',
    12  : 'l',
    13  : 'm',
    14  : 'n',
    15  : 'o',
    16  : 'p',
    17  : 'q',
    18  : 'r',
    19  : 's',
    20  : 't',
    21  : 'u',
    22  : 'v',
    23  : 'w',
    24  : 'x',
    25  : 'y',
    26  : 'z'
}

# Goal: To find all vectors for letter 'a' and see if the model works.
testing_data_path               = 'data/isolet5.data'
feature_matrix, class_vector    = load_dataset(testing_data_path)

# Model requirements.
dimensionality              = 10000
level_hypervector_array     = np.load('level_hypervector_array.npy', allow_pickle = True)
id_hypervector_array        = np.load('id_hypervector_array.npy', allow_pickle = True)
associative_memory          = np.load('associative_memory.npy', allow_pickle = True)
quantized_range             = np.load('quantized_range.npy', allow_pickle = True)
a_instances                 = 0

# Find all instances of letter a' in feature_matrix, and test classification.
for index in range(len(class_vector)):
    if alphabet[int(class_vector[index])] == 'a':
        a_instances += 1
        successes = 0

        # We all know it belongs to class 'a'!.
        query_hypervector = transform(feature_matrix[index], dimensionality, quantized_range, level_hypervector_array, id_hypervector_array)

        # Classify the unknown hypervector.
        predicted_class, minimum_distance = classify(associative_memory, query_hypervector)

        if (alphabet[predicted_class] == 'a'):
            successes += 1

print("Instances of letter 'a': %d." % a_instances)
print("Correct predictions: %d." % successes)