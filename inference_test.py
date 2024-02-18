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
level_hypermatrix           = np.load('level_hypermatrix.npy', allow_pickle = True)
position_hypermatrix        = np.load('position_hypermatrix.npy', allow_pickle = True)
associative_memory          = np.load('associative_memory.npy', allow_pickle = True)
quantization_range          = np.load('quantization_range.npy', allow_pickle = True)
a_instances                 = 0
successes                   = 0
test_class                  = 2

# Find all instances of letter a' in feature_matrix, and test classification.
for class_index in range(len(class_vector)):
    if alphabet[int(class_vector[class_index])] == alphabet[test_class]:
        feature_vector = feature_matrix[class_index]
        a_instances += 1
        print(f'Found {a_instances} instances of class {alphabet[test_class]}', end = '\r')

        # We all know it belongs to class 'a'!.
        query_hypervector = encode(feature_vector, quantization_range, level_hypermatrix, position_hypermatrix)

        # Classify the unknown hypervector.
        predicted_class, maximum_similarity = classify(associative_memory, query_hypervector)
        if (alphabet[predicted_class] == 'b'):
            successes += 1

print(f'Instances: {a_instances} | Correct Predictions: {successes} | Accuracy: {(successes / a_instances) * 100}%')