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

testing_data_path               = 'data/ISOLET/isolet5.data'
feature_matrix, class_vector    = load_dataset(testing_data_path)

level_hypermatrix               = np.load('level_hypermatrix.npy', allow_pickle = True)
position_hypermatrix            = np.load('position_hypermatrix.npy', allow_pickle = True)
associative_memory              = np.load('associative_memory.npy', allow_pickle = True)
quantization_range              = np.load('quantization_range.npy', allow_pickle = True)

number_of_data_elements         = len(class_vector)
number_of_classes               = 26
number_of_correct_predictions   = 0
number_of_class_instances       = 0

# Find all instances of letter a' in feature_matrix, and test classification.
for test_class in range(1, number_of_classes + 1):
    for class_index in range(len(class_vector)):
        if alphabet[int(class_vector[class_index])] == alphabet[test_class]:
            feature_vector = feature_matrix[class_index]
            number_of_class_instances += 1
            print(f'Instances: {number_of_class_instances} of {number_of_data_elements} | Correct Predictions: {number_of_correct_predictions} | Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%', end = '\r')

            # We all know it belongs to class 'a'!.
            query_hypervector = encode(feature_vector, quantization_range, level_hypermatrix, position_hypermatrix)

            # Classify the unknown hypervector.
            predicted_class = classify(associative_memory, query_hypervector)
            if (alphabet[predicted_class] == alphabet[test_class]):
                number_of_correct_predictions += 1

print(f'Instances: {number_of_class_instances} of {number_of_data_elements} | Correct Predictions: {number_of_correct_predictions} | Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%')