from hdcpy import *

def load_feature_data(file_path):
    with open(file_path, 'r') as f:
        data = [line.strip().split() for line in f]
    return np.array(data, dtype = np.double)

def load_class_data(file_path):
    with open(file_path, 'r') as f:
        data = [line.strip().split()[0] for line in f]
    return np.array(data, dtype = np.uint)

testing_feature_data            = load_feature_data('data/UCIHAR/X_test.txt')
testing_class_data              = load_class_data('data/UCIHAR/y_test.txt')

level_hypermatrix               = np.load('level_hypermatrix.npy', allow_pickle = True)
position_hypermatrix            = np.load('position_hypermatrix.npy', allow_pickle = True)
associative_memory              = np.load('associative_memory.npy', allow_pickle = True)
quantization_range              = np.load('quantization_range.npy', allow_pickle = True)

number_of_data_elements         = np.shape(testing_feature_data)[0]
number_of_correct_predictions   = 0
number_of_class_instances       = 0

for index in range(number_of_data_elements):
    number_of_class_instances   += 1
    feature_vector              = testing_feature_data[index][:]
    actual_class                = testing_class_data[index]
    query_hypervector           = encode(feature_vector, quantization_range, level_hypermatrix, position_hypermatrix)
    predicted_class             = classify(associative_memory, query_hypervector)

    if predicted_class == actual_class:
        number_of_correct_predictions += 1

    print(f'Instances: {number_of_class_instances} of {number_of_data_elements} | Correct Predictions: {number_of_correct_predictions} | Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%', end = '\r')

print(f'Instances: {number_of_class_instances} of {number_of_data_elements} | Correct Predictions: {number_of_correct_predictions} | Accuracy: {((number_of_correct_predictions / number_of_data_elements) * 100):0.2f}%')