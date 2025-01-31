from hdcpy                  import *
from hdcpy_classification   import *
from hdcpy_auxiliary        import *

# 0) Preparation.
equivalence_dictionary = {
    '1'     : 0,
    '2'     : 1,
    '3'     : 2,
    '4'     : 3,
    '5'     : 4,
    '6'     : 5,
    '7'     : 6, 
    '8'     : 7,
    '9'     : 8,
    '10'    : 9,
    '11'    : 10,
    '12'    : 11,
    '13'    : 12,
    '14'    : 13,
    '15'    : 14,
    '16'    : 15,
    '17'    : 16,
    '18'    : 17,
    '19'    : 18,
    '20'    : 19,
    '21'    : 20,
    '22'    : 21,
    '23'    : 22,
    '24'    : 23,
    '25'    : 24,
    '26'    : 25
}

labels = [
    '1', 
    '2', 
    '3', 
    '4', 
    '5', 
    '6', 
    '7', 
    '8', 
    '9', 
    '10',
    '11',
    '12',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20',
    '21',
    '22',
    '23',
    '24',
    '25',
    '26'
]

# 1) Fetch dataset.
dataset_name    = 'isolet'
save_directory  = '../data'
test_proportion = 0.2

training_features, testing_features, training_labels, testing_labels = get_dataset(dataset_name, save_directory, test_proportion)

# 2) Define model parameters.
vsa                     = 'BSC'
number_of_dimensions    = 10000
number_of_classes       = 26
number_of_levels        = 10
number_of_ids           = np.shape(training_features)[1]

# 3) Model preparation.
id_item_memory          = get_id_hypermatrix(number_of_ids, number_of_dimensions, vsa)
level_item_memory       = get_level_hypermatrix(number_of_levels, number_of_dimensions, vsa)
associative_memory      = np.empty((number_of_classes, number_of_dimensions), np.int_)

number_of_hits  = 0
number_of_tests = np.shape(testing_features)[0]

# 4) Train the model.
for current_label in labels:
    prototype_hypermatrix = np.empty(number_of_dimensions, dtype = np.int_)

    for index in range(np.shape(training_labels)[0]):
        if training_labels[index] == current_label:
            prototype_hypermatrix = np.vstack((prototype_hypermatrix, encode_analog(training_features[index], level_item_memory, id_item_memory, vsa)))

    associative_memory[equivalence_dictionary[current_label]] = multibundle(prototype_hypermatrix[1:][:], vsa)

# 5) Retrain the model.
retrained_associative_memory = retrain_analog(
    associative_memory, training_features,
    training_labels, level_item_memory,
    id_item_memory, equivalence_dictionary,
    vsa)

# 6) Test the model.
for index in range(np.shape(testing_features)[0]):
    query_hypervector   = encode_analog(testing_features[index][:], level_item_memory, id_item_memory, vsa)
    predicted_class     = classify(query_hypervector, retrained_associative_memory, vsa)
    label               = testing_labels[index]
    actual_class        = equivalence_dictionary[testing_labels[index]]

    if predicted_class == actual_class:
        number_of_hits += 1

print(f'Accuracy: {(number_of_hits / number_of_tests * 100):0.2f}')