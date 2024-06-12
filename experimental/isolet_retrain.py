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

# 4) Train the model.
associative_memory = train_analog(training_features, training_labels, number_of_classes, id_item_memory, level_item_memory, vsa)
# 5) Retrain the model.
retrained_associative_memory = retrain_analog(associative_memory, training_features, training_labels, level_item_memory, id_item_memory, vsa)

# 6) Test the model.
accuracy = test_analog(testing_features, testing_labels, retrained_associative_memory,
                       id_item_memory, level_item_memory,
                       vsa)

#accuracy = test_analog(testing_features, testing_labels,
#                       equivalence_dictionary, associative_memory,
#                       id_item_memory, level_item_memory,
#                       vsa)

print(f'Accuracy: {accuracy:0.2f}')