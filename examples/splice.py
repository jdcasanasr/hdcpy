from hdcpy                  import *
from hdcpy_classification   import *
from hdcpy_auxiliary        import *

# 0) Preparation.
equivalence_dictionary = {
    'N'     : 0,
    'EI'    : 1,
    'IE'    : 2
}

labels = [
    'A', 
    'G', 
    'T', 
    'C',
     
    'D', 
    'N',
    'S',
    'R'
]

# 1) Fetch dataset.
dataset_name    = 'har'
save_directory  = '../data'
test_proportion = 0.2

training_features, testing_features, training_labels, testing_labels = get_dataset(dataset_name, save_directory, test_proportion)

# 2) Define model parameters.
vsa                     = 'MAP'
number_of_dimensions    = 10000
number_of_classes       = 3
number_of_levels        = 10
number_of_ids           = np.shape(training_features)[1]

# 3) Model preparation.
id_item_memory          = get_id_hypermatrix(number_of_ids, number_of_dimensions, vsa)
level_item_memory       = get_level_hypermatrix(number_of_levels, number_of_dimensions, vsa)

number_of_hits          = 0
number_of_tests         = np.shape(testing_features)[0]

# 4) Train the model.
associative_memory = train_analog(training_features, training_labels,
                                  labels, equivalence_dictionary,
                                  id_item_memory, level_item_memory,
                                  vsa)

# 5) Retrain the model.
#retrained_associative_memory = retrain_analog(
#    associative_memory, training_features,
#    training_labels, level_item_memory,
#    id_item_memory, equivalence_dictionary,
#    vsa)

# 6) Test the model.
accuracy = test_analog(testing_features, testing_labels,
                       equivalence_dictionary, associative_memory,
                       id_item_memory, level_item_memory,
                       vsa)

print(f'Accuracy: {accuracy:0.2f}')