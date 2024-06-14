from hdcpy                  import *
from hdcpy_classification   import *
from hdcpy_auxiliary        import *

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
id_item_memory              = get_id_hypermatrix(number_of_ids, number_of_dimensions, vsa)
level_item_memory           = get_level_hypermatrix(number_of_levels, number_of_dimensions, vsa)
encoded_training_dataset    = encode_dataset(training_features, level_item_memory, id_item_memory, vsa)
encoded_testing_dataset     = encode_dataset(testing_features, level_item_memory, id_item_memory, vsa)
associative_memory          = np.empty((number_of_classes, number_of_dimensions), np.int_)

# 4) Train the model.
associative_memory = train_analog(encoded_training_dataset, training_labels, number_of_classes, number_of_dimensions, vsa)

# 5) Retrain the model.
retrained_associative_memory = retrain_analog(associative_memory, encoded_training_dataset, training_labels, vsa)

# 6) Test the model.
accuracy = test_analog(encoded_testing_dataset, testing_labels, associative_memory, vsa)

print(f'Accuracy: {accuracy:0.2f}')