from hdcpy                  import *
from hdcpy_classification   import *
from hdcpy_auxiliary        import *

#1) Fetch dataset.
dataset_name    = 'mnist_784'
save_directory  = '../data'
test_proportion = 0.2

training_features, testing_features, training_labels, testing_labels = get_dataset(dataset_name, save_directory, test_proportion)

# 2) Define model parameters.
vsa                     = 'BSC'
number_of_dimensions    = 10000
number_of_classes       = 10
number_of_bases         = 0
number_of_ids           = np.shape(training_features)[1]