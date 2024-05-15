from hdcpy                  import *
from hdcpy_classification   import *
from hdcpy_auxiliary        import *

from sklearn.preprocessing import MinMaxScaler

# 0) Preparation.
label_array = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

label_dictionary = {
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
}

# 1) Fetch dataset.
dataset_name    = 'cardiotocography'
save_directory  = '../data'
test_proportion = 0.2

training_features, testing_features, y_train, y_test = get_dataset(dataset_name, save_directory, test_proportion)

# Normalize features between -1.0 and 1.0
scaler      = MinMaxScaler(feature_range = (-1.0, 1.0))
X_train     = scaler.fit_transform(training_features.astype(float))
X_test      = scaler.fit_transform(testing_features.astype(float))

# 2) Define model parameters.
vsa                     = 'BSC'
number_of_dimensions    = 10000
number_of_classes       = 10
number_of_levels        = 10
number_of_ids           = np.shape(X_train)[1]

# 3) Model preparation.
id_item_memory          = get_id_hypermatrix(number_of_ids, number_of_dimensions, vsa)
level_item_memory       = get_level_hypermatrix(number_of_levels, number_of_dimensions, vsa)

number_of_hits          = 0
number_of_tests         = np.shape(X_test)[0]

# 4) Train the model.
associative_memory = train_analog(X_train, y_train,
                                  label_array, label_dictionary,
                                  id_item_memory, level_item_memory,
                                  vsa)

# 5) Test the model.
accuracy = test_analog(X_test, y_test,
                       label_dictionary, associative_memory,
                       id_item_memory, level_item_memory,
                       vsa)

print(f'Accuracy: {accuracy:0.2f}')