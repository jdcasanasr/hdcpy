from hdcpy import *

dataset                 = 'isolet'
vsa                     = 'BSC'
number_of_levels        = 10
number_of_dimensions    = 10000

save_directory          = './data'
test_proportion         = 0.2

X_train, X_test, y_train, y_test = get_dataset(dataset, save_directory, test_proportion)

number_of_classes       = get_number_of_classes(y_train)
number_of_ids           = np.shape(X_train)[1]

id_item_memory          = get_id_hypermatrix(number_of_ids, number_of_dimensions, vsa)
level_item_memory       = get_level_hypermatrix(number_of_levels, number_of_dimensions, vsa)

encoded_X_train         = encode_dataset(X_train, number_of_dimensions, level_item_memory, id_item_memory, vsa)
encoded_X_test          = encode_dataset(X_test, number_of_dimensions, level_item_memory, id_item_memory, vsa)

associative_memory      = train_analog(encoded_X_train, y_train, number_of_classes, number_of_dimensions, vsa)

accuracy                = test_analog(encoded_X_test, y_test, associative_memory, vsa)

print(f'Accuracy: {accuracy:0.2f}')