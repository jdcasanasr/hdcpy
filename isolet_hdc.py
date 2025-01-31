import sys
import time

from hdcpy import *

dataset                 = "isolet"
vsa                     = "BSC"
number_of_levels        = 10
number_of_dimensions    = 10000

save_directory          = '/home/jdcasanasr/Development/hdcpy/data'
test_proportion         = 0.2

X_train, X_test, y_train, y_test = get_dataset(dataset, save_directory, test_proportion)

start_time = time.time()

number_of_classes       = get_number_of_classes(y_train)

if dataset == "splice":
    number_of_symbols   = np.shape(X_train)[1]
    symbol_item_memory  = get_id_hypermatrix(number_of_symbols, number_of_dimensions, vsa)
    
else:
    number_of_ids           = np.shape(X_train)[1]

    id_item_memory          = get_id_hypermatrix(number_of_ids, number_of_dimensions, vsa)
    level_item_memory       = get_level_hypermatrix(number_of_levels, number_of_dimensions, vsa)

    encoded_X_train         = encode_dataset(X_train, number_of_dimensions, level_item_memory, id_item_memory, vsa)
    encoded_X_test          = encode_dataset(X_test, number_of_dimensions, level_item_memory, id_item_memory, vsa)

    associative_memory      = train_analog(encoded_X_train, y_train, number_of_classes, number_of_dimensions, vsa)

    accuracy                = test_analog(encoded_X_test, y_test, associative_memory, vsa)

end_time = time.time()

execution_time = end_time - start_time

# Dump contents.
save_array_to_csv (id_item_memory, "position_hypermatrix.csv")
save_array_to_csv (level_item_memory, "level_hypermatrix.csv")
save_array_to_csv (associative_memory, "associative_hypermatrix.csv")

print(f'{execution_time:0.3f},{dataset},{vsa},{number_of_levels},{number_of_dimensions},{accuracy:0.2f}')
