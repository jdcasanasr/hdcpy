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
associative_memory      = np.empty((number_of_classes, number_of_dimensions), np.int_)

row_item_memory         = get_id_hypermatrix(number_of_ids, number_of_dimensions, vsa)
column_item_memory      = np.empty(number_of_ids, np.int_)
max_pixel_value         = 256
pixel_level_memory      = get_id_hypermatrix(max_pixel_value, number_of_dimensions, vsa)

training_sample_size    = 500 # 627
inference_sample_size   = 100 # 103
number_of_samples       = 0
number_of_hits          = 0

hypermatrix = np.empty((training_sample_size, number_of_dimensions), np.int_)

# Generate column hypervectors by permuting existing row hypervectors.
for column_index in range(number_of_ids):
    column_item_memory = permute(row_item_memory[column_index], column_index)

# Train the model.
for label in range(number_of_classes):
    for digit, index in enumerate(training_features):
        if label == training_labels[index] and number_of_samples <= training_sample_size:
            # Encode digit.
            hypermatrix[index] = encode_image()
            number_of_samples += 1

        associative_memory[label] = multibundle(hypermatrix, vsa)

# Test model.
for index, actual_label in enumerate(testing_labels[0:inference_sample_size - 1]):
    distance_array = []
    query_hypervector = encode_image(testing_features[index], column_item_memory, row_item_memory, pixel_level_memory, vsa)

    for class_label, class_hypervector in enumerate(associative_memory):
        distance_array.append(hamming_distance(query_hypervector, class_hypervector))

    predicted_label = int(np.argmin(distance_array))

    if predicted_label == actual_label:
        number_of_hits += 1

print(f'Accuracy: {number_of_hits / inference_sample_size * 100}:0.2f')