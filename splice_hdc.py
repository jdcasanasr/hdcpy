import sys
import time

from hdcpy      import *
from hdc_dna    import *


vsa             = str(sys.argv[1])
dimensionality  = int(sys.argv[2])
test_proportion = 0.2

X_train, X_test, y_train, y_test = get_dataset('splice', '.', test_proportion)

feature_dictionary = {
    'A' : random_hypervector(dimensionality, vsa), 
    'G' : random_hypervector(dimensionality, vsa), 
    'T' : random_hypervector(dimensionality, vsa), 
    'C' : random_hypervector(dimensionality, vsa),

    'D' : random_hypervector(dimensionality, vsa), 
    'N' : random_hypervector(dimensionality, vsa),
    'S' : random_hypervector(dimensionality, vsa),
    'R' : random_hypervector(dimensionality, vsa)
}

sequence_length     = np.shape(X_train)[1]
id_item_memory      = get_id_hypermatrix(sequence_length, dimensionality, vsa)
X_train_encoded     = np.empty((np.shape(X_train)[0], dimensionality), np.int_)
X_test_encoded      = np.empty((np.shape(X_test)[0], dimensionality), np.int_)

start_time = time.time()

# Encode the training dataset.
for index, dna_sequence in enumerate(X_train):
    X_train_encoded[index] = encode_dna_sequence(dna_sequence, feature_dictionary, id_item_memory, dimensionality, vsa)

# Encode the testing dataset.
for index, dna_sequence in enumerate(X_test):
    X_test_encoded[index] = encode_dna_sequence(dna_sequence, feature_dictionary, id_item_memory, dimensionality, vsa)

associative_memory  = train_dna(X_train_encoded, y_train, dimensionality, vsa)
accuracy            = test_dna(X_test_encoded, y_test, associative_memory, feature_dictionary, dimensionality, vsa)

end_time        = time.time()
execution_time  = end_time - start_time

print(f'{execution_time:.3f},{vsa},{dimensionality},{(accuracy * 100):.2f}')