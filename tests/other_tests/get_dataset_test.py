import numpy as np
from hdcpy_auxiliary import *

dataset         = 'har'
save_directory  = '/home/jdcasanasr/Development/hdcpy/data'
test_proportion = 0.2

X_train, X_test, y_train, y_test = get_dataset(dataset, save_directory, test_proportion)

pass