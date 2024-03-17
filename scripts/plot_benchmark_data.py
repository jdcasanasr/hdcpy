import sys
import pandas as pd
import numpy as np

#log_file = str(sys.argv[1])
log_file = '/home/jdcasanasr/Development/hdcpy/logs/wall-robot-navigation.csv'

log_dataframe       = pd.read_csv(log_file, header = None)

dimensions          = np.array(log_dataframe.iloc[:, 0].values)
quantization_levels = np.array(log_dataframe.iloc[:, 1].values)
accuracy            = np.array(log_dataframe.iloc[:, 2].values)
time_per_dataset    = np.array(log_dataframe.iloc[:, 3].values)
time_per_query      = np.array(log_dataframe.iloc[:, 4].values)


for number_of_dimensions in range(1000, 10000, 1000):
    for number_of_quantization_levels in range(10, 100, 10):
        # Fetch 'accuracy[index]' where
        # 'dimensions[index] == number_of_dimensions' and
        # 'quantization_levels[index] == number_of_quantization_levels'.
        index = np.where((dimensions == number_of_dimensions) & 
                         (quantization_levels == number_of_quantization_levels))
        if len(index) > 0:
            index = index[0]  # Take the first index if multiple matches are found
            accuracy_value = accuracy[index]

pass
