import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log_file        = sys.argv[1]
file_name       = re.sub(r'\.[^.]*$', '', os.path.basename(sys.argv[1]))
log_dataframe   = pd.read_csv(log_file, header = None)

dimensions          = np.array(log_dataframe.iloc[:, 0].values)
quantization_levels = np.array(log_dataframe.iloc[:, 1].values)
accuracy            = np.array(log_dataframe.iloc[:, 2].values)
time_per_dataset    = np.array(log_dataframe.iloc[:, 3].values)
time_per_query      = np.array(log_dataframe.iloc[:, 4].values)

# Note: d = dimensions. q = quantization levels.
minimum_d   = np.min(dimensions)
maximum_d   = np.max(dimensions)
d_step      = np.sort(np.unique(dimensions))[1] - minimum_d

minimum_q   = np.min(quantization_levels)
maximum_q   = np.max(quantization_levels)
q_step      = np.sort(np.unique(quantization_levels))[1] - minimum_q

x_axis = np.arange(minimum_d, maximum_d + d_step, d_step)

x_ticks = np.linspace(np.min(dimensions), np.max(dimensions), 10)
y_ticks = np.linspace(np.min(accuracy), np.max(accuracy), 10)

# Plot 'accuracy' vs. dimensions.
for number_of_quantization_levels in range(minimum_q, maximum_q + q_step, q_step):
    y_axis      = []
    for number_of_dimensions in range(minimum_d, maximum_d + d_step, d_step):
        # Get average accuracy over all tests with the current settings.
        condition_1         = dimensions == number_of_dimensions
        condition_2         = quantization_levels == number_of_quantization_levels
        indeces             = np.nonzero(condition_1 & condition_2)[0]
        average_accuracy    = sum(accuracy[indeces]) / len(indeces)
        
        y_axis.append(average_accuracy)

    plt.plot(x_axis, y_axis, label = f'Q = {number_of_quantization_levels}')

plt.title(f'{file_name.upper()} Dataset')
plt.xlabel('Number of Dimensions (Bits)')
plt.xticks(x_axis)
plt.ylabel('Accuracy (%)')
plt.yticks(y_ticks)
plt.legend()
plt.grid()
plt.savefig(f'../assets/{file_name}.pdf', format = 'pdf')

# Plot 'accuracy' vs. 'time_per_dataset'.
# Code goes here...

# Plot 'accuracy' vs. 'time_per_query'.
# Code goes here...