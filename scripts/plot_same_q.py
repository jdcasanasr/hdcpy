import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log_file        = sys.argv[1]
#log_file        = '/home/jdcasanasr/Development/hdcpy/logs/isolet_same_q.csv'
file_name       = re.sub(r'\.[^.]*$', '', os.path.basename(sys.argv[1]))
#file_name       = re.sub(r'\.[^.]*$', '', os.path.basename(log_file))
log_dataframe   = pd.read_csv(log_file, header = None)

dimensions          = np.array(log_dataframe.iloc[:, 0].values)
quantization_levels = np.array(log_dataframe.iloc[:, 1].values)
accuracy            = np.array(log_dataframe.iloc[:, 2].values)

x_axis = np.arange(1000, 11000, 1000)

x_ticks = np.linspace(np.min(dimensions), np.max(dimensions), 10)
y_ticks = np.linspace(70, 90, 10)

# Iterate over all different tests, and plot each 
# graph separately.
# ToDo: Detect the number of tests automatically.
index = 0

for test in range(5):
    y_axis = []
    for number_of_dimensions in range(1000, 11000, 1000):
        accuracy_array = accuracy[index:index + 5]
        y_axis.append(sum(accuracy_array) / len(accuracy_array))
        index += 5
    plt.plot(x_axis, y_axis, label = f'Run {test + 1}')

# ToDo: Prompt the user for a custom title.
plt.title('ISOLET Dataset (Q = 10)')
plt.xlabel('Number of Dimensions (Bits)')
plt.xticks(x_axis)
plt.ylabel('Accuracy (%)')
plt.yticks(y_ticks)
plt.legend()
plt.grid()
#plt.show()
plt.savefig(f'../assets/{file_name}.jpeg', format = 'jpeg')