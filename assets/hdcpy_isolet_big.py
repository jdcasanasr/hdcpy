import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from hdcpy import *

log_path            = '/home/jdcasanasr/Development/hdcpy/logs/ucihar.log'
dataframe           = pd.read_csv(log_path, header = None)
number_of_tests     = 3

dimensions          = dataframe.iloc[:, 0].values
quantization_levels = dataframe.iloc[:, 1].values
accuracy            = dataframe.iloc[:, 2].values

effective_accuracies            = []
effective_dimensions            = []
effective_quantization_levels   = []

i = 0

while i < len(accuracy):
    effective_accuracies.append(round(((accuracy[i] + accuracy[i + 1] + accuracy[i + 2]) / number_of_tests), 2))
    i += 3

i = 0

while i < len(dimensions):
    effective_dimensions.append(dimensions[i])
    i += 3

i = 0

while i < len(quantization_levels):
    effective_quantization_levels.append(quantization_levels[i])
    i += 3

a = np.array(effective_accuracies)
d = np.array(effective_dimensions)
q = np.array(effective_quantization_levels)

df = pd.DataFrame({"Dimensions" : d, "Quantization Levels" : q, "Accuracy" : a})
df.to_csv("ucihar.csv", index = False)