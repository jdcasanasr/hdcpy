import numpy as np
import os

number_of_dimensions            = np.linspace(1000, 10000, 10).astype(int) # Step = 1000
number_of_quantization_levels   = np.linspace(2, 10, 9).astype(int)
number_of_total_iterations      = np.size(number_of_dimensions) * np.size(number_of_quantization_levels)
test_script                     = 'ucihar_unified_test.py'
number_of_current_iterations    = 0

for d in number_of_dimensions:
    for q in number_of_quantization_levels:
        number_of_current_iterations += 1
        print(f'Test: \'{test_script}\' {((number_of_current_iterations / number_of_total_iterations) * 100):0.2f}| D = {d} | Q = {q}')
        os.system("ucihar_unified_test.py {d} {q}")
