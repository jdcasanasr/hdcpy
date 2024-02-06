import numpy as np

def majority_sum_1(arr1, arr2, arr3):
    # Ensure input arrays are of type numpy.bool_
    if not all(isinstance(arr, np.ndarray) and arr.dtype == np.bool_ for arr in [arr1, arr2, arr3]):
        raise ValueError("Input arrays must be numpy.bool_ arrays")

    # Calculate the majority sum element-wise
    result = (arr1 & arr2 & arr3)#.astype(np.int_)

    return result

def majority_sum(a, b, c):
    if not all(isinstance(arr, np.ndarray) and arr.dtype == np.bool_ for arr in [a, b, c]):
        raise ValueError("Input arrays must be numpy.bool_ arrays")
    
    return np.logical_or(np.logical_and(c, np.logical_xor(a, b)), np.logical_and(a, b))
    
def random_bool_array(size):
    return np.random.choice([True, False], size = size, p = [0.5, 0.5])

# Example usage:
#a = np.array([False, False, True])
#b = np.array([False, True, True])
#c = np.array([False, False, True])

a = random_bool_array(5)
b = random_bool_array(5)
c = random_bool_array(5)

print(a)
print(b)
print(c)

result = majority_sum(a, b, c)
print("Majority Sum:", result)
