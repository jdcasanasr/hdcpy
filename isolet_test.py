# Define the path to your training and testing data
train_data_path = '/home/jdcasanasr/Downloads/isolet1+2+3+4.data.Z'
test_data_path  = '/home/jdcasanasr/Downloads/isolet5.data.Z'

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual values and convert them to floats
            values = [float(val) for val in line.strip().split(',')]
            data.append(values)
    return data

# Load the training and testing data
train_data = load_data(train_data_path)
test_data = load_data(test_data_path)

print(train_data.head())
print(test_data.head())
