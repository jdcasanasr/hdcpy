def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
    #with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # Split the line into individual values and convert them to floats
            values = [float(val) for val in line.strip().split(',')]
            data.append(values)
    return data



# Define the path to your training and testing data
train_data_path = 'data/isolet1+2+3+4.data'
test_data_path  = 'data/isolet5.data'

# Load the training and testing data
train_data = load_data(train_data_path)
test_data = load_data(test_data_path)

sample_array    = []
class_array     = []

for data_string in train_data:
    sample_array.append(data_string[0:616])
    class_array.append(data_string[617])

print(class_array)

#for _ in train_data:
#    print(train_data[index])
#    index += step

#isolet_dataset  = open(train_data_path, 'r')
#isolet_lines    = isolet_dataset.readlines()
#isolet_data     = []
#value_data_array = []
#
#for line in isolet_lines:
#    for value in line.strip().split(','):
#        value_data_array.append(float(value))
#
#    isolet_data.append(value_data_array)

#print(train_data.head())
#print(test_data.head())
#print(train_data)
#print(test_data)



# Define the path to your training and testing data
#train_data_path = 'path/to/isolete.train.csv'
#test_data_path = 'path/to/isolete.test.csv'

# Load the data into DataFrames
#train_data  = pd.read_csv(train_data_path, header=None)
#test_data   = pd.read_csv(test_data_path, header=None)
#
#print(train_data.head())
#print(test_data.head())
