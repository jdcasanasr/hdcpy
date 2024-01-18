#import codecs
#import pandas as pd

# Define the path to your training and testing data
train_data_path = '/home/jdcasanasr/Downloads/isolet1+2+3+4.data'
test_data_path  = '/home/jdcasanasr/Downloads/isolet5.data'

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
    #with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # Split the line into individual values and convert them to floats
            values = [float(val) for val in line.strip().split(',')]
            data.append(values)
    return data

# Load the training and testing data
train_data = load_data(train_data_path)
test_data = load_data(test_data_path)

#print(train_data.head())
#print(test_data.head())
#print(train_data)
print(test_data)



# Define the path to your training and testing data
#train_data_path = 'path/to/isolete.train.csv'
#test_data_path = 'path/to/isolete.test.csv'

# Load the data into DataFrames
#train_data  = pd.read_csv(train_data_path, header=None)
#test_data   = pd.read_csv(test_data_path, header=None)
#
#print(train_data.head())
#print(test_data.head())
