import csv
import matplotlib.pyplot as plt

# Load data from file
data_file = '/home/jdcasanasr/Development/hdcpy/ucihar.csv'  # Change this to your file name
data = []
with open(data_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append((int(row[0]), int(row[1]), float(row[2])))

# Filter data for values 20 and 30 in the second column
filtered_data = [entry for entry in data if entry[1] in range(20, 110, 10)]

# Separate data by label
labels = set(entry[1] for entry in filtered_data)
grouped_data = {label: [[], []] for label in labels}
for entry in filtered_data:
    grouped_data[entry[1]][0].append(entry[0])  # x values
    grouped_data[entry[1]][1].append(entry[2])  # y values

# Plotting
plt.figure()
for label, (x, y) in grouped_data.items():
    plt.plot(x, y, label=f'Q =  {label}')

plt.xlabel('Dimensions (Bits)')
plt.ylabel('Accuracy (%)')
plt.title('UCIHAR Dataset')
plt.legend()
plt.grid(True)
plt.show()
