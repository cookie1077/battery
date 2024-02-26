import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

folder_path = "dataset\\"
file_list = os.listdir(folder_path)
# data = torch.load("Data\\Public Data\\Lu 2022\\Manipulated_Datasets\\dataset_3\\dataset_3_cell_#1")

rul = []
max_capacity = []
feature_1 = []
feature_2 = []

for file_name in file_list:
    data = torch.load(folder_path + file_name)

    rul_item = data[:, 0] # Cycles
    max_capacity_item = data[:, 1] # Max Capacity
    feature_1_item = data[:, 2] # Mean Voltage
    feature_2_item = data[:, 3] # Max Current

    for i in range(len(rul_item)):
        rul.append(rul_item[i])
        max_capacity.append(max_capacity_item[i])
        feature_1.append(feature_1_item[i])
        feature_2.append(feature_2_item[i])

analysis_1 = stats.pearsonr(rul, feature_1)
correlation_coefficient_1 = analysis_1[0]
p_value_1 = analysis_1[1]

analysis_2 = stats.pearsonr(rul, feature_2)
correlation_coefficient_2 = analysis_2[0]
p_value_2 = analysis_2[1]

analysis_3 = stats.pearsonr(max_capacity, feature_1)
correlation_coefficient_3 = analysis_3[0]
p_value_3 = analysis_3[1]

analysis_4 = stats.pearsonr(max_capacity, feature_2)
correlation_coefficient_4 = analysis_4[0]
p_value_4 = analysis_4[1]

print("Feature #1 Analysis")
print("Correlation: {:.4f}".format(correlation_coefficient_1))
print("p_value: {:.4f}".format(p_value_1))
print("")

plt.scatter(feature_1, rul, s=3)
plt.title("RUL Correlation")
plt.xlabel("Mean Voltage")
plt.ylabel("RUL")
plt.show()

print("Feature #2 Analysis")
print("Correlation: {:.4f}".format(correlation_coefficient_2))
print("p_value: {:.4f}".format(p_value_2))
print("")

plt.scatter(feature_2, rul, s=3)
plt.title("RUL Correlation")
plt.xlabel("Max Current")
plt.ylabel("RUL")
plt.show()

print("Feature #3 Analysis")
print("Correlation: {:.4f}".format(correlation_coefficient_3))
print("p_value: {:.4f}".format(p_value_3))
print("")

plt.scatter(feature_1, max_capacity, s=3)
plt.title("RUL Correlation")
plt.xlabel("Mean Voltage")
plt.ylabel("Max Capacity")
plt.show()

print("Feature #4 Analysis")
print("Correlation: {:.4f}".format(correlation_coefficient_4))
print("p_value: {:.4f}".format(p_value_4))
print("")

plt.scatter(feature_2, max_capacity, s=3)
plt.title("Max Capacity Correlation")
plt.xlabel("Max Current")
plt.ylabel("Max Capacity")
plt.show()