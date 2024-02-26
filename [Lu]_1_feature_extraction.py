"""
Lu 2022 Public Data Feature Study

Only 55 Cells with random charging policies were treated (discharging rate is fixed at 3C)
For each cell, first 20 cycles were neglected.

Data Column Heads
Data_Point	Test_Time(s)	Current(A)	Capacity(Ah)	Voltage(V)	Energy(Wh)	Temperature(℃)	Date_Time	Cycle_Index
"""

import numpy as np
import pandas as pd
import torch
import os
import re
from scipy.stats import skew


#################################################################
raw_path = "Raw Data/"
save_path = "Code_for_intern/dataset_f1/"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#################################################################

def _find_cycles(file_list):
    if 'first' in file_list[0]:
        return file_list[0], file_list[1]
    else:
        return file_list[1], file_list[0]

def build_dataset_early(folder_path, save_folder_path, save_name):
    """
    Following (Lu et al, 2022)'s implementation 
    creates capacity-voltage matrix.
    """

    
    folder_list = os.listdir(folder_path) # List of Cell Folders
    folder_list = [f for f in folder_list if len(f) < 4] # remove .DS_Store
    print(folder_list)
    
    folder_list = ['#68', '#50', '#66', '#59', '#61', '#60', '#67', '#51', '#56', '#69', '#33', '#34', '#11', '#27', '#18', '#20', '#74', '#7', '#73', '#9', '#42', '#21', '#17', '#28', '#43', '#8', '#72', '#1', '#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']

    try:
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
    except OSError:
        print('Error: Creating directory. ' + save_folder_path)

    for i in range(len(folder_list)):
        print(folder_list[i])
        print(folder_list)
        file_list = os.listdir(folder_path + folder_list[i]) # List of the files in each folder
        data = pd.read_excel(folder_path + folder_list[i] + "/" + file_list[1]) # the first 20 cycles
        cycle, capacity, voltage, current = [], [], [], []

        for cycle_index in range(2, 22):
            cycle.append(cycle_index)
            capacity.append(np.max(data["Capacity(Ah)"][data["Cycle_Index"] == cycle_index]))
            voltage.append(np.mean(data["Voltage(V)"][(data["Cycle_Index"] == cycle_index) & (data["Voltage(V)"] > 3.7)]))
            current.append(np.max(data["Current(A)"][(data["Cycle_Index"] == cycle_index)]))

        cycle = np.array(cycle).reshape(-1, 1)
        capacity = np.array(capacity).reshape(-1, 1)
        voltage = np.array(voltage).reshape(-1 ,1)
        current = np.array(current).reshape(-1, 1)

        features = np.hstack((cycle, capacity, voltage, current))

        features = torch.FloatTensor(features)

        torch.save(features, save_folder_path + save_name + folder_list[i])

        print(f"Data Preparation for {folder_list[i]} is completed")


def build_dataset_lu(folder_path, save_folder_path, save_name):
    """
    Following (Lu et al, 2022)'s implementation 
    creates current operation plan data.
    """

    folder_list = os.listdir(folder_path) # List of Cell Folders
    folder_list = [f for f in folder_list if len(f) < 4] # remove .DS_Store
    print(folder_list)

    folder_list = ['#68', '#50', '#66', '#59', '#61', '#60', '#67', '#56', '#69', '#33', '#34', '#11', '#27', '#18', '#20', '#74', '#7', '#73', '#9', '#42', '#21', '#17', '#28', '#43', '#8', '#72', '#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']
    #folder_list = ['#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']


    try:
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
    except OSError:
        print('Error: Creating directory. ' + save_folder_path)

    for i in range(len(folder_list)):
        file_list = os.listdir(folder_path + folder_list[i]) # List of the files in each folder
        early, cycles = _find_cycles(file_list)
        data = pd.read_excel(folder_path + folder_list[i] + "/" + cycles)
        data_early = pd.read_excel(folder_path + folder_list[i] + "/" + early)  
        current_plan = []

        for cycle_index in range(2, 102):
            current = []
            charge = np.max(data["Current(A)"][(data["Cycle_Index"] == cycle_index)])
            discharge = np.min(data["Current(A)"][(data["Cycle_Index"] == cycle_index)])

            current.append(charge)
            current.append(discharge)

            current_plan.append(current)
        
        print(len(current_plan))

        # q-v matrix
        
        qv_matrix = []
        
        for cycle_index in range(2, 22):
            capacity = data_early["Capacity(Ah)"][data_early["Cycle_Index"] == cycle_index]
            voltage = data_early['Voltage(V)'][data_early["Cycle_Index"] == cycle_index]

            capacity = capacity.reset_index(drop=True)
            voltage = voltage.reset_index(drop=True)

            q = []
            j, last_j = 0, 0
            start_vol = voltage[0]
            high_vol = np.max(voltage)
            gap = (high_vol-start_vol)/120

            while(j == 0 or voltage[j] < high_vol) :
                if voltage[j] >= start_vol :
                    q.append(capacity[j] - capacity[last_j])
                    last_j = j
                    start_vol += gap
                
                j += 1
            
            assert len(q) == 120
            qv_matrix.append(q)


        current_plan = np.array(current_plan)
        qv_matrix = np.array(qv_matrix)

        # Add padding
        qv_matrix = np.pad(qv_matrix, ((0, 80), (0, 0)), 'constant', constant_values=0)

        input = np.hstack((current_plan, qv_matrix))
        input = torch.FloatTensor(input)

        torch.save(input, save_folder_path + save_name + folder_list[i])
        print(f"Data Preparation for {folder_list[i]} is completed")

    return 0

def build_dataset_f1(folder_path, save_folder_path, save_name):
    """
    Preparation for the Capacity data at every cycle for each battery cell.
    Now, some of the previous capacity values are included in the input.
    """

    folder_list = os.listdir(folder_path) # List of Cell Folders
    folder_list = [f for f in folder_list if len(f) < 4] # remove .DS_Store
    folder_list = ['#68', '#50', '#66', '#59', '#61', '#60', '#67', '#51', '#56', '#69', '#33', '#34', '#11', '#27', '#18', '#20', '#74', '#7', '#73', '#9', '#42', '#21', '#17', '#28', '#43', '#8', '#72', '#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']
    print(folder_list)
    #folder_list = ['#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']

    try:
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
    except OSError:
        print('Error: Creating directory. ' + save_folder_path)

    for i in range(len(folder_list)):
        file_list = os.listdir(folder_path + folder_list[i]) # List of the files in each folder
        early, cycles = _find_cycles(file_list)
        data = pd.read_excel(folder_path + folder_list[i] + "/" + cycles, engine='openpyxl') # The first 20 cycles are neglected
        cycle, capacity, variance, skewness, maxima = [], [], [], [], []

        for cycle_index in range(2, 102):
            cycle.append(cycle_index)
            capacity.append(np.max(data["Capacity(Ah)"][data["Cycle_Index"] == cycle_index]))
            voltage = data["Voltage(V)"][data["Cycle_Index"] == cycle_index-1]
            voltage = list(voltage.reset_index(drop=True))

            max_index = voltage.index(max(voltage))
            min_index = voltage.index(min(voltage))

            voltage = voltage[max_index:min_index]

            variance.append(np.var(voltage))
            skewness.append(skew(voltage))
            maxima.append(np.max(voltage))

        cycle = np.array(cycle).reshape(-1, 1)
        capacity = np.array(capacity).reshape(-1, 1)
        variance = np.array(variance).reshape(-1, 1)
        skewness = np.array(skewness).reshape(-1 ,1)
        maxima = np.array(maxima).reshape(-1, 1)

        features = np.hstack((cycle, capacity, variance, skewness, maxima))

        features = torch.FloatTensor(features)

        torch.save(features, save_folder_path + save_name + folder_list[i])

        print(f"Data Preparation for {folder_list[i]} is completed")

    return 0


def build_dataset_f2(folder_path, save_folder_path, save_name):
    """
    Preparation for the Capacity data at every cycle for each battery cell.
    Now, some of the previous capacity values are included in the input.
    """

    folder_list = os.listdir(folder_path) # List of Cell Folders
    folder_list = [f for f in folder_list if len(f) < 4] # remove .DS_Store
    folder_list = ['#68', '#50', '#66', '#59', '#61', '#60', '#67', '#51', '#56', '#69', '#33', '#34', '#11', '#27', '#18', '#20', '#74', '#7', '#73', '#9', '#42', '#21', '#17', '#28', '#43', '#8', '#72', '#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']
    print(folder_list)
    #folder_list = ['#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']

    try:
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
    except OSError:
        print('Error: Creating directory. ' + save_folder_path)

    for i in range(len(folder_list)):
        file_list = os.listdir(folder_path + folder_list[i]) # List of the files in each folder
        early, cycles = _find_cycles(file_list)
        data = pd.read_excel(folder_path + folder_list[i] + "/" + cycles, engine='openpyxl') # The first 20 cycles are neglected
        capacity, voltage, capacity_prev, d_v, d_q = [], [], [], [], []

        for cycle_index in range(12, 102):
            capacity.append(np.max(data["Capacity(Ah)"][data["Cycle_Index"] == cycle_index]))
            capacity_prev.append(np.max(data["Capacity(Ah)"][data["Cycle_Index"] == cycle_index-1]))
            voltage.append(np.max(data["Voltage(V)"][(data["Cycle_Index"] == cycle_index-1) & (data["Voltage(V)"] > 3.7)]))
            d_v.append(np.max(data["Voltage(V)"][(data["Cycle_Index"] == cycle_index-11)]))
            d_q.append(np.max(data["Capacity(Ah)"][data["Cycle_Index"] == cycle_index-11]))
        
        
        # Create an array of indices for the original vector
        x_indices = np.arange(len(capacity_prev))

        # Create a new array of indices for the interpolated vector
        new_indices = np.linspace(0, len(capacity_prev)-1, 100)

        # Use numpy.interp for linear interpolation
        capacity = np.interp(new_indices, x_indices, np.array(capacity))
        capacity_prev = np.interp(new_indices, x_indices, np.array(capacity_prev))
        voltage = np.interp(new_indices, x_indices, np.array(voltage))
        d_v = np.interp(new_indices, x_indices, np.array(d_v))
        d_q = np.interp(new_indices, x_indices, np.array(d_q))

        capacity = np.array(capacity).reshape(-1, 1)
        capacity_prev = np.array(capacity_prev).reshape(-1, 1)
        voltage = np.array(voltage).reshape(-1 ,1)
        d_v = np.array(d_v).reshape(-1, 1)
        d_q = np.array(d_q).reshape(-1, 1)

        features = np.hstack((capacity, capacity_prev, voltage, d_v, d_q))

        features = torch.FloatTensor(features)

        torch.save(features, save_folder_path + save_name + folder_list[i])

        print(f"Data Preparation for {folder_list[i]} is completed")

    return 0


def build_dataset_f3(folder_path, save_folder_path, save_name):
    """
    Preparation for the Capacity data at every cycle for each battery cell.
    Now, some of the previous capacity values are included in the input.
    """

    folder_list = os.listdir(folder_path) # List of Cell Folders
    folder_list = [f for f in folder_list if len(f) < 4] # remove .DS_Store
    folder_list = ['#68', '#50', '#66', '#59', '#61', '#60', '#67', '#51', '#56', '#69', '#33', '#34', '#11', '#27', '#18', '#20', '#74', '#7', '#73', '#9', '#42', '#21', '#17', '#28', '#43', '#8', '#72', '#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']
    print(folder_list)
    #folder_list = ['#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']

    try:
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
    except OSError:
        print('Error: Creating directory. ' + save_folder_path)

    for i in range(len(folder_list)):
        file_list = os.listdir(folder_path + folder_list[i]) # List of the files in each folder
        early, cycles = _find_cycles(file_list)
        data = pd.read_excel(folder_path + folder_list[i] + "/" + cycles, engine='openpyxl') # The first 20 cycles are neglected
        voltages = []

        for cycle_index in range(2, 102):
            voltage = data["Voltage(V)"][data["Cycle_Index"] == cycle_index-1]
            voltage = list(voltage.reset_index(drop=True))
            max_index = voltage.index(max(voltage))

            voltage = voltage[max_index:]
            first_voltage = []

            for j in range(10):
                first_voltage.append(list([voltage[j], voltage[j+1] - voltage[j]]))

            voltages.append(first_voltage)
    

        voltages = torch.FloatTensor(voltages)

        torch.save(voltages, save_folder_path + save_name + folder_list[i])

        print(f"Data Preparation for {folder_list[i]} is completed")

    return 0


def build_dataset(folder_path, save_folder_path, save_name):
    """
    Preparation for the Capacity data at every cycle for each battery cell.
    Now, some of the previous capacity values are included in the input.
    """

    folder_list = os.listdir(folder_path) # List of Cell Folders
    folder_list = [f for f in folder_list if len(f) < 4] # remove .DS_Store
    folder_list = ['#68', '#50', '#66', '#59', '#61', '#60', '#67', '#51', '#56', '#69', '#33', '#34', '#11', '#27', '#18', '#20', '#74', '#7', '#73', '#9', '#42', '#21', '#17', '#28', '#43', '#8', '#72', '#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']
    print(folder_list)
    #folder_list = ['#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']

    try:
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
    except OSError:
        print('Error: Creating directory. ' + save_folder_path)

    for i in range(len(folder_list)):
        file_list = os.listdir(folder_path + folder_list[i]) # List of the files in each folder
        early, cycles = _find_cycles(file_list)
        data = pd.read_excel(folder_path + folder_list[i] + "/" + cycles, engine='openpyxl') # The first 20 cycles are neglected
        cycle, capacity, voltage, current, capacity_prev = [], [], [], [], []

        for cycle_index in range(2, 102):
            cycle.append(cycle_index)
            capacity.append(np.max(data["Capacity(Ah)"][data["Cycle_Index"] == cycle_index]))
            voltage.append(np.mean(data["Voltage(V)"][(data["Cycle_Index"] == cycle_index) & (data["Voltage(V)"] > 3.7)]))
            current.append(np.max(data["Current(A)"][(data["Cycle_Index"] == cycle_index)]))
            capacity_prev.append(np.max(data["Capacity(Ah)"][data["Cycle_Index"] == cycle_index-1]))

        cycle = np.array(cycle).reshape(-1, 1)
        capacity = np.array(capacity).reshape(-1, 1)
        voltage = np.array(voltage).reshape(-1 ,1)
        current = np.array(current).reshape(-1, 1)
        capacity_prev = np.array(capacity_prev).reshape(-1, 1)

        features = np.hstack((cycle, capacity, voltage, current, capacity_prev))

        features = torch.FloatTensor(features)

        torch.save(features, save_folder_path + save_name + folder_list[i])

        print(f"Data Preparation for {folder_list[i]} is completed")

    return 0


#A = build_dataset(raw_path, save_path, "dataset_cell_")

build_dataset_f3(raw_path, "Code_for_intern/dataset_f3/", "dataset_cell_")