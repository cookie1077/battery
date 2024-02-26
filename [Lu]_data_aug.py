import torch
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataset_current(data):
    """function to read raw data and then save charge capacity 

    Args:
        data : The tensor dataset 

    Returns:
        None
    """

    current_plan = []

    for cycle in range(1, 102):
        current = []
        charge = np
        discharge = np.max()

    if not it :
        print('shape of result')
        print(data_out_x.shape)
        print(data_out_y.shape)
        it = True

    return data_out_x, data_out_y



data_path = "Code_for_intern/dataset/"
model_path = "Code_for_intern/Model/"
result_path = "Code_for_intern/Results/"

print(os.getcwd())

data_list = os.listdir(data_path)

train_list = [file for file in data_list]


for data_name in train_list:
    data = torch.load(data_path + data_name)

    build_dataset_current(data)
