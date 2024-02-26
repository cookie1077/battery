import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import csv

from model_new import RNN_LSTM, RNN_GRU, TransformerModel, RNN_LU

device = "cuda" if torch.cuda.is_available() else "cpu"


####################################################################################
data_path = "Code_for_intern/dataset_f1/"
data_new_path = "Code_for_intern/dataset_new/"
data_lu_path = "Code_for_intern/dataset_lu_new/"
model_path = "Code_for_intern/Model/"
result_path = "Code_for_intern/Results/"
error_plot_path = "Code_for_intern/Error_plot/"

print(os.getcwd())

data_list = os.listdir(data_path)
#data_list = ['#68', '#50', '#66', '#59', '#61', '#60', '#67', '#51', '#56', '#69', '#33', '#34', '#11', '#27', '#18', '#20', '#74', '#7', '#73', '#9', '#42', '#21', '#17', '#28', '#43', '#8', '#72', '#1', '#75', '#31', '#36', '#54', '#62', '#65', '#39', '#37', '#30', '#64', '#63', '#55', '#4', '#70', '#3', '#77', '#46', '#12', '#15', '#24', '#47', '#40', '#76', '#5', '#71', '#25', '#14']

# train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=1)

test_list = ["dataset_cell_#3", "dataset_cell_#7", "dataset_cell_#25", "dataset_cell_#28", "dataset_cell_#30",
             "dataset_cell_#31", "dataset_cell_#33", "dataset_cell_#54", "dataset_cell_#70",
             "dataset_cell_#71", "dataset_cell_#76"]
train_list = [file for file in data_list if file not in test_list and file != "dataset_cell_#1" and file != "dataset_cell_#51"]
#train_list = ["dataset_cell_#3"]

random.shuffle(train_list)

model_name_gru = "GRU_Lu"
model_name_lstm = "LSTM_Lu"
model_name_transformer = "Transformer_Lu"

####################################################################################

all_results_lu = []
all_results_gru = []
all_results_lstm = []
all_results_transformer = []
csv_filename_lu = os.path.join(result_path, "results_lu.csv")
csv_filename_gru = os.path.join(result_path, "results_gru.csv")
csv_filename_lstm = os.path.join(result_path, "results_lstm.csv")
csv_filename_transformer = os.path.join(result_path, "results_transformer.csv")
it = False


def build_dataset(data_x, data_y, seq_length):
    """function to read raw data and then transform them into length n-1 and n-2 packets

    Args:
        data_x : [current, capacity_prev]
        data_y : [capacity]
        seq_length : length of sequence to look back 

    Returns:
        data_out_x : list of sequence length chops of data
        data_out_y : list of corresponding capacity data

        for i in range(len(data_x)):
        max = np.max(data_x[i])
        min = np.min(data_x[i])

        data_x[i] = ((data_x[i] - min) / (max - min)) - (max - min)/2
    """

    data_out_x = []
    data_out_y = []

    for i in range(0, len(data_x) - seq_length):
        _x = data_x[i:i + seq_length, :]
        data_out_x.append(_x)

    for i in range(0, len(data_y) - seq_length):
        _y = data_y[i+seq_length, :]
        data_out_y.append(_y)

    data_out_x = np.array(data_out_x)
    data_out_x = torch.FloatTensor(data_out_x)
    data_out_x = data_out_x.to(device)

    data_out_y = np.array(data_out_y)
    data_out_y = torch.FloatTensor(data_out_y)
    data_out_y = data_out_y.to(device)
    

    return data_out_x, data_out_y


def run_experiment_lu(iteration, epochs=200, seq_length=10, lr=1e-3):
    """function to run experiment, evaluate, and save results and logs for lu model 

    Args:
        iteration: the number of iterations wanted

    Returns:
        None
    """
    print(f"Experiment {iteration} started.")

    results_filename = os.path.join(result_path, "results_lu.txt")

    iteration_path = os.path.join(result_path, f"LU_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(iteration_path, exist_ok=True)

    error_path = os.path.join(error_plot_path, f"LU_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(error_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train : model to device! could change the dimensions.
    model = RNN_LU(input_dim=122, hidden_dim=256, output_dim=2).to(device)
    #print(list(model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8) 
 
    average_loss_list = []
    average_rmse_test = []

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0
        num_data = len(train_list)
        for data_name in train_list:
            data = torch.load(data_new_path + data_name)
            lu_data = torch.load(data_lu_path + data_name)

            # Key : use only two 
            x = lu_data
            y = data[:, 1][:, None]  # Capacity

            train_x = x.unsqueeze(0).to(device)
            train_y = y.to(device)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()
        average_epoch_loss = epoch_loss / num_data
        average_loss_list.append(average_epoch_loss)

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_tmp = []
            r2_scores_tmp = []
            for file_name in test_list:
                test_data = torch.load(data_new_path + file_name)
                test_lu_data = torch.load(data_lu_path + file_name)

                x = test_lu_data
                y = test_data[:, 1][:, None]

                test_x = x.unsqueeze(0).to(device)
                test_y = y.to(device)

                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[:, 0]

                    y_ = model(test_x)

                    y_true_tmp.append(y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)


                    rmse_loss_tmp = np.sqrt(mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())

                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_tmp.append(rmse_loss_tmp)
                        r2_scores_tmp.append(r2_tmp)

            if rmse_scores_tmp and r2_scores_tmp:
                average_rmse_tmp = np.mean(rmse_scores_tmp)
                average_r2_tmp = np.mean(r2_scores_tmp)
                average_rmse_test.append(average_rmse_tmp)
            else:
                average_rmse_tmp = np.nan
                average_r2_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_tmp}, Average R2: {average_r2_tmp}")

    torch.save(model, model_path + model_name_transformer)

    # Evaluating the model
    model = torch.load(model_path + model_name_transformer)

    # Saving error plot 
    indices = np.linspace(0, len(average_rmse_test)-1, len(average_loss_list))

    # Interpolate list2 to match the length of list1
    list2_interp = np.interp(indices, np.arange(len(average_rmse_test)), average_rmse_test)

    # Now you can plot list1 and list2_interp
    plt.plot(average_loss_list, label='Train loss')
    plt.plot(list2_interp, label='Test loss')
    plt.legend()
    plt.ylim([0.0, 0.3])
    plt.savefig(os.path.join(error_path, f"EMSE.png"), dpi=200)
    plt.close()

    for file_name in test_list:
        test_data = torch.load(data_new_path + file_name)
        test_lu_data = torch.load(data_lu_path + file_name)

        x = test_lu_data
        y = test_data[:, 1][:, None]  

        test_x = x.unsqueeze(0).to(device)
        test_y = y.to(device)

        y_true, y_pred = [], []
        model.eval()

        with torch.no_grad():
            cycle = test_data[:, 0]
            y_ = model(test_x)

            y_true.append(y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        plt.plot(cycle.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True Capacity")
        plt.plot(cycle.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted Capacity")
        plt.title(f"Degradation Prediction of the Battery {file_name}")
        plt.xlabel("Number of Cycles")
        plt.ylabel("Battery Capacity (Ah)")
        plt.legend()
        plt.savefig(os.path.join(iteration_path, f"{file_name}.png"), dpi=200)
        plt.close()

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)


    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")

    all_results_lu.append({
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[LU] Experiment {iteration} completed and results saved.")
    print("")


def run_experiment_gru(iteration, epochs=100, seq_length=10, lr=1e-3):
    """function to run experiment, evaluate, and save results and logs for gru model 

    Args:
        iteration: the number of iterations wanted

    Returns:
        None
    """
    print(f"Experiment {iteration} started.")

    results_filename = os.path.join(result_path, "results_gru.txt")

    iteration_path = os.path.join(result_path, f"GRU_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(iteration_path, exist_ok=True)

    error_path = os.path.join(error_plot_path, f"GRU_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(error_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train : model to device! could change the dimensions.
    model = RNN_GRU(input_dim=3, hidden_dim=10, output_dim=1, layers=3).to(device)
    #print(list(model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    # 100 epochs for training 

    average_loss_list = []
    average_rmse_test = []

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0
        num_data = len(train_list)
        for data_name in train_list:
            data = torch.load(data_path + data_name)

            # Key : use only two 
            x = data[:, 2:]  
            y = data[:, 1][:, None]  # Capacity

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()
        average_epoch_loss = epoch_loss / num_data
        average_loss_list.append(average_epoch_loss)

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_tmp = []
            r2_scores_tmp = []
            for file_name in test_list:
                test_data = torch.load(data_path + file_name)

                x = test_data[:, 2:]  
                y = test_data[seq_length:, 1][:, None]  

                test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[seq_length:, 0]

                    y_ = model(test_x)

                    y_true_tmp.append(y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())

                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_tmp.append(rmse_loss_tmp)
                        r2_scores_tmp.append(r2_tmp)

            if rmse_scores_tmp and r2_scores_tmp:
                average_rmse_tmp = np.mean(rmse_scores_tmp)
                average_r2_tmp = np.mean(r2_scores_tmp)
                average_rmse_test.append(average_rmse_tmp)
            else:
                average_rmse_tmp = np.nan
                average_r2_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_tmp}, Average R2: {average_r2_tmp}")

    torch.save(model, model_path + model_name_transformer)

    # Evaluating the model
    model = torch.load(model_path + model_name_transformer)

    # Saving error plot 
    indices = np.linspace(0, len(average_rmse_test)-1, len(average_loss_list))

    # Interpolate list2 to match the length of list1
    list2_interp = np.interp(indices, np.arange(len(average_rmse_test)), average_rmse_test)

    # Now you can plot list1 and list2_interp
    plt.plot(average_loss_list, label='Train loss')
    plt.plot(list2_interp, label='Test loss')
    plt.ylim([0.0, 0.3])
    plt.legend()
    plt.savefig(os.path.join(error_path, f"RMSE.png"), dpi=200)
    plt.close()


    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, 2:]  # [current, capacity_prev]
        y = test_data[seq_length:, 1][:, None]  # Capacity

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()

        with torch.no_grad():
            cycle = test_data[seq_length:, 0]
            y_ = model(test_x)

            y_true.append(y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        plt.plot(cycle.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True Capacity")
        plt.plot(cycle.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted Capacity")
        plt.title(f"Degradation Prediction of the Battery {file_name}")
        plt.xlabel("Number of Cycles")
        plt.ylabel("Battery Capacity (Ah)")
        plt.legend()
        plt.savefig(os.path.join(iteration_path, f"{file_name}.png"), dpi=200)
        plt.close()

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)


    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")

    all_results_gru.append({
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[GRU] Experiment {iteration} completed and results saved.")
    print("")


def run_experiment_lstm(iteration, epochs=100, seq_length=10, lr=1e-3):
    """function to run experiment, evaluate, and save results and logs for lstm model 

    Args:
        iteration: the number of iterations wanted

    Returns:
        None
    """
    print(f"Experiment {iteration} started.")

    results_filename = os.path.join(result_path, "results_lstm.txt")

    iteration_path = os.path.join(result_path, f"LSTM_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(iteration_path, exist_ok=True)

    error_path = os.path.join(error_plot_path, f"LSTM_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(error_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train
    model = RNN_LSTM(input_dim=3, hidden_dim=10, output_dim=1, layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    average_loss_list = []
    average_rmse_test = []

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0
        num_data = len(train_list)
        for data_name in train_list:
            data = torch.load(data_path + data_name)

            x = data[:, 2:]  # [current, capacity_prev]
            y = data[:, 1][:, None]  # Capacity

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()
        average_epoch_loss = epoch_loss / num_data
        average_loss_list.append(average_epoch_loss)

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_tmp = []
            r2_scores_tmp = []
            for file_name in test_list:
                test_data = torch.load(data_path + file_name)

                x = test_data[:, 2:]  # [current, capacity_prev]
                y = test_data[seq_length:, 1][:, None]  # Capacity

                test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[seq_length:, 0]

                    y_ = model(test_x)

                    y_true_tmp.append(y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())
                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_tmp.append(rmse_loss_tmp)
                        r2_scores_tmp.append(r2_tmp)

            if rmse_scores_tmp and r2_scores_tmp:
                average_rmse_tmp = np.mean(rmse_scores_tmp)
                average_r2_tmp = np.mean(r2_scores_tmp)
                average_rmse_test.append(average_rmse_tmp)
            else:
                average_rmse_tmp = np.nan
                average_r2_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_tmp}, Average R2: {average_r2_tmp}")

    torch.save(model, model_path + model_name_transformer)

    # Evaluating the model
    model = torch.load(model_path + model_name_transformer)

    # Saving error plot 
    indices = np.linspace(0, len(average_rmse_test)-1, len(average_loss_list))

    # Interpolate list2 to match the length of list1
    list2_interp = np.interp(indices, np.arange(len(average_rmse_test)), average_rmse_test)

    # Now you can plot list1 and list2_interp
    plt.plot(average_loss_list, label='Train loss')
    plt.plot(list2_interp, label='Test loss')
    plt.ylim([0.0, 0.3])
    plt.legend()
    plt.savefig(os.path.join(error_path, f"RMSE.png"), dpi=200)
    plt.close()

    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, 2:]  # [current, capacity_prev]
        y = test_data[seq_length:, 1][:, None]  # Capacity

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            cycle = test_data[seq_length:, 0]

            y_ = model(test_x)

            y_true.append(y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        plt.plot(cycle.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True Capacity")
        plt.plot(cycle.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted Capacity")
        plt.title(f"Degradation Prediction of the Battery {file_name}")
        plt.xlabel("Number of Cycles")
        plt.ylabel("Battery Capacity (Ah)")
        plt.legend()
        plt.savefig(os.path.join(iteration_path, f"{file_name}.png"), dpi=200)
        plt.close()

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)


    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")

    all_results_lstm.append({
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[LSTM] Experiment {iteration} completed and results saved.")
    print("")


def run_experiment_transformer(iteration, epochs=200, seq_length=10, lr=1e-3):
    """function to run experiment, evaluate, and save results and logs for transformer model 

    Args:
        iteration: the number of iterations wanted

    Returns:
        None
    """
    print(f"Experiment {iteration} started.")

    results_filename = os.path.join(result_path, "results_transformer.txt")

    iteration_path = os.path.join(result_path, f"Transformer_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(iteration_path, exist_ok=True)

    error_path = os.path.join(error_plot_path, f"Transformer_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(error_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train
    model = TransformerModel(input_dim=3, hidden_dim=32, output_dim=1, num_layers=3, nhead=8, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    average_loss_list = []
    average_rmse_test = []

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0
        num_data = len(train_list)
        for data_name in train_list:
            data = torch.load(data_path + data_name)

            x = data[:, 2:]  # [current, capacity_prev]
            y = data[:, 1][:, None]  # Capacity

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()
        average_epoch_loss = epoch_loss / num_data
        average_loss_list.append(average_epoch_loss)

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_tmp = []
            r2_scores_tmp = []
            for file_name in test_list:
                test_data = torch.load(data_path + file_name)

                x = test_data[:, 2:]  # [current, capacity_prev]
                y = test_data[seq_length:, 1][:, None]  # Capacity

                test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[seq_length:, 0]

                    y_ = model(test_x)

                    y_true_tmp.append(y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())
                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_tmp.append(rmse_loss_tmp)
                        r2_scores_tmp.append(r2_tmp)

            if rmse_scores_tmp and r2_scores_tmp:
                average_rmse_tmp = np.mean(rmse_scores_tmp)
                average_r2_tmp = np.mean(r2_scores_tmp)
                average_rmse_test.append(average_rmse_tmp)
            else:
                average_rmse_tmp = np.nan
                average_r2_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_tmp}, Average R2: {average_r2_tmp}")

    torch.save(model, model_path + model_name_transformer)

    # Evaluating the model
    model = torch.load(model_path + model_name_transformer)

    # Saving error plot 
    indices = np.linspace(0, len(average_rmse_test)-1, len(average_loss_list))

    # Interpolate list2 to match the length of list1
    list2_interp = np.interp(indices, np.arange(len(average_rmse_test)), average_rmse_test)

    # Now you can plot list1 and list2_interp
    plt.plot(average_loss_list, label='Train loss')
    plt.plot(list2_interp, label='Test loss')
    plt.ylim([0.0, 0.3])
    plt.legend()
    plt.savefig(os.path.join(error_path, f"RMSE.png"), dpi=200)
    plt.close()

    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, 2:]  # [current, capacity_prev]
        y = test_data[seq_length:, 1][:, None]  # Capacity

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            cycle = test_data[seq_length:, 0]

            y_ = model(test_x)

            y_true.append(y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        plt.plot(cycle.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True Capacity")
        plt.plot(cycle.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted Capacity")
        plt.title(f"Degradation Prediction of the Battery {file_name}")
        plt.xlabel("Number of Cycles")
        plt.ylabel("Battery Capacity (Ah)")
        plt.legend()
        plt.savefig(os.path.join(iteration_path, f"{file_name}.png"), dpi=200)
        plt.close()

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)


    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")

    all_results_transformer.append({
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[Transformer] Experiment {iteration} completed and results saved.")
    print("")


# Run experiments!
s = [50, 60, 70, 80, 90]
lr = [1e-5, 1e-4, 5e-4]
#lr = [7e-3]


for i in range(1):
    #run_experiment_lu(i, epochs=200)
    #run_experiment_gru(i, epochs = 200, seq_length=10, lr=6e-3)
    #run_experiment_lstm(i, epochs = 200, seq_length=10, lr=3e-3)
    run_experiment_transformer(i, epochs = 200, seq_length=10, lr=1e-3)




'''
with open(csv_filename_lu, 'w', newline='') as csvfile:
    fieldnames = list(all_results_lu[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_lu:
        writer.writerow(result)


with open(csv_filename_gru, 'w', newline='') as csvfile:
    fieldnames = list(all_results_gru[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_gru:
        writer.writerow(result)


with open(csv_filename_lstm, 'w', newline='') as csvfile:
    fieldnames = list(all_results_lstm[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_lstm:
        writer.writerow(result)


with open(csv_filename_transformer, 'w', newline='') as csvfile:
    fieldnames = list(all_results_transformer[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_transformer:
        writer.writerow(result)

'''


print("All results saved to CSV.")