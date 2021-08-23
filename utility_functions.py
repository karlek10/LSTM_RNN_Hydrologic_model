# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:42:41 2021

@author: karlo
"""

import glob
from platform import python_version
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import Variable
import seaborn as sns
import matplotlib.dates as mdates
from fastprogress import master_bar, progress_bar
from sklearn.preprocessing import StandardScaler
import os
import datetime




def print_vesrions():
    print("python version==%s" % python_version())
    print("pandas==%s" % pd.__version__)
    print("numpy==%s" % np.__version__)
    print("sklearn==%s" % sklearn.__version__)
    print("torch==%s" % torch.__version__)



def load_data(model_number="Drava_model_1"):
    path = "input_data/" + model_number + "_input_data.csv"
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, index_col="Date", encoding="windows-1250") # Day first if Croation format
    model_data = df[df.columns[:]].replace(np.nan,0)
    return model_data    

# function to plot the loss during training
def plot_loss(epochs, train_loss, test_loss):
    plt.figure(figsize=[12., 6.])
    plt.plot(epochs, train_loss, label = "Training Loss")
    plt.plot(epochs, test_loss, label = "Validation Loss")
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show() 


def plot_timeseries(df, model_number, start_date=None, end_date = None, xlabel="Date", ylabel="Runoff (m$^3$/s)", label = "Training"):
    """
    Takes in a dataframe, plots the values as timeserieses.
    Parameters
    ----------
    df : dataframe
    start_date : string --> ("2000-01-01") ("Y-m-d")   
    end_date : string --> ("2001-03-31") ("Y-m-d")
        DEFAULT: whole DF
    xlabel : string, "Date"
        The default is "Date".
    ylabel : string, "Runoff", "Temperature", etc...
        DESCRIPTION. The default is "Runoff (m$^3$/s)".
    """
    sns.set()
    fig, ax = plt.subplots(figsize=(14,8), dpi=600)
    df.plot(ax=ax)
    # date fromating
    myFmt = mdates.DateFormatter('%d.%m.%Y')
    ax.xaxis.set_major_formatter(myFmt)
    # format the labels and ticks
    ax.xaxis.set_tick_params(which='major', pad=0, labelsize=18, rotation=30)
    ax.xaxis.set_tick_params(which='minor', pad=0, labelsize=15)   
    ax.yaxis.set_tick_params(which="major", labelsize=18)
    ax.set_xlabel(xlabel, size=18, va = 'top')
    ax.set_ylabel(ylabel, size=18)
    #setting the axis limits
    ax.set_xlim((start_date, end_date))
    labels = df.columns.tolist()
    ax.legend(labels=[x.capitalize() for x in labels], fontsize = 18) # display the legend
    ax.set_title(model_number + " - " + label, fontsize=20)
    plt.show()
    fig.savefig(str(label)+ "_"+ str(model_number) + ".png", bbox_inches='tight')         
    

# Streamflow upstream + rain --> streamflow downstream
def transform_data_multistep(arr, seq_len_in, seq_len_out, input_size):
    x, y = [], []
    
    for i in range(len(arr)-(seq_len_in+seq_len_out)): # len(df)-(seq_len_in+seq_len_out)
        
        x_i = arr[i : i+seq_len_in+1, 1:input_size+1]  # 
        y_i = arr[i + seq_len_in : i + seq_len_in + seq_len_out,0]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len_in+1, input_size).astype("float32")
    y_arr = np.array(y).reshape(-1, seq_len_out).astype("float32")
    x_var = Variable(torch.from_numpy(x_arr))
    y_var = Variable(torch.from_numpy(y_arr))
    return x_var, y_var

# SPlitting the data ino train ,valid and test subset
def train_val_test_split(model_data, 
                         train_end = "2006-01-01", 
                         val_start="2009-01-01", 
                         test_start="2012-01-01",
                         seq_len_in=1, seq_len_out=1, input_size=1, 
                         batch_size = 32, device = "cpu"):
    
    q_scaler = StandardScaler()
    df_scaler = StandardScaler()
    
    df_train = model_data[model_data.index < train_end]
    df_valid = model_data[(model_data.index >= val_start)  & (model_data.index < test_start)]
    df_test = model_data[model_data.index >= test_start]
    
    train_scaled = df_scaler.fit_transform(df_train)
    valid_scaled = df_scaler.fit_transform(df_valid)
    test_scaled = df_scaler.fit_transform(df_test)
    
    _ = q_scaler.fit(np.array(df_train.iloc[:,0]).reshape(-1,1))
    
    x_train, y_train = transform_data_multistep(train_scaled, seq_len_in, seq_len_out, input_size)
    x_valid, y_valid = transform_data_multistep(valid_scaled, seq_len_in, seq_len_out, input_size)
    x_test, y_test = transform_data_multistep(test_scaled, seq_len_in, seq_len_out, input_size)
    
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    test_ds = TensorDataset(x_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size, num_workers=1)    
    
    return train_loader, valid_loader, test_loader, q_scaler, df_scaler, train_scaled, valid_scaled, test_scaled


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len_out, device, dropout=0.1):
        super(LSTM, self).__init__()
        # Initializing the model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len_out = seq_len_out
        self.device = device
        # Layer 1: LSTM # batch_size first ()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout=dropout)
        # # Layer 2: Fully coneccted (linear) layer
        self.linear = nn.Linear(self.hidden_size, self.seq_len_out)
    def forward(self, input_seq, prints = False):
        # Reshaping the input_seq
        input_seq = input_seq.view(-1, input_seq.shape[1], self.input_size)
        if prints: print("input_seq shape:", input_seq.shape, "->[num_batches, seq_len, num_features]")     
        # LSTM 
        output, (h_state, c_state) = self.lstm(input_seq)
        if prints: print("LSTM: output shape:" , output.shape, "->[num_batches, seq_len, hidden_size]", 
                         "\n " "LSTM: h_state shape:", h_state.shape, 
                          "->[num_layers*num_directions, num_batches, hidden_size]", "\n"
                          "LSTM: c_state shape:", c_state.shape, 
                          "->[num_layers*num_directions, num_batches, hidden_size]")
        # Reshaping to take last tensor as output
        output = output[:, -1, :]
        if prints: print("LSTM Output reshaped:", output.shape, "->[num_batches, hidden_size]")
        # Fully connected layer
        output = self.linear(output)
        if prints: print("FNN: Final outpu shape:", output.shape, "->[num_batches, num_features]")
        #print ("type of the LSTM output is: ", type(output))
        return output
        
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len_out, device):
        super(RNN, self).__init__()
        # Initializing the model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len_out = seq_len_out
        self.device = device
        # Layer 1: RNN # batch_size first ()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True, dropout=0.1)
        # # Layer 2: Fully coneccted (linear) layer
        self.linear = nn.Linear(self.hidden_size, self.seq_len_out)
    def forward(self, input_seq, prints = False):
        # Reshaping the input_seq
        input_seq = input_seq.view(-1, input_seq.shape[1], self.input_size)
        if prints: print("input_seq shape:", input_seq.shape, "->[num_batches, seq_len, num_features]")        
        # RNN
        output, h_state = self.rnn(input_seq)
        if prints: print("RNN: output shape:" , output.shape, "->[num_batches, seq_len, hidden_size]", 
                         "\n " "RNN: h_state shape:", h_state.shape, 
                          "->[num_layers*num_directions, num_batches, hidden_size]")
        # Reshape 
        output = output[:, -1, :]
        if prints: print("Output reshaped:", output.shape, "->[num_batches, hidden_size]")
        # Fully connected layer
        output = self.linear(output)
        if prints: print("FNN: Final outpu shape:", output.shape, "->[num_batches, num_features]")
        return output
        
def create_model(save_dir, input_size, hidden_size, num_layers, seq_len_out, architecture = "LSTM", device = "cpu", dropout = 0.1):
    if os.path.exists(save_dir):
        if architecture == "LSTM":
            model = LSTM(input_size, hidden_size, num_layers, seq_len_out, device)
        else:
            model = RNN(input_size, hidden_size, num_layers, seq_len_out, device)
        model, prev_epochs, optimizer = model_loader(save_dir)
        return model, prev_epochs, optimizer
    else:
        if architecture == "LSTM":
            model = LSTM(input_size, hidden_size, num_layers, seq_len_out, device)
        else:
            model = RNN(input_size, hidden_size, num_layers, seq_len_out, device)
        prev_epochs = 0
        return model, prev_epochs

def model_validation(model, test_data, criterion, device):
    """
    Validation function, to use in the training function, 
    and later to be used to calculate the metrics on the test dataset. 
    """
    model.to(device)
    test_loss = 0
    for k, (x, y) in enumerate(test_data):
        x, y = x.to(device), y.to(device)
        # getiong the outputs of the network
        out = model(x)
        # Compute the loss
        loss = criterion(out, y)
        test_loss += loss.item()  
    return test_loss

def train_network(model, train_loader, test_loader, prev_epochs = 0, 
                  num_epochs = 5, num_batches = 64, learning_rate = 0.001, 
                  device = "cpu"):
    """Trains the model and compztes the average accuracy for train and tesat data."""
    print("Get data ready...")
    print ("The model was trained for {} epochs, and will be trained for {} new epochs.".format(prev_epochs, num_epochs))
    # get number of new epochs to train for
    new_epochs = prev_epochs + num_epochs
    # Create Criterion and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adamax(model.parameters(), lr = learning_rate)
    print ("Training started...")
    # Train the data multiple times
    print_every = 25
    steps = 0
    loss_vals_train = []
    loss_vals_test = []
    epoch = 0
    for epoch in progress_bar(range(prev_epochs, new_epochs)):
        train_loss = 0
        test_loss = 0
        print ("Epoch {}/{}".format(epoch+1, new_epochs))
        print ("-" * 20)
        # Set the model to training mode
        model.train()
        model.to(device)        
        for k, (x, y) in enumerate(train_loader): # k je iteracija -- > if k % 1 == 0: print ....
            x, y = x.to(device), y.to(device)
            steps += 1
            # getion the outputs of the network
            out = model(x)
            # Clear the gradients from the prviuos iteration
            optimizer.zero_grad() 
            # Compute the loss
            loss = criterion(out, y)
            # Compute the gradients for the neurons
            loss.backward()
            # Save Loss after each iteration
            train_loss += loss.item()
            # Update the weights
            optimizer.step()
            # Print Loss per training epoch   
            
            if steps % print_every == 0:
                print("TRAIN | MSE: {:.3f} | RMSE: {:.3f} | k={}".format(train_loss/(k+1), sqrt(train_loss/(k+1)), k+1))
                
                test_loss = model_validation(model, test_loader, criterion, device)
               
                print("TEST  | MSE: {:.3f} | RMSE: {:.3f} | k={}".format(test_loss/(k+1), sqrt(test_loss/(k+1)), k+1))
        steps = 0
        loss_vals_train.append(train_loss/len(train_loader))
        loss_vals_test.append(test_loss/len(test_loader))
    plot_loss(range(prev_epochs, new_epochs), loss_vals_train, loss_vals_test)
    checkpoint = {"model": model,
                  "epoch": epoch+1,
                  "model_state": model.state_dict(),
                  "optimizer_state": optimizer.state_dict(),
                 }
    
    return checkpoint

def save_model(checkpoint, save_dir):
    """
    Saving: model, optimizer, epochs, state dict
    """
    print ("Model has been saved to {}.".format(save_dir))
    return torch.save(checkpoint, save_dir)

def model_loader(file_pth, device = "cpu"):
    """
    The function loads a checkpoint from the model. Either to continue
    training, or for inference.
    """
    checkpoint = torch.load(file_pth)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["model_state"])
    optimizer = checkpoint["optimizer_state"]
    prev_epochs = checkpoint["epoch"]
    
    return model, prev_epochs, optimizer

def get_metrics(df):
    """
    Parameters
    ----------
    df - results dataframe from the model
    Returns
    -------
    metrics : dictionary - metrics values - mse, rmse, nse, r2, mae
    """
    pred = df["predicted"]
    obs = df["observed"]
    if len(pred) == len (obs):
        nse = 1-(np.sum(np.square(obs - pred)/np.sum(np.square(obs - np.mean(obs)))))
        mse = 1/len(pred)*np.sum(np.square(obs-pred))
        rmse = np.sqrt(mse)
        ss_res=np.square(np.sum((obs-np.mean(obs))*(pred-np.mean(pred))))
        ss_tot=np.sum(np.square(obs-np.mean(obs)))*np.sum(np.square(pred-np.mean(pred)))        
        r2 = ss_res/ss_tot
        mae = 1/len(pred)*np.sum(np.absolute(obs-pred))
        metrics = {"MSE": mse, "RMSE":rmse, "NSE":nse, "R2":r2, "MAE":mae}
        return metrics
    
    
def get_metrics_sep(df, train_end = "2006-01-01", 
                         val_start="2009-01-01", 
                         test_start="2012-01-01",):
    """
    Parameters
    ----------
    predicted : series - calculated values
    observed : series - observed (measured) values
    Returns
    -------
    metrics : dictionary - metrics values
    """
    df_train = df.loc[:train_end]
    df_valid = df.loc[val_start:test_start]
    df_test = df.loc[test_start:]
    
    metrics_train = get_metrics(df_train)
    metrics_valid = get_metrics(df_valid)
    metrics_test = get_metrics(df_test)
    metrics_sep_dict = {"Train_metrics": metrics_train,
                        "Valid_metrics": metrics_valid,
                        "Test_metrics": metrics_test}
    return metrics_sep_dict
    


def make_prediction(input_data, observed_data, model_data, seq_len_in, model, q_scaler, model_number, train_end):
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        predicted = model(input_data).cpu().detach().numpy()
    # converting tensor to arrays
    predicted = q_scaler.inverse_transform(predicted[:,0])
    observed = q_scaler.inverse_transform(observed_data.numpy()[:,0])      
    #starting_date = model_data.index[0]+datetime.timedelta(days=seq_len_in)
    dates = pd.date_range(start=model_data.index[0]+datetime.timedelta(days=seq_len_in+1), 
                          periods=len(model_data)-(seq_len_in+1))
    results_df = pd.DataFrame({"observed":observed, "predicted":predicted}, columns=["observed", "predicted"], dtype=np.float32)
    results_df.index = dates
    # plot_timeseries(results_df, model_number, start_date=dates[0], end_date=datetime.datetime.strptime(train_end, "%Y-%m-%d"))
    return results_df































