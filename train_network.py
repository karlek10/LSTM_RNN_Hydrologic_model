# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:51:53 2021

@author: karlo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import utility_functions as uf
import os


# ====== MODEL STATICS ======
input_size = 23                 # number of features (stations) --> rain, temp, hum, etc. 
hidden_size = 120              # number of hidden neurons
num_layers = 2                  # number of LSTM layers
seq_len_in = 120               # length of the training time series
seq_len_out = 1                # number of output step for predicted runoff
train_end = "2013-01-01"        # end date of training period
val_start = "2013-01-01"        # start date of validation period
test_start = "2014-01-01"       # testing period start date
basin_name = "Drava"            # basin name (folder name with data)
network_architecture = "LSTM"   # network architecture - LSTM or RNN supported right now
# =====================

# ====== HYPERPARAMETERS ======
dropout = 0.4                  # dropout on LSTM layers
num_batches = 32                # number of batches
num_epochs = 30                 # number of new epochs to train for
learning_rate = 0.00035          # learning rate
device = "cuda"                 # select the device to train on "cuda" or "cpu"
model_number = "Drava_model_3"        # model number
resume_training = True
# ============================





if __name__ == "__main__":
    
    save_dir = model_number + "_lay-"+ str(num_layers)+ "_" + str(seq_len_in) +"_drp-" + str(dropout) +"_hidd-"+ str(hidden_size) + "_ep-"+ str(num_epochs)+".pth"    
    
    model_data = uf.load_data(model_number)
        
    train_loader, valid_loader, test_loader, q_scaler, df_scaler, train_scaled, valid_scaled, test_scaled = uf.train_val_test_split(
        model_data, train_end,  val_start, test_start, seq_len_in, seq_len_out, input_size)  
    
    
    # Dotjerati ovaj resume training da ne zeza kod otpakiravanja novog modela !!!!!
    if resume_training:
        model, prev_epochs = uf.create_model(save_dir, input_size, hidden_size, num_layers, 
                                             seq_len_out, network_architecture, device, dropout)
    else:
        model, prev_epochs = uf.create_model(save_dir, input_size, hidden_size, num_layers, 
                                     seq_len_out, network_architecture, device, dropout)

    checkpoint =  uf.train_network(model, train_loader, valid_loader, prev_epochs=prev_epochs, num_epochs=num_epochs, 
                                num_batches= num_batches, learning_rate = learning_rate, device = device)
    
    uf.save_model(checkpoint, save_dir)





