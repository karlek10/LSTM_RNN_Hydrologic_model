# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:35:30 2021

@author: karlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import utility_functions as uf
import os
import datetime



# ====== MODEL STATICS ======
input_size = 11                 # number of features (stations) --> rain, temp, hum, etc. 
hidden_size = 30                # number of hidden neurons
num_layers = 2                  # number of LSTM layers
seq_len_in = 150                # length of the training time series
seq_len_out = 1                # number of output step for predicted runoff
train_end = "2013-01-01"        # end date of training period
val_start = "2013-01-01"        # start date of validation period
test_start = "2014-01-01"       # testing period start date
basin_name = "Drava"            # basin name (folder name with data)
# =====================


if __name__ == "__main__":
    
    save_dir = basin_name + "_" + str(seq_len_in) + ".pth"
    
    model_data = uf.load_data(basin_name="Drava")
    
    train_loader, valid_loader, test_loader, q_scaler, df_scaler, train_scaled, valid_scaled, test_scaled = uf.train_val_test_split(
        model_data, train_end,  val_start, test_start, seq_len_in, seq_len_out, input_size) 
    
    
    model, prev_epochs = uf.create_model(save_dir, input_size, hidden_size, num_layers, seq_len_out, device="cpu")
    
    eval_data = np.concatenate([train_scaled, valid_scaled, test_scaled], axis=0)
    
    eval_tr, eval_te = uf.transform_data_multistep(eval_data, seq_len_in, seq_len_out, input_size)
    
    results_df = uf.make_prediction(eval_tr, eval_te, model_data, seq_len_in, model, q_scaler)
    
    uf.plot_timeseries(results_df, start_date="2010-01-01", end_date="2010-12-31")
    
    