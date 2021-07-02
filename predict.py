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
input_size = 11                # number of features (stations) --> rain, temp, hum, etc. 
hidden_size = 30                # number of hidden neurons
num_layers = 2                  # number of LSTM layers
seq_len_in = 30               # length of the training time series
seq_len_out = 1                 # number of output step for predicted runoff
train_end = "2013-01-01"        # end date of training period
val_start = "2013-01-01"        # start date of validation period
test_start = "2014-01-01"       # testing period start date
basin_name = "Drava"            # basin name (folder name with data)
save_dir = "Drava_model_1_lay-2_30_drp-0.15_hidd-30_ep-30.pth"      # filename of the saved model
model_number = "Drava_model_1"  # Model number
# =====================


if __name__ == "__main__":
    
    model_data = uf.load_data(model_number)
    
    train_loader, valid_loader, test_loader, q_scaler, df_scaler, train_scaled, valid_scaled, test_scaled = uf.train_val_test_split(
        model_data, train_end,  val_start, test_start, seq_len_in, seq_len_out, input_size) 
    
    
    model, prev_epochs, optimizer = uf.create_model(save_dir, input_size, hidden_size, num_layers, seq_len_out, device="cpu")
    
    eval_data = np.concatenate([train_scaled, valid_scaled, test_scaled], axis=0)
    
    eval_tr, eval_te = uf.transform_data_multistep(eval_data, seq_len_in, seq_len_out, input_size)
    
    results_df = uf.make_prediction(eval_tr, eval_te, model_data, seq_len_in, model, q_scaler, model_number, train_end)
    
    uf.plot_timeseries(results_df, model_number, end_date="2012-12-31", label = "Training")
    
    uf.plot_timeseries(results_df, model_number, start_date="2013-01-01", end_date="2014-12-31", label = "Validation - Testing")

    metrics_df = pd.DataFrame.from_dict(uf.get_metrics_sep(results_df, train_end, val_start, test_start), orient="index")
    
    metrics_df.to_csv("results_metrics/" + save_dir[:-4]+".csv")
    