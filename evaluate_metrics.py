# -*- coding: utf-8 -*-
"""
Created on Thu May 20 09:16:00 2021

@author: karlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set()

# ====== DEFINE PARAMETERS ======

select_metrics = ["R2","MSE"]         # provide desired (MSE, RMSE, NSE, R2, MAE) 
                                       # metrics to plot as list




# ===============================


def read_dfs(filename, select_metrics):
    df = pd.read_csv(filename)
    df = df.rename(columns = {"Unnamed: 0": "Period"})
    df["Period"] = df["Period"].str.split("_").str[0]
    df.index = df["Period"]
    df = df.drop(["Period"], axis=1)
    df = df [select_metrics]
    return df


def plot_metrics_bar(df, metric, transparency = True):
    fig, ax = plt.subplots(figsize=(10,8), dpi=600)
    df.plot.bar(ax = plt.gca())
    ax.set_ylim(df.to_numpy().min()*0.95, df.to_numpy().max()*1.05)
    ax.xaxis.set_tick_params(which='major', pad=7, labelsize=22, rotation=0)
    ax.yaxis.set_tick_params(which='major', pad=7, labelsize=22, rotation=0)
    ax.set_title(metric.upper(), fontsize=28)
    ax.legend(fontsize=22)
  
    fig.savefig(metric+".png", transparent=transparency)
    print ("Graphs have been saved to result_mertrics folder.")
    
    
    
    
# df = read_dfs(filename)    
    
    
if __name__ == "__main__":
    curr_dir = os.getcwd()
    os.chdir(curr_dir+ "/results_metrics/")
    dict_names_files = dict()

    for file in os.listdir(curr_dir+ "/results_metrics/"):
        if file.endswith(".csv"):
            name = file.split("_lay", 1)
            dict_names_files.update({name[0]:file})
    
    dict_files = {name:read_dfs(file, select_metrics) for name, file in dict_names_files.items()} 
    
    df = pd.concat(dict_files)[select_metrics]

    for metric in df.columns:
        plot_metrics_bar(df[metric].unstack(), metric, transparency=False)
        