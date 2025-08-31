import numpy as np
import pandas as pd
import setproctitle
import copy
import multiprocessing
import datetime
import warnings
from UniBIP.utils import *
from UniBIP.preprocess import Config, BioGraphData
from UniBIP.trian_model import training_params
# Set the start method for multiprocessing to 'spawn', which is safe for use with CUDA
multiprocessing.set_start_method('spawn')
# Set the process title to identify this process easily in the system
setproctitle.setproctitle("DDI01_UniBIP")
from pandas import DataFrame
import sqlite3




if __name__ == '__main__':
    # Define dataset information and corresponding hyperparameters
    config = Config(gpu_id=3,epochs=200, print_epoch = 10, repeats=1)
    config.val_size = 0.1
    config.True_edge_matrix = True
    base_seed = 2040
    Gdata = BioGraphData(mode='H',config=config)
    Gdata.load_edge(file_path='./dataset/rename_edgelist.txt', sep=' ')
    Gdata.load_I2H_features_matrix(file_path='./dataset/combined_features.csv',header=0,index_col=0, sep=',')
    config.set_random(base_seed)
    model, Gdata_predict, best_result_df, results_df = training_params(Gdata.copy(), config.copy())
    print("All tasks completed.")  # Log the completion of all tasks

