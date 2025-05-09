#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input output functions
Created on Wed Jan 31 12:00:43 2024

@author: timhermans
"""
import xarray as xr
import pandas as pd
import numpy as np
import os


def load_predictand(input_dir,tg): #open csv files with predictands for a tide gauge
    '''
    input_dir:   directory where predictands are stored
    tg:          name of tide gauge to open predictands for
    '''
    predictand = pd.read_csv(os.path.join(input_dir,tg))
    predictand['date'] = pd.to_datetime(predictand['date'])
    return predictand

def load_predictors(input_dir,tg): #open ERA5 wind and pressure predictors around a tide gauge (currently 5x5 degree input by default)
    '''
    input_dir:   directory where predictors are stored
    tg:          name of tide gauge to open predictors for
    '''
    
    if input_dir.startswith('gs://'):
        predictors = xr.open_dataset(os.path.join(input_dir,tg.replace('.csv','_era5Predictors_5x5.nc')),engine='zarr')
    else:
        predictors = xr.open_dataset(os.path.join(input_dir,tg.replace('.csv','_era5Predictors_5x5.nc')))
    if 'w' not in predictors.variables:
        predictors['w'] = np.sqrt(predictors['u10']**2+predictors['v10']**2) #compute wind speed from x/y components
    return predictors

def train_predict_output_to_ds(o,yhat,t,hyperparam_settings,model_architecture,lf_name):
    return xr.Dataset(data_vars=dict(o=(["time"], o),yhat=(["time"], yhat),hyperparameters=(['p'],list(hyperparam_settings)),),
            coords=dict(time=t,p=['batch_size', 'n_steps', 'n_convlstm', 'n_convlstm_units','n_dense', 'n_dense_units', 'dropout', 'lr', 'l2','dl_alpha'],),
            attrs=dict(description=model_architecture+" - neural network prediction performance.",loss_function=lf_name),)

def setup_output_dirs(output_dir,store_model,model_architecture):
    performance_dir = os.path.join(output_dir,'performance',model_architecture)
    model_dir = os.path.join(output_dir,'keras_models',model_architecture)
    if os.path.exists(performance_dir)==False:
        os.makedirs(performance_dir)
    
    if os.path.exists(model_dir)==False and store_model==True:
        os.makedirs(model_dir)