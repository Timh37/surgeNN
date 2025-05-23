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
    predictors['w'] = np.sqrt(predictors['u10']**2+predictors['v10']**2) #compute wind speed from x/y components
    return predictors