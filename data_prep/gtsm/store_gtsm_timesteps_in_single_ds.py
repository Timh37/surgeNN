#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:08:23 2023

@author: timhermans
"""

import xarray as xr
import numpy as np
import os

gtsm_type = 'GTSM_HighResMIP_HadGEM3-GC31-HM'#'CoDEC_ERA5'#'GTSM_HighResMIP_HadGEM3_GC31_HM' 
tg_coords = xr.open_dataset('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/input/gesla3_tg_coordinates_eu.nc')

#gtsm_data = xr.open_mfdataset('/Volumes/Naamloos/PhD_Data/CoDEC/era5/surge_at_gesla3/10min/*nc') #--> hourly means but better to use hourly to be consistent across tgs? (and predictors are also (3/6)hourly)
#gtsm_data = xr.open_mfdataset('/Volumes/Naamloos/PhD_Data/GTSM_HighResMIP/HadGEM3-GC31-HM/surge_at_gesla3/hourly/*nc')
gtsm_data = xr.open_mfdataset('/Volumes/Naamloos/PhD_Data/GTSM_HighResMIP/HadGEM3-GC31-HM/surge_at_gesla3/stretched_3hourly_rounded10min_after/*nc')
timesteps = xr.open_dataset('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/input/GTSM/stretched_3hourly_360day_HadGEM3_rounded10min_after.nc')

out_dir = '/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/input/'
try:
    gtsm_data = gtsm_data.rename({'stations':'tg'})
    gtsm_data = gtsm_data.assign_coords({'tg':tg_coords.tg})

except:
    try:
        gtsm_data = gtsm_data.rename({'site':'tg'})
        gtsm_data = gtsm_data.assign_coords({'tg':tg_coords.tg})

    except:
        pass

gtsm_data=gtsm_data.reindex_like(timesteps.time) #fill up timesteps missing in GTSM data #.load()

gtsm_data_anoms = gtsm_data.copy(deep=True) #remove annual means
gtsm_data_anoms['surge'] = gtsm_data_anoms['surge'].groupby(gtsm_data_anoms.time.dt.year) - gtsm_data_anoms['surge'].groupby(gtsm_data_anoms.time.dt.year).mean('time')

#gtsm_data_anoms = gtsm_data_anoms.where(gtsm_data_anoms.time.dt.minute == 0,drop=True) #subsample hourly (if needed)

gtsm_data_anoms.to_netcdf(os.path.join(out_dir,gtsm_type+'_at_gesla3_tgs_stretched_3hourly_rounded10min_after.nc'),mode='w')
#gtsm_data_anoms.to_netcdf('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/CoDEC_ERA5_at_gesla3_tgs_eu_hourly_anoms.nc',mode='w')
