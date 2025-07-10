#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:21:19 2023

@author: timhermans
"""

import numpy as np
import xarray as xr
import os
import pandas as pd
import fnmatch

def compute_distances_to_point(lon,lat,lons,lats):
    
    distances = np.zeros(len(lons)) #initialize array
    
    distances = distances + 2*np.arcsin( np.sqrt(
            np.sin( (np.pi/180) * 0.5*(lats-lat) )**2 +
            np.cos((np.pi/180)*lat)*np.cos((np.pi/180)*lats)*np.sin((np.pi/180)*0.5*(lons-lon))**2) )
    
    return distances*6371

### paths
in_dir = '/Volumes/Naamloos/PhD_Data/GTSM_HighResMIP/HadGEM3-GC31-HM/surge/stretched_3hourly_rounded10min_after/' #'/Volumes/Naamloos/PhD_Data/CoDEC/surge/era5/'
out_dir = '/Volumes/Naamloos/PhD_Data/GTSM_HighResMIP/HadGEM3-GC31-HM/surge_at_gesla3/stretched_3hourly_rounded10min_after/' #'/Volumes/Naamloos/PhD_Data/CoDEC/surge/era5_at_gesla3_tgs_eu'

tg_coords = xr.open_dataset('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/input/gesla3_tg_coordinates_eu.nc')

my_files = fnmatch.filter(os.listdir(in_dir),'*nc')

ff = 0
for f,file in enumerate(my_files):
    print(file)
    if file in os.listdir(out_dir):
        continue
    surge = xr.open_mfdataset(os.path.join(in_dir,file))
   
    if ff==0:
        max_lon = ((surge.station_x_coordinate.values + 180) % 360 - 180).max()
        min_lon = ((surge.station_x_coordinate.values + 180) % 360 - 180).min()
        max_lat = (surge.station_y_coordinate.values).max()
        min_lat = (surge.station_y_coordinate.values).min()
        tg_coords = tg_coords.where( (tg_coords.lon<max_lon) & (tg_coords.lon>min_lon) & (tg_coords.lat<max_lat) & (tg_coords.lat>min_lat),drop=True) #because codec only covers europe
        iNearest = np.zeros(len(tg_coords.tg))
        min_dists = np.zeros(len(tg_coords.tg))

    if ff==0:
        for t,site in enumerate(tg_coords.tg):
        #for t,tg in enumerate(tg_coords.tg):    
            dists = compute_distances_to_point(tg_coords.lon.sel(tg=site),tg_coords.lat.sel(tg=site),surge.station_x_coordinate,surge.station_y_coordinate)
            iNearest[t] = np.argmin(dists.values)
            min_dists[t] = np.min(dists.values)
    surge_at_tgs = surge.isel(stations=iNearest.astype('int'))
    ff=1
    
    surge_at_tgs = surge_at_tgs.rename_dims({'stations':'tg'})
    surge_at_tgs['tg'] = tg_coords['tg'].values
    surge_at_tgs.to_netcdf(os.path.join(out_dir,file),mode='w')
    


