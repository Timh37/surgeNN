#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:36:46 2022

@author: timhermans
"""

#import packages
import numpy as np
import os
import xarray as xr

def get_era5_around_tgs(era5_ds,grid_size,tg_coords,output_coord_arrays=False):
    
    resolution = float(era5_ds.resolution.replace('p','.').split('x')[0])
    
    #get grid coordinates around point
    grid_lats = np.zeros((len(tg_coords.tg),int(grid_size/resolution)))
    grid_lons = np.zeros((len(tg_coords.tg),int(grid_size/resolution)))
    
    for t,tg in enumerate(tg_coords.tg.values):
        grid_lats[t,:] = era5_ds.latitude[((era5_ds.latitude>=(tg_coords.sel(tg=tg).lat-grid_size/2)) & (era5_ds.latitude<=(tg_coords.sel(tg=tg).lat+grid_size/2)))][0:int(grid_size/resolution)]
        grid_lons[t,:] = era5_ds.longitude[((era5_ds.longitude>=(tg_coords.sel(tg=tg).lon-grid_size/2)) & (era5_ds.longitude<=(tg_coords.sel(tg=tg).lon+grid_size/2)))][0:int(grid_size/resolution)]
    
    #create da's to index ERA5 with
    lons_da = xr.DataArray(grid_lons,dims=['tg','lon_around_tg'],coords={'tg':tg_coords.tg,'lon_around_tg':np.arange(0,int(grid_size/resolution))})
    lats_da = xr.DataArray(grid_lats,dims=['tg','lat_around_tg'],coords={'tg':tg_coords.tg,'lat_around_tg':np.arange(0,int(grid_size/resolution))})
    
    era5_around_tgs = era5_ds.sel(latitude=lats_da,longitude=lons_da)

    if output_coord_arrays:
        return era5_around_tgs, lons_da, lats_da
    else:
        return era5_around_tgs
