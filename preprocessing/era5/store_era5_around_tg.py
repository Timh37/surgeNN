#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:33:17 2024

@author: timhermans
"""
import xarray as xr
import os
import pandas as pd
import numpy as np


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
    
#tgs = ['barcelona-bar-esp-cmems.csv','den_helder-denhdr-nld-rws.csv', 'brest-822a-fra-uhslc.csv', 
#            'immingham-imm-gbr-bodc.csv','stavanger-svg-nor-nhs.csv']

tgs = ['den_helder-denhdr-nld-rws.csv']
grid_size_around_tgs=5
freq = 3 #'h'

in_dir = '/Volumes/Naamloos/PhD_Data/era5/3hourly_wind_msl'
out_dir = '/Volumes/Naamloos/PhD_Data/era5/predictors_'+str(freq)+'hourly_around_gesla3'

#open era5
era5 = xr.open_mfdataset([os.path.join(in_dir,x) for x in os.listdir(in_dir) if '.nc' in x],concat_dim="time", combine="nested",
                  data_vars='minimal', coords='minimal', compat='override')#.chunk({'time':100000})#,chunks={'longitude':30,'latitude':30}).chunk({'time':100000})
era5.attrs['resolution'] = '0p25x0p25'

era5 = era5.where(era5.time.dt.hour%freq == 0,drop=True) #subsample

tg_coords = xr.open_dataset('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/input/gesla3_tg_coordinates_eu.nc')

for tg in tgs:#tg_coords.tg.values:
    print('processing: '+tg)
    era5_around_tgs = get_era5_around_tgs(era5,grid_size_around_tgs,tg_coords.sel(tg=[tg])) #get predictors around each TG


    output_fn = tg.replace('.csv','_era5Predictors_'+str(grid_size_around_tgs)+'x'+str(grid_size_around_tgs)+'.nc')
    
    #if output_tg.replace('.csv','_era5Predictors_'+str(grid_size_around_tgs)+'x'+str(grid_size_around_tgs)+'.nc') not in os.listdir('predictors/'):
        #da_around_tgs.sel(tg=output_tg).to_netcdf('predictors/'+output_tg.replace('.csv','_era5Predictors_'+str(grid_size_around_tgs)+'x'+str(grid_size_around_tgs)+'.nc'))
    #da_around_tgs.sel(tg=output_tg).to_zarr(os.path.join('gs://leap-persistent/timh37/era5/6hourly/5x5_around_gesla3',output_fn),mode='w')
    era5_around_tgs.sel(tg=tg).to_netcdf(os.path.join(out_dir,output_fn.replace('.zarr','.nc')),mode='w')
