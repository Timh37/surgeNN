#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:46:41 2023

@author: timhermans
"""

import numpy as np
import sys
sys.path.append('/Users/timhermans/Documents/GitHub/projectESL/projectESL')
from preprocessing import extract_GESLA3_locations, ingest_GESLA3_files


from copy import deepcopy
import os
import pandas as pd

def subtract_ameans(df):
    annual_means_at_timesteps = df.groupby(df.date.dt.year).transform('mean')['surge'].astype('float64')
    df['surge'] = df['surge'] - annual_means_at_timesteps
    return df

def compute_distances_to_point(lon,lat,lons,lats):
    
    distances = np.zeros(len(lons)) #initialize array
    
    distances = distances + 2*np.arcsin( np.sqrt(
            np.sin( (np.pi/180) * 0.5*(lats-lat) )**2 +
            np.cos((np.pi/180)*lat)*np.cos((np.pi/180)*lats)*np.sin((np.pi/180)*0.5*(lons-lon))**2) )
    
    return distances*6371

sites_to_drop = ['donges-396-fra-refmar','cordemais_60minute-cor-fra-cmems','donges_60minute-don-fra-cmems','lepellerin_60minute-lep-fra-cmems','cordemais-397-fra-refmar','le_pellerin-398-fra-refmar',
            'nantes_usine_brulee-399-fra-refmar','alteweser-alt-deu-cmems','bake_a-3611-deu-wsv','bake_c_scharhorn-3905-deu-wsv','nantesusinebrulee_60minute-nan-fra-cmems',
            'grober_vogelsand-82-deu-wsv','dwarsgat-4100-deu-wsv','robbensudsteert-4110-deu-wsv','nieuwestatenzijl-nie-nld-cmems','oterdum-otdm-nld-rws',
            'oostmahorn-oostmhn-nld-rws','maassluis-maasss-nld-rws','bhv_alter_leuchtturm-4098-deu-wsv',
            'nieuwe_statenzijl-nieuwstzl-nld-rws','hellevoetsluis-hellvss-nld-rws','noordwijk_meetpost-noordwmpt-nld-rws','leuchtturm_alte_wese-4094-deu-wsv']


GESLA3_locations = extract_GESLA3_locations('/Volumes/Naamloos/PhD_Data/GESLA3',20)

area_lon = [-30,10]
area_lat = [36,62]

idx_eu = np.where((np.array(GESLA3_locations[2])>area_lon[0]) & (np.array(GESLA3_locations[2])<area_lon[-1]) & (np.array(GESLA3_locations[1])>area_lat[0]) & (np.array(GESLA3_locations[1])<area_lat[-1]))[0]

preproc_settings = { 'min_yrs': 20,                                                                 # [int/float] minimum tide gauge length
 'resample_freq': 'D_max'      ,                                                # ['H_mean', 'D_mean','D_max']
 'deseasonalize': 1     ,                                                       # [0,1] whether to deseasonalize raw data prior to estimating ESL distribution
 'detrend':0          ,                                                        # [0,1] whether to detrend raw data prior to estimating ESL distribution
 'subtract_amean': 1   ,                                                        # [0,1] whether to subtract annual means of raw data prior to estimating ESL distribution
 'ref_to_msl': 0   ,                                                            # [0,1] whether to reference observations to mean sea level during defined period
                                              # reference period to compute MSL for, if referencing observations to MSL (TG data must cover at least half of this period)
 'declus_method': 'rolling_max' ,                                               # ['iterative_descending','rolling_max'] how to decluster the extremes
 'declus_window': 3,                                                            # [int] declustering window length to use
 'extremes_threshold': 99,                                                      # [0<float<100] threshold above which to define events at resamlpe_freq frequency extreme
 'store_esls': 0      }
dmax = ingest_GESLA3_files('/Volumes/Naamloos/PhD_Data/GESLA3',preproc_settings,fns=list(np.array(GESLA3_locations[3])[idx_eu]))

#hmean_anoms = subtract_amean_from_gesla_dfs(hmeans)
dmax_ = deepcopy(dmax)
nyears = []
for k,v in dmax.items():
    if (np.sum(((v.index>='1980' ) & (v.index<='2023')))/365.25)<20: #how many years of daily maxima between 1980 and 2023
        del dmax_[k] #if less than 20, remove from dictionary
    if k in sites_to_drop: #drop offshore sites and sites far into river outlets
        del dmax_[k]
#generate list of coordinates for included sites
lons = {}
lats = {}
for k in dmax_.keys():
    idx = np.where(np.array(GESLA3_locations[3])==k)[0][0]
    lons[k] = np.array(GESLA3_locations[2])[idx]
    lats[k] = np.array(GESLA3_locations[1])[idx]

#exclude sites if too close to already included sites
dmax_filtered = {}
lons_filtered = []
lats_filtered = []

i=0
for k,v in dmax_.items():
    if len(dmax_filtered)==0:
        dmax_filtered[k] = v
        lons_filtered.append(lons[k])
        lats_filtered.append(lats[k])
    else:
        dists = compute_distances_to_point(lons[k],lats[k],np.array(lons_filtered),np.array(lats_filtered))
        if (dists<3).any():
            continue
        else:
            dmax_filtered[k] = v
            lons_filtered.append(lons[k])
            lats_filtered.append(lats[k])   
    i+=1
        
#make a map to check
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig=plt.figure(figsize=(9.5,4)) #generate figure  
gs = fig.add_gridspec(1,1)
ax = plt.subplot(gs[0,0],projection=ccrs.Robinson(central_longitude=0))
ax.coastlines(zorder=5)
ax.set_extent([-24, 13, 35, 65], crs=ccrs.PlateCarree())
ax.scatter(lons_filtered,lats_filtered,transform=ccrs.PlateCarree())

#for selected sites, open hourly means, compute anomalies relative to annual means, and store to csv files to be detided
preproc_settings = { 'min_yrs': 20,                                                                 # [int/float] minimum tide gauge length
 'resample_freq': 'H_mean'      ,                                                # ['H_mean', 'D_mean','D_max']
 'deseasonalize': 1     ,                                                       # [0,1] whether to deseasonalize raw data prior to estimating ESL distribution
 'detrend':0          ,                                                        # [0,1] whether to detrend raw data prior to estimating ESL distribution
 'subtract_amean': 1   ,                                                        # [0,1] whether to subtract annual means of raw data prior to estimating ESL distribution
 'ref_to_msl': 0   ,                                                            # [0,1] whether to reference observations to mean sea level during defined period
                                              # reference period to compute MSL for, if referencing observations to MSL (TG data must cover at least half of this period)
 'declus_method': 'rolling_max' ,                                               # ['iterative_descending','rolling_max'] how to decluster the extremes
 'declus_window': 3,                                                            # [int] declustering window length to use
 'extremes_threshold': 99,                                                      # [0<float<100] threshold above which to define events at resamlpe_freq frequency extreme
 'store_esls': 0      }

h_means = ingest_GESLA3_files('/Volumes/Naamloos/PhD_Data/GESLA3',preproc_settings,fns=list(dmax_filtered.keys()))

preproc_settings = { 'min_yrs': 20,                                                                 # [int/float] minimum tide gauge length
 'resample_freq': 'H'      ,                                                # ['H_mean', 'D_mean','D_max']
 'deseasonalize': 1     ,                                                       # [0,1] whether to deseasonalize raw data prior to estimating ESL distribution
 'detrend':0          ,                                                        # [0,1] whether to detrend raw data prior to estimating ESL distribution
 'subtract_amean': 1   ,                                                        # [0,1] whether to subtract annual means of raw data prior to estimating ESL distribution
 'ref_to_msl': 0   ,                                                            # [0,1] whether to reference observations to mean sea level during defined period
                                              # reference period to compute MSL for, if referencing observations to MSL (TG data must cover at least half of this period)
 'declus_method': 'rolling_max' ,                                               # ['iterative_descending','rolling_max'] how to decluster the extremes
 'declus_window': 3,                                                            # [int] declustering window length to use
 'extremes_threshold': 99,                                                      # [0<float<100] threshold above which to define events at resamlpe_freq frequency extreme
 'store_esls': 0      }

hourly = ingest_GESLA3_files('/Volumes/Naamloos/PhD_Data/GESLA3',preproc_settings,fns=list(dmax_filtered.keys()))
'''
hourly = ingest_GESLA3_files('/Volumes/Naamloos/PhD_Data/GESLA3',preproc_settings,fns=['alicante_i-ali-esp-da_mm'])
lats_filtered=[38.33827100]
lons_filtered=[-0.47787700]
'''

'''
i=0
for k,v in h_means.items():
    full_date_range = pd.date_range(str(v.index[0].year), str(v.index[-1].year+1), freq='h')[0:-1]
    v = v.reindex(full_date_range)
    v = v.reset_index()
    v = v.rename(columns={'index':'date','sea_level':'surge'})
    
    v['lat'] = lats_filtered[i]
    v['lon'] = lons_filtered[i]
    
    v = subtract_ameans(v)
    
    v.to_csv(os.path.join('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/detide_GESLA3/GESLA3_hourly_means/',k+'.csv'))
    i+=1
    
'''
i=0
for k,v in hourly.items():
    full_date_range = pd.date_range(str(v.index[0].year), str(v.index[-1].year+1), freq='h')[0:-1]
    v = v.reindex(full_date_range)
    v = v.reset_index()
    v = v.rename(columns={'index':'date','sea_level':'surge'})
    
    v['lat'] = lats_filtered[i]
    v['lon'] = lons_filtered[i]
    
    v = subtract_ameans(v)
    
    v.to_csv(os.path.join('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/detide_GESLA3/GESLA3_hourly/',k+'.csv'))
    i+=1

