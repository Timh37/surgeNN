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
import fnmatch
import gcsfs

fs = gcsfs.GCSFileSystem() #list stores, stripp zarr from filename, load 

class Predictand():
    def __init__(self, path):
        self.path = path
        
    def open_dataset(self,tg):
        self.data = open_predictand(self.path,tg) #open tide gauge predictand file
        
    def trim_dates(self,date0,date1): #select predictand data at and between dates 0 and 1
        self.data = self.data[(self.data['date']>=date0) & (self.data['date']<=date1)]

    def resample_fillna(self,freq): #add NaNs at missing timesteps every freq (e.g., '3h')
        resampled = self.data.set_index('date').resample(freq).fillna(method=None)
        self.data = resampled.reset_index()[['surge','date','lon','lat']]   
      
    def deseasonalize(self): #remove mean annual cycle
        monthly_means_at_timesteps = self.data.groupby(self.data['date'].dt.month).transform('mean')['surge'].astype('float64')
        self.data['surge'] = self.data['surge'] - monthly_means_at_timesteps + np.mean(monthly_means_at_timesteps)
    
    def subtract_ameans(self): #subtract annual means at each timestep
        df = self.data
        annual_means_at_timesteps = df.groupby(df.date.dt.year).transform('mean')['surge'].astype('float64')
        df['surge'] = df['surge'] - annual_means_at_timesteps
        self.data = df
    
    def rolling_mean(self,window_len,temp_freq): #apply rolling mean
        #crude way to filter out peaks due to uncorrected tides (Tiggeloven et al., 2021) (window_len = 12 #(hours))
        self.data['surge'] = self.data['surge'].rolling(window=int(window_len/temp_freq+1),min_periods=int(window_len/temp_freq+1),center=True).mean() 
        
class Predictor():
    def __init__(self, path):
        self.path = path
        
    def open_dataset(self,tg,predictor_vars,n_cells): #open predictor dataset
        self.data = open_predictors(self.path,tg,n_cells)[predictor_vars]
      
    def subtract_annual_means(self): #subtract annual means at each timestep
        for var in list(self.data.keys()): #for each data variable
            self.data[var] = self.data[var].groupby(self.data.time.dt.year) -self.data[var].groupby(self.data.time.dt.year).mean('time') #remove annual means
   
    def deseasonalize(self): #subtract mean annual cycle
        for var in list(self.data.keys()):
            da_minus_aslc = self.data[var].groupby(self.data.time.dt.month) - self.data[var].groupby(self.data.time.dt.month).mean('time')
            self.data[var] = da_minus_aslc + (self.data[var].mean(dim='time') - da_minus_aslc.mean(dim='time'))
            
    def trim_years(self,year0,year1): #select between years
        self.data = self.data.sel(time=slice(str(year0),str(year1)))


def open_predictors(path,tg,n_cells):
    #Input:
        #path: input path with predictor files [str]
        #tg: tide gauge to open predictor file for [str]
        #n_cells: cells around tide gauge to use [int]
    #Output:
        #predictors: dataset with predictor data [xr ds]
        
    in_cloud = path.startswith('gs://')
        
    if in_cloud:
        fns = fs.ls(path)
        tg_fns = fnmatch.filter(fns,'*'+tg.replace('.csv','')+'*')
    else:
        fns = os.listdir(path)
        tg_fns = fnmatch.filter(fns,'*'+tg.replace('.csv','')+'*')

    if len(tg_fns)==0:
        raise Exception('No predictor dataset found for tide gauge: '+tg)
    elif len(tg_fns)>1:
        raise Exception('Multiple predictor datasets found for tide gauge: '+tg)
    else:
        fn = tg_fns[0].split('/')[-1]

    if in_cloud:
        predictors = xr.open_dataset(os.path.join(path,fn),engine='zarr',chunks='auto')
    else:
        predictors = xr.open_dataset(os.path.join(path,fn),chunks='auto')

    if len(predictors.lon_around_tg)!=len(predictors.lat_around_tg):
        raise Exception('Predictor data must be provided on a squared lon-lat grid.')

    max_n_cells = len(predictors.lon_around_tg)
    
    if n_cells > max_n_cells:
        print('Warning - "n_cells" is larger than the the available number of grid cells. Using the available number instead.')
              
    n_cells = np.min([n_cells,max_n_cells])

    predictors = predictors.isel(lon_around_tg = np.arange(0+int((max_n_cells-n_cells)/2),max_n_cells-int((max_n_cells-n_cells)/2)),
                         lat_around_tg = np.arange(0+int((max_n_cells-n_cells)/2),max_n_cells-int((max_n_cells-n_cells)/2)))

    if 'w' not in predictors.variables:
        predictors['w'] = np.sqrt(predictors['u10']**2+predictors['v10']**2) #compute wind speed from x/y components #rename in case it's called different?

    return predictors
    
    
def open_predictand(path,tg):
    #Input:
        #path: input path to dir with 1 .csv file per tide gauge or netcdf file with all tide gauges [str]
        #tg: tide gauge to get predictand for [str]

    #Output:
        #predictand: dataframe with predictand data [pd df]
        
    if path.endswith('.nc'):
        predictand = xr.open_dataset(path).sel(tg=tg)
        predictand = predictand.rename({'time':'date'})
        predictand = predictand.to_pandas().reset_index()[['surge','date','lon','lat']]

    elif os.path.isdir(path):
        fns = os.listdir(path)
        tg_fns = fnmatch.filter(fns,'*'+tg.replace('.csv','')+'*')
        
        if len(tg_fns)==0:
            raise Exception('No predictand file found for tide gauge: '+tg)
        elif len(tg_fns)>1:
            raise Exception('Multiple predictand files found for tide gauge: '+tg)
        else:
            fn = tg_fns[0].split('/')[-1]

        predictand = pd.read_csv(os.path.join(path,fn))
        predictand['date'] = pd.to_datetime(predictand['date'])

    else:
        raise Exception('Predictand path must be .nc file or folder with .csv files.')
            
    return predictand

def train_predict_output_to_ds(o,yhat,t,hyperparam_settings,tgs,model_architecture,lf_name):

    return xr.Dataset(data_vars=dict(o=(["time","tg"], np.array(o).reshape(-1, len(tgs))),yhat=(["time","tg"], np.array(yhat).reshape(-1, len(tgs))),hyperparameters=(['p'],list(hyperparam_settings)),),
            coords=dict(time=t,tg=tgs,p=['batch_size', 'n_steps', 'n_convlstm', 'n_convlstm_units','n_dense', 'n_dense_units', 'dropout', 'lr', 'l2','dl_alpha'],),
            attrs=dict(description=model_architecture+" - neural network prediction performance.",loss_function=lf_name),)

def add_loss_to_output(output,train_history,n_epochs):
    
    loss = np.nan*np.zeros(n_epochs) #add loss of training to output ds
    val_loss = np.nan*np.zeros(n_epochs)

    loss[0:len(train_history.history['loss'])] = train_history.history['loss']
    val_loss[0:len(train_history.history['val_loss'])] = train_history.history['val_loss']

    output['loss'] = (['e'],loss)
    output['val_loss'] = (['e'],val_loss)
    
    return output

def setup_output_dirs(output_dir,store_model,model_architecture):
    performance_dir = os.path.join(output_dir,'performance',model_architecture)
    model_dir = os.path.join(output_dir,'keras_models',model_architecture)
    if os.path.exists(performance_dir)==False:
        os.makedirs(performance_dir)
    
    if os.path.exists(model_dir)==False and store_model==True:
        os.makedirs(model_dir)
        
        
class Output():
    def __init__(self, path):
        self.path = path
        
    def open_performance_data(self,tgs,stored_per_tg=True):
        
        in_cloud = self.path.startswith('gs://')
        is_file = self.path.endswith(('.nc', '.zarr'))
        
        if is_file:
            if in_cloud:
                ds = xr.open_dataset(self.path,engine='zarr').sel(tg=tgs)
            else:
                ds = xr.open_dataset(self.path).sel(tg=tgs)
        
        if not is_file:
         
            if not stored_per_tg:
                
                if in_cloud:
                    fns = ['gs://'+k for k in fs.ls(self.path) if not k.startswith('.')]
                    ds = xr.open_mfdataset(fns,combine='nested',concat_dim='it',engine='zarr').sel(tg=tgs)
                else:
                    fns = [os.path.join(self.path,k) for k in os.listdir(self.path) if not k.startswith('.')]
                    ds = xr.open_mfdataset(fns,combine='nested',concat_dim='it').sel(tg=tgs)
            else:
                datasets = []

                if in_cloud:
                    fns = fs.ls(self.path)

                    for tg in tgs:
                        tg_fns = ['gs://'+k for k in fnmatch.filter(fns,'*'+tg.replace('.csv','')+'*')]
                        if len(tg_fns) == 0:
                            raise Exception('No performance datasets found for tide gauge: '+tg)
                        datasets.append(xr.open_mfdataset(tg_fns,combine='nested',concat_dim='it',engine='zarr')) 
                else:
                    fns = os.listdir(self.path)
                    for tg in tgs:
                        tg_fns = [os.path.join(self.path,k) for k in fnmatch.filter(fns,'*'+tg.replace('.csv','')+'*')]
                        if len(tg_fns) == 0:
                            raise Exception('No performance datasets found for tide gauge: '+tg)
                        datasets.append(xr.open_mfdataset(tg_fns,combine='nested',concat_dim='it'))

                ds = xr.concat(datasets,dim='tg')
                ds['tg'] = tgs
        self.data = ds
    
    def observed_thresholds(self):
        return self.data.o.isel(it=0).quantile(self.data['quantile'],dim='time')
    
    def observed_stds(self):
        return self.data.o.isel(it=0).std(dim='time',ddof=0)
    
    def get_split_ratios(self):
        return np.isfinite(self.data.o.isel(it=0)).sum(dim='time')/np.isfinite(self.data.o.isel(it=0)).sum(dim='time').sum(dim='split') #fractions of splits
    
    def get_split_lengths(self):
        return np.isfinite(self.data.o.isel(it=0)).sum(dim='time') #total number of finite timesteps in each split