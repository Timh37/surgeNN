import numpy as np
from sklearn.model_selection import train_test_split
import xarray as xr
import tensorflow as tf
import itertools
from scipy import signal 

#subroutines to generate the train-validation-test splits --->
def split_predictand_and_predictors_chronological(predictand,predictors,split_fractions,n_steps):
    '''
    Split predictand and predictors chronologically.
    
    Input:
        predictand: panda dataframe with predictand timeseries
        predictors: xarray dataset with predictor data
        split_fractions: list of fractions in the order [train,test,val]
        n_steps: number of timesteps at which to use predictors to predict predictand
    Output:
        splitted predictor & predictand data
    '''  
    if np.sum(split_fractions)!=1:
        raise Exception('sum of split fractions must be 1')
        
    idx_train_finite,idx_test_finite,idx_val_finite = get_train_test_val_idx(predictand['surge'].values,
                                                                             split_fractions,shuffle=False) #get split indices (based on fractions of finite (!) observations)
    idx_train, idx_val, idx_test = [range(k[0],k[-1]) for k in [idx_train_finite,idx_val_finite,idx_test_finite]] #expand the indices to finite and NaN observations:

    #take predictand splits
    y_train, y_val, y_test = [predictand['surge'].values[k] for k in [idx_train,idx_val,idx_test]]
    
    #set first n_steps observations in each split to NaN to avoid leakage:
    y_train[0:n_steps-1] = np.nan #predictors not available at t<0
    y_val[0:n_steps-1] = np.nan #to avoid leakage
    y_test[0:n_steps-1] = np.nan #to avoid leakage

    #take predictor splits
    x_train, x_val, x_test = [ predictors.sel(time=slice(predictand['date'].values[k[0]],predictand['date'].values[k[-1]])) for k in [idx_train,idx_val,idx_test]]
    
    if n_steps>1: #prepend nan data to the n-1 steps before the first timestep that predictor data is available to be able to predict y at t_split=0
        x_train,x_val,x_test = [k.reindex({'time':np.append(k.time[0:n_steps-1] - k.time.diff(dim='time')[0] * (n_steps-1),k.time)}) for k in [x_train,x_val,x_test]]
      
    return idx_train,idx_val,idx_test,x_train,x_val,x_test,y_train,y_val,y_test


def split_predictand_stratified(predictand,split_fractions,start_month,how):
    '''
    Split predictand into years, stratify years according to the metric 'how', and randomly assign years from each stratum to the splits.
    
    Input:
        predictand: panda dataframe with predictand timeseries
        split_fractions: list of fractions in the order [train,test,val]
        start_month: month at which to separate the years
        how: stratify on 'amax', '[n]pct' with n an number, e.g., '99pct'
    Output:
        predictand splits
    '''
    predictand['shifted_year'] = [k.year + (np.floor_divide(k.month,start_month)>0).astype('int') for k in predictand.date] #split years from starting month until starting month next year
    
    if split_fractions == [0.5,0.25,0.25]:
        bin_len = 4
    elif split_fractions == [0.6,0.2,0.2]:
        bin_len = 5
    else:
        raise Exception('split fraction not yet implemented')
        #!to-do!: implement something that automatically works out appropriate bins for custom split_fractions and takes into account missing values in the timeseries. For now, only accepting these exact fractions.
        
    if how == 'amax':
        grouped_years = predictand.groupby(predictand.shifted_year).surge.max() #ignores nans by default
    elif 'pct' in how:
        grouped_years = predictand.groupby(predictand.shifted_year).surge.quantile(float(how.replace('pct',''))/100)
    else:
        raise Exception('stratification method not yet implemented')

    ranked_years = grouped_years.sort_values(ascending=False) #sort years by metric
    ranked_years = ranked_years[np.isfinite(ranked_years)] #get rid of years without any observations, for which max/quantile would evaluate to nan
    
    binned_years = [ranked_years[k:k+bin_len] for k in np.arange(0,len(ranked_years),bin_len)] #create stratas of sorted years according to bin length

    years_train = []
    years_val = []
    years_test = []

    #determine which splits to assign years in each bin to:
    for this_bin in binned_years: 
        if len(this_bin)==1: #if only 1 year, there is no need to optimize the distribution
            this_perm = this_bin #pass year to test split directly
            #pass #-->shouldnt this be 'this_perm = this_bin' ?? now the previous this_perm is appended twice??!! I guess what I wanted to do here was CONTINUE instead of PASS??, or tab the append(this_perm) block below the if statement??
            #it doesnt really matter though for the list comprehension that comes after, its just that a bin with 1 year isnt used?
        else:
            bin_idx = np.arange(len(this_bin))
            all_perms = [this_bin.iloc[np.array(k)] for k in list(itertools.permutations(bin_idx))] #get all permutations of this bin's years values

            #minimize sum of absolute deviation from each split to the mean, select corresponding permutations 
            #(this avoids that all the highest or lowest years go to the train split)
            dev_from_mean = [np.nansum([np.abs(np.mean(k.values[0:-2])-k.mean()),np.abs(k.values[-2]-k.mean()),np.abs(k.values[-1]-k.mean())]) for k in all_perms]

            i_minDev = np.where(dev_from_mean<=np.min(dev_from_mean))[0]

            unq_perms,idx_unq_perms = np.unique(np.array(all_perms)[i_minDev,-2:],axis=0,return_index=True)
            optimal_perms = [all_perms[k] for k in i_minDev[idx_unq_perms]]
            this_perm = optimal_perms[np.random.randint(0,len(optimal_perms))]

        years_train.append(this_perm.index.values[0:-2])
        years_val.append(this_perm.index.values[-2:-1])
        years_test.append(this_perm.index.values[-1])

    years_train = np.hstack(years_train)
    years_val = np.hstack(years_val)   
    years_test = np.hstack(years_test)

    predictand_train = predictand[[k in years_train for k in predictand.shifted_year]]
    predictand_val = predictand[[k in years_val for k in predictand.shifted_year]]
    predictand_test = predictand[[k in years_test for k in predictand.shifted_year]]
    
    return predictand_train,predictand_test,predictand_val

def split_predictand_and_predictors_with_stratified_years(predictand,predictors,split_fractions,n_steps,start_month,seed,how):
    '''
    Split predictand and predictors into years, stratify years according to the metric 'how', and semi-randomly assign years from each stratum to the splits.
    
    Input:
        predictand: panda dataframe with predictand timeseries
        predictors: xarray dataset with predictor timeseries
        split_fractions: list of fractions in the order [train,test,val]
        start_month: month at which to separate the years
        seed: seed for random generator
        how: stratify on 'amax', '[n]pct' with n an number, e.g., '99pct'
    Output:
        idx_train, idx_val, idx_test: split indices
        x_train, x_val, x_test: splitted predictor data
        y_train, y_val, y_test: splitted predictand data
    '''
    if np.sum(split_fractions)!=1:
        raise Exception('sum of split fractions must be 1')
        
    np.random.seed(seed)
 
    y_train, y_test, y_val = split_predictand_stratified(predictand,split_fractions,start_month=start_month,how=how)
    idx_train,idx_val,idx_test = [k.index.values for k in [y_train,y_val,y_test]]
 
    x_train,x_val,x_test = [predictors.sel(time=k['date'].values) for k in [y_train,y_val,y_test]]
    y_train, y_val, y_test = [k.surge.values for k in [y_train,y_val,y_test]]

    #set first n_steps observations in each split to NaN to avoid leakage between splits
    y_train[0:n_steps-1] = np.nan #predictors not available at t<0
    y_val[0:n_steps-1] = np.nan #to avoid leakage
    y_test[0:n_steps-1] = np.nan #to avoid leakage

    #prepend nan data to the n-1 steps before the first timestep that predictor data is available to be able to predict y at t_split=0
    if n_steps>1:
        x_train,x_val,x_test = [k.reindex({'time':np.append(k.time[0:n_steps-1] - k.time.diff(dim='time')[0] * (n_steps-1),k.time)}) for k in [x_train,x_val,x_test]]

    return idx_train,idx_val,idx_test,x_train,x_val,x_test,y_train,y_val,y_test

def get_train_test_val_idx(x,split_fractions,shuffle=False,random_state=0):
    '''
    divide x into train, test and validation splits and get indices of the timesteps in each split.
    splits according to fraction of FINITE (!) values in x 
    
    Input:
        x: data to split
        split_fractions: list of fractions in the order [train,test,val]
        shuffle: whether to random shuffle x before taking splits
        random_state: seed for random shuffling
    
    Output: 
        split indices
    '''
    if np.sum(split_fractions)!=1:
        raise Exception('sum of split fractions must be 1')
    train_fraction, test_fraction, val_fraction = split_fractions
    
    idx_finite = np.where(np.isfinite(x))[0] #do not count nans toward requested split fractions
    x_finite = x[np.isfinite(x)]
  
    if shuffle: #first split into train and test:
        x_train, x_test, idx_train, idx_test = train_test_split(x_finite, idx_finite, test_size=1- train_fraction,shuffle=shuffle,random_state=random_state)
    else:
        x_train, x_test, idx_train, idx_test = train_test_split(x_finite, idx_finite, test_size=1 - train_fraction,shuffle=shuffle)
        
    #then split test further into validation and test:
    x_val, x_test, idx_val,idx_test = train_test_split(x_test, idx_test, test_size=test_fraction/(test_fraction + val_fraction),shuffle=False) 

    return idx_train,idx_test,idx_val

#normalization & standardization --->
def normalize_predictand_splits(y_train,y_val,y_test,output_transform=False):
    '''
    Input:
        y_train,y_val,y_test: predictands divided into different splits
        output_transform: whether to output train mean and standard deviation
    Output:
        normalize predictands & optionally transform used to normalize
    '''       
    #transform based on train split only:
    y_train_min = np.nanmin(y_train) 
    y_train_max = np.nanmax(y_train)
               
    y_train,y_val,y_test = [(k - y_train_min)/(y_train_max-y_train_min) for k in [y_train,y_val,y_test]] #note that val & test splits are normalized using train transform
    
    if output_transform == False:
        return y_train,y_val,y_test
    else:
        return y_train,y_val,y_test,y_train_min,y_train_max
    
def normalize_predictor_splits(x_train,x_val,x_test,output_transform=False):
    '''
    Input:
        x_train,x_val,x_test: predictands divided into different splits
        output_transform: whether to output train min and max
    Output:
        normalized predictor xarray datasets & optionally transform used to normalize
    '''    
    #transform based on train split only:
    x_train_min = x_train.min(dim='time') #skips nan by default
    x_train_max = x_train.max(dim='time')

    x_train,x_val,x_test = [(k - x_train_min)/(x_train_max-x_train_min) for k in [x_train,x_val,x_test]] #note that val & test splits are normalized using train transform
    
    if output_transform == False:
        return x_train,x_val,x_test
    else:
        return x_train,x_val,x_test,x_train_min,x_train_max
    
def standardize_predictand_splits(y_train,y_val,y_test,output_transform=False):
    '''
    Input:
        y_train,y_val,y_test: predictands divided into different splits
        output_transform: whether to output train mean and standard deviation
    Output:
        standardized predictands & optionally transform used to standardize
    '''    
    #transform based on train split:
    y_train_mean = np.nanmean(y_train) 
    y_train_sd = np.nanstd(y_train,ddof=0)
               
    y_train, y_val, y_test = [(k - y_train_mean)/y_train_sd for k in [y_train,y_val,y_test]] #note that val & test splits are standardized using train transform

    if output_transform == False:
        return y_train,y_val,y_test
    else:
        return y_train,y_val,y_test,y_train_mean,y_train_sd
    
def standardize_predictor_splits(x_train,x_val,x_test,output_transform=False):
    '''
    Input:
        x_train,x_val,x_test: predictor xarray datasets divided into different splits
        output_transform: whether to output train mean and standard deviation
    Output:
        standardized predictor xarray datasets & optionally transform used to standardize
    '''
    #transform based on train split:
    x_train_mean = x_train.mean(dim='time') #skips nan by default
    x_train_sd = x_train.std(dim='time',ddof=0) #skips nan by default
    
    x_train, x_val, x_test = [(k - x_train_mean)/x_train_sd for k in [x_train,x_val,x_test]] #note that val & test splits are standardized using train transform
    
    if output_transform == False:
        return x_train,x_val,x_test
    else:
        return x_train,x_val,x_test,x_train_mean,x_train_sd

def standardize_timeseries(timeseries):
    return ( timeseries - np.nanmean(timeseries) ) / np.nanstd( timeseries, ddof=0)

#format input in the right way to train the neural networks --->
def generate_windowed_filtered_np_input(x,y,n_steps,w=None):
    '''
    Generate numpy arrays of windowed nan-filtered input data (same as below but using numpy in memory arrays instead of tensorflow datasets for efficiency)
    Input:
        x: predictors
        y: predictands
        n_steps: number of timesteps to use predictors at
        w: sample weights of predictands, optional
    Output:
        x_out: windowed, nan-filtered predictors
        y_out: nan-filtered predictands
    '''
    x_out = np.stack([x[k:k+n_steps,:] for k in np.arange(x.shape[0])][0:-(n_steps-1)],axis=0) #create windowed predictor array (x(t=-n_steps to t=0) to predict y(t=0)
    
    #filter where y is nan
    where_y_is_finite = np.isfinite(y)
    x_out = x_out[where_y_is_finite,...]
    y_out = y[where_y_is_finite]

    if w is not None: #do the same for the weights, if any
        w_out = w[where_y_is_finite]
        return x_out,y_out,w_out
    else:
        return x_out,y_out

def generate_batched_windowed_filtered_tf_input(x,y,n_steps,batch_size,weights=None):
    '''
    Generate tensorflow datasets of windowed nan-filtered input data (avoids having to load everything into memory)
    Input:
        x: predictors
        y: predictands
        n_steps: number of timesteps to use predictors at
        batch_size: batch size for model training
        weights: sample weights of predictands, optional
    Output:
        x_out: windowed, nan-filtered predictors
        y_out: nan-filtered predictands
    '''
    x_ds = tf.data.Dataset.from_tensor_slices(x).window(n_steps, shift=1, drop_remainder=True) #create windowed dataset
    x_ds = x_ds.flat_map(lambda x: x).batch(n_steps)

    y_ds = tf.data.Dataset.from_tensor_slices(y).window(1, shift=1, drop_remainder=True) #create windowed dataset of length 1
    y_ds = y_ds.flat_map(lambda x: x).batch(1)
        
    if weights is not None:
        w_ds = tf.data.Dataset.from_tensor_slices(weights).window(1, shift=1, drop_remainder=True) #create windowed dataset of length 1
        w_ds = w_ds.flat_map(lambda x: x).batch(1)
    
        ds = tf.data.Dataset.zip((x_ds,y_ds,w_ds)) #zip x,y,w
        
        filter_nan = lambda x_ds, y_ds, w_ds: not tf.reduce_any(tf.math.is_nan(y_ds)) #filter out nan-observations
        
    else:# as above, without weights
        ds = tf.data.Dataset.zip((x_ds,y_ds))
  
        filter_nan = lambda x_ds, y_ds: not tf.reduce_any(tf.math.is_nan(y_ds))
    
    ds_filtered =  ds.filter(filter_nan)

    return ds_filtered.batch(batch_size,drop_remainder=True) #split into batches

def batched_windowed_dataset_from_dataset(dataset, n_steps, batch_size):
    '''
    generate windows (expand n_steps preceding timesteps along new dimension) from tensorflow dataset & split into batches
    Input:
        dataset: tensorflow dataset
        n_steps: number of timesteps to use predictors at
        batch_size: batch size for model training
    Output:
        batched and windowed dataset
    '''
    ds = dataset.window(n_steps, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x).batch(n_steps)
    return ds.batch(batch_size)

def stack_predictors_for_lstm(predictors,var_names):
    ''' stack predictors to prepare for lstm input'''
    return np.reshape(np.stack([predictors[k].values for k in var_names],axis=-1),
                      (len(predictors.time),len(predictors.latitude) * len(predictors.longitude) * len(var_names))) #stack grid cells & variables

def stack_predictors_for_convlstm(predictors,var_names):
    ''' stack predictors to prepare for convlstm input'''
    return np.stack([predictors[k].values for k in var_names],axis=-1) #stack variables

def deseasonalize_df_var(df_in,var,time_var):
    '''subtract long-term monthly means from variable in dataframe '''
    df = df_in.copy(deep=True)
    monthly_means_at_timesteps = df.groupby(df[time_var].dt.month).transform('mean')[var].astype('float64') #contains mean of all timesteps in month for all years together at each timestep in that month
    df[var] = df[var] - monthly_means_at_timesteps + np.mean(monthly_means_at_timesteps) #subtract without changing the overall mean of timeseries
    return df

def deseasonalize_da(da):
    '''subtract long-term monthly means from variable in dataset'''
    
    deseasoned_da = da.groupby(da.time.dt.month) - da.groupby(da.time.dt.month).mean('time')
    deseasoned_da = deseasoned_da + (da.mean(dim='time') - deseasoned_da.mean(dim='time'))

    return deseasoned_da

def bandstop_filter_hourly_predictand(predictand,attenuation,f1,f2,order):
    
    def bandstop_filter(data,attenuation,f1,f2,order):
        
        sos = signal.cheby2(order, attenuation, [f1,f2], 'bandstop', output='sos',fs=3600)
        
        return signal.sosfilt(sos, data)
    
    predictand_ffill = predictand.copy(deep=True)
    predictand_ffill = predictand_ffill.set_index('date').resample('1h').fillna(method='ffill')
    predictand_ffill = predictand_ffill.reset_index()[['surge','date','lon','lat']]
    predictand_ffill['surge'] = bandstop_filter(predictand_ffill['surge'].values,attenuation,f1,f2,order)
    predictand_ffill = predictand_ffill.loc[predictand_ffill['date'].isin(predictand['date'])]

    return predictand_ffill