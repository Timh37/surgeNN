import tensorflow as tf
import matplotlib.pyplot as plt
import xarray as xr
import fnmatch
import numpy as np
import os
import sys
import keras
from surgeNN.io import load_predictand,load_predictors
from surgeNN.denseLoss import get_denseloss_weights #if starting with a clean environment, first, in terminal, do->'mamba install kdepy'
from surgeNN.preprocessing import split_predictand_and_predictors_chronological,split_predictand_and_predictors_with_stratified_years
from surgeNN.preprocessing import generate_batched_windowed_filtered_tf_input, generate_windowed_filtered_np_input, deseasonalize_da, deseasonalize_df_var
from surgeNN.evaluation import add_error_metrics_to_prediction_ds,rmse
from surgeNN.preprocessing import stack_predictors_for_lstm, stack_predictors_for_convlstm, normalize_predictand_splits, normalize_predictor_splits,standardize_predictand_splits, standardize_predictor_splits
from surgeNN.models import build_LSTM_stacked, build_ConvLSTM2D_with_channels
from surgeNN.losses import gevl,exp_negexp_mse,obs_squared_weighted_mse, obs_weighted_mse
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import itertools

import gc #callback to clean up garbage after each epoch, not sure if needed (usage: callbacks = [GC_Callback()])
class GC_Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
def train_and_predict(loss_function,loss_function_,possible_params,store_model,
                      split_fractions,split_method,split_start_month,split_seed,
                      n_runs,n_iterations,n_epochs,patience,
                     input_dir,output_dir,
                     tgs,var_names,n_degrees,temp_freq,architecture):
    tg_lons = []
    tg_lats = []
    
    try:
        loss_function_ = eval(loss_function)
    except:
        loss_function_ = loss_function
        
    for tg in tqdm(tgs): #loop over TGs
        #load & process predictors
        #predictors = load_predictors('/home/jovyan/test_surge_models/input/predictors_'+str(temp_freq)+'hourly',tg,n_cells) #open predictor xarray dataset
        predictors = load_predictors('gs://leap-persistent/timh37/era5_predictors/'+str(temp_freq)+'hourly',tg)
        predictors = predictors.sel(time=slice('1979','2017')) #2018 because of end year GTSM simulations that are used as benchmark
        
        if n_degrees > 5:
            print('Cannot use more grid cells than provided in input files, so setting n_degrees to 5 instead of '+str(n_degrees))
            n_degrees = 5
        n_cells = int(n_degrees * 4) #era5 resolution = 0.25 degree

        predictors = predictors.isel(lon_around_tg = np.arange(0+int((20-n_cells)/2),20-int((20-n_cells)/2)),
                                     lat_around_tg = np.arange(0+int((20-n_cells)/2),20-int((20-n_cells)/2))) #standard is 20 by 20, reduce if n_cells<5
    
        if 'w' in var_names and 'w' not in predictors.variables:
            predictors['w'] == np.sqrt((predictors.u10**2+predictors.v10**2))

        for var in var_names: #remove amean
            predictors[var] = predictors[var].groupby(predictors.time.dt.year) - predictors[var].groupby(predictors.time.dt.year).mean('time') #remove annual means
            predictors[var] = deseasonalize_da(predictors[var]) #remove mean seasonal cycle

        #load & process predictands
        predictand = load_predictand('/home/jovyan/test_surge_models/input/t_tide_'+str(temp_freq)+'h_hourly_deseasoned_predictands',tg) #open predictand csv
        predictand = predictand[(predictand['date']>=predictors.time.isel(time=0).values) & (predictand['date']<=predictors.time.isel(time=-1).values)]  # only use predictands in the period for which we also have predictor values, including at preceding timesteps
        predictand = deseasonalize_df_var(predictand,'surge','date') #remove mean seasonal cycle

        predictand = predictand.set_index('date').resample(str(temp_freq)+'h').fillna(method=None) #insert nans where timesteps are missing
        predictand = predictand.reset_index()[['surge','date','lon','lat']]
        #predictand['surge'] = predictand['surge'].rolling(window=int(12/temp_freq+1),min_periods=int(12/temp_freq+1),center=True).mean() #crude way to filter out peaks due to uncorrected tides (Tiggeloven et al., 2021)

        #configure hyperparameter settings
        possible_settings = list(itertools.product(*possible_params))
        n_settings = len(possible_settings)
        
        if n_runs<n_settings:
            selected_settings = np.random.choice(possible_settings, n_runs, replace=False)
        else:
            selected_settings = possible_settings
        
        #train & evaluate models n_iterations * n_runs times:
        for it in np.arange(n_iterations):
            tg_datasets = [] #list to store output
            
            for i,these_settings in enumerate(selected_settings): #loop over hyperparameter combinations
                this_batch_size,this_n_steps,this_n_convlstm,this_n_convlstm_units,this_n_dense,this_n_dense_units,this_dropout,this_lr,this_l2,this_dl_alpha =these_settings

                #generate splits
                #idx_train,idx_val,idx_test,x_train,x_val,x_test,y_train,y_val,y_test = split_predictand_and_predictors_chronological(predictand,predictors,split_fractions,this_n_steps)
                idx_train,idx_val,idx_test,x_train,x_val,x_test,y_train,y_val,y_test = split_predictand_and_predictors_with_stratified_years(predictand,predictors,
                                                                                                                                        split_fractions,this_n_steps,split_start_month,seed=split_seed,how=split_method)

                #standardize the input based on the mean & sd of the train split
                y_train,y_val,y_test,y_train_mean,y_train_sd = standardize_predictand_splits(y_train,y_val,y_test,output_transform = True)
                x_train,x_val,x_test = standardize_predictor_splits(x_train,x_val,x_test)

                if architecture == 'convlstm':
                    x_train, x_val, x_test = [stack_predictors_for_convlstm(k,var_names) for k in [x_train,x_val,x_test]] #stack all predictor variables at each grid cell for LSTM input
                elif architecture == 'lstm':
                    x_train, x_val, x_test = [stack_predictors_for_lstm(k,var_names) for k in [x_train,x_val,x_test]] #stack all predictor variables at each grid cell for LSTM input

                #get Denseloss weights
                w_train,w_val = [get_denseloss_weights(k, alpha = this_dl_alpha ) for k in [y_train,y_val]] #generate DenseLoss weights for train & val splits   

                #get values & timestamps of observations to compare predictions with
                t_train = predictand['date'].values[idx_train][np.isfinite(y_train)]
                t_val = predictand['date'].values[idx_val][np.isfinite(y_val)]
                t_test = predictand['date'].values[idx_test][np.isfinite(y_test)]

                #get windowed predictors & filter rows with nan observations from inputs (for tensorflow pipeline (to handle out-of-memory windowed predictors) see code below)
                x_train,y_train,w_train = generate_windowed_filtered_np_input(x_train,y_train,this_n_steps,w_train)
                x_val,y_val,w_val = generate_windowed_filtered_np_input(x_val,y_val,this_n_steps,w_val)
                x_test,y_test = generate_windowed_filtered_np_input(x_test,y_test,this_n_steps)

                o_train,o_val,o_test = [y_train_sd * k + y_train_mean for k in [y_train,y_val,y_test]] #back-transform observations

                #build model
                if architecture == 'convlstm':
                    model = build_ConvLSTM2D_with_channels(this_n_convlstm, this_n_dense,
                                                           (np.ones(this_n_convlstm)*this_n_convlstm_units).astype(int), 
                                              (np.ones(this_n_dense)*this_n_dense_units).astype(int),
                                              this_n_steps,n_cells,n_cells,len(var_names), 'convlstm0',
                                                           this_dropout, this_lr, loss_function_,l2=this_l2)
                elif architecture == 'lstm':
                    model = build_LSTM_stacked(this_n_convlstm, this_n_dense, 
                                      (np.ones(this_n_convlstm)*this_n_convlstm_units).astype(int), 
                                      (np.ones(this_n_dense)*this_n_dense_units).astype(int), 
                                       this_n_steps,n_cells,n_cells,len(var_names), 'lstm0',
                                       this_dropout, this_lr, loss_function_,l2=this_l2) #loss_function

                #train model:
                if this_dl_alpha: #if using DenseLoss weights
                    train_history = model.fit(x=x_train,y=y_train,epochs=n_epochs,batch_size=this_batch_size,sample_weight=w_train,validation_data=(x_val,y_val,w_val),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                        restore_best_weights=True),GC_Callback()],verbose=2) #with numpy arrays input
                else: #else
                    train_history = model.fit(x=x_train,y=y_train,epochs=n_epochs,batch_size=this_batch_size,validation_data=(x_val,y_val),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                        restore_best_weights=True),GC_Callback()],verbose=2) #with numpy arrays input

                #make predictions & back-transform
                yhat_train = model.predict(x_train,verbose=0).flatten()*y_train_sd + y_train_mean
                yhat_val = model.predict(x_val,verbose=0).flatten()*y_train_sd + y_train_mean
                yhat_test = model.predict(x_test,verbose=0).flatten()*y_train_sd + y_train_mean

                #store results into xr dataset for current i-j
                ds_train = xr.Dataset(data_vars=dict(o=(["time"], o_train),yhat=(["time"], yhat_train),hyperparameters=(['p'],list(these_settings)),),
                coords=dict(time=t_train,p=['batch_size', 'n_steps', 'n_convlstm', 'n_convlstm_units','n_dense', 'n_dense_units', 'dropout', 'lr', 'l2','dl_alpha'],),
                attrs=dict(description=architecture+" - neural network prediction performance.",loss_function=loss_function),)

                ds_val = xr.Dataset(data_vars=dict(o=(["time"], o_val),yhat=(["time"], yhat_val),hyperparameters=(['p'],list(these_settings)),),
                coords=dict(time=t_val,p=['batch_size', 'n_steps', 'n_convlstm', 'n_convlstm_units','n_dense', 'n_dense_units', 'dropout', 'lr', 'l2','dl_alpha'],),
                attrs=dict(description=architecture+" - neural network prediction performance.",loss_function=loss_function),)

                ds_test = xr.Dataset(data_vars=dict(o=(["time"], o_test),yhat=(["time"], yhat_test),hyperparameters=(['p'],list(these_settings)),),
                coords=dict(time=t_test,p=['batch_size', 'n_steps', 'n_convlstm', 'n_convlstm_units','n_dense', 'n_dense_units', 'dropout', 'lr', 'l2','dl_alpha'],),
                attrs=dict(description=architecture+" - neural network prediction performance.",loss_function=loss_function),)

                ds_i = xr.concat((ds_train,ds_val,ds_test),dim='split',coords='different') #concatenate results for each split
                ds_i = ds_i.assign_coords(split = ['train','val','test'])

                loss = np.nan*np.zeros(n_epochs) #add loss of training
                val_loss = np.nan*np.zeros(n_epochs)

                loss[0:len(train_history.history['loss'])] = train_history.history['loss']
                val_loss[0:len(train_history.history['val_loss'])] = train_history.history['val_loss']

                ds_i['loss'] = (['e'],loss)
                ds_i['val_loss'] = (['e'],val_loss)

                tg_datasets.append(ds_i) #append output of current iteration to list of all outputs
                
                if store_model:
                    my_path = os.path.join(output_dir,'keras_models',architecture)
                    my_fn = architecture+'_'+str(temp_freq)+'h_'+tg.replace('.csv','')+'_'+loss_function+'_hp1_i'+str(i)+'_it'

                    model.save(os.path.join(my_path,
                     my_fn+str(len(fnmatch.filter(os.listdir(my_path),my_fn+'*')))+'.keras'))
                
                del model, train_history, ds_i #, x_train, x_val, x_test
                tf.keras.backend.clear_session()
                gc.collect()

            #concatenate across runs & compute statistics
            out_ds = xr.concat(tg_datasets,dim='i',coords='different')
            out_ds = add_error_metrics_to_prediction_ds(out_ds,[.95,.98,.99,.995]) #optional third argument 'max_numT_between_isolated_extremes' to exclude extremes isolated by 12h or more from another extreme from evaluation (to avoid including extremes mainly due to semi-diurnal tides, see manuscript for more explanation)

            out_ds = out_ds.assign_coords(tg = np.array([tg]))
            out_ds = out_ds.assign_coords(lon = ('tg',np.array([predictand['lon'].values[0]])))
            out_ds = out_ds.assign_coords(lat = ('tg',np.array([predictand['lat'].values[0]])))

            out_ds.attrs['temp_freq'] = temp_freq
            out_ds.attrs['n_cells'] = n_cells
            out_ds.attrs['n_epochs'] = n_epochs
            out_ds.attrs['patience'] = patience
            out_ds.attrs['loss_function'] = loss_function
            out_ds.attrs['split_fractions'] = split_fractions
            out_ds.attrs['split_method'] = split_method+'_'+str(split_start_month)+'_'+str(split_seed)
            
            my_path = os.path.join(output_dir,'performance',architecture)
            my_fn = architecture+'_'+str(temp_freq)+'h_'+tg.replace('.csv','')+'_'+loss_function+'_hp1_ndeg'+str(n_degrees)+'_it'
            
            out_ds.to_netcdf(os.path.join(my_path,my_fn+str(len(fnmatch.filter(os.listdir(my_path),my_fn+'*')))+'.nc'),mode='w')
    return out_ds

if __name__ == "__main__":
    '''
    tgs        = ['stavanger-svg-nor-nhs.csv','wick-wic-gbr-bodc.csv','esbjerg-esb-dnk-dmi.csv',
                  'immingham-imm-gbr-bodc.csv','den_helder-denhdr-nld-rws.csv', 'fishguard-fis-gbr-bodc.csv',  
                  'brest-822a-fra-uhslc.csv', 'vigo-vigo-esp-ieo.csv',  'alicante_i_outer_harbour-alio-esp-da_mm.csv'] #all tide gauges to process
    '''
    # --- allow to set specific function arguments using commandline in "execute_train_and_predit.sh"
    default_tgs = ['den_helder-denhdr-nld-rws.csv'] #set default values
    default_architecture = 'lstm'
    default_alpha = np.array([0,1,3,5]).astype('int')
    default_n_degrees = 5
    
    arguments = sys.argv
    if len(arguments)==5: #if specifying tgs, architecture & alpha values from commandline
        tgs = [arguments[1]] #tg to process
        architecture = arguments[2] #architecture to use
        dl_alpha = eval(arguments[3]) #density-based weights tuning parameter
        n_degrees = eval(arguments[4])
    else:
        tgs = default_tgs
        architecture = default_architecture
        dl_alpha = default_alpha
        n_degrees = default_n_degrees
    
    print('Training & predicting for tide gauge: '+str(tgs))
    print('Architecture: '+architecture+'; dl_alpha: ' + str(dl_alpha))
    # --- 
    
    temp_freq = 3 # [hours] temporal frequency to use

    var_names = ['msl','u10','v10','w'] #variables to use

    loss_function = 'mse' # default tensorflow loss function string or string of custom loss function of surgeNN.losses (e.g., 'gevl({gamma})')
    
    input_dir  = '/home/jovyan/test_surge_models/input/' #directory with predictor & predictand data
    output_dir = '/home/jovyan/test_surge_models/results/nn_tests/'#performance/',architecture+'/') #where to store the results

    n_runs = 24 #how many iterations with different hyperparameter combinations to run
    n_iterations = 1
    n_epochs = 100 #how many training epochs
    patience = 10 #early stopping patience

    store_model = 1 #whether to store the tensorflow model
    
    split_fractions = [.6,.2,.2] #train, test, val
    split_method = '99pct'
    split_start_month = 7
    split_seed = 0

    #hyperparameters:
    #dl_alpha = np.array([0,1,3,5]).astype('int')
    batch_size = np.array([128]).astype('int')
    n_steps = np.array([9]).astype('int') #at and 12h or 24h or 36h before (to-do: also use timesteps after?)
    n_convlstm = np.array([1]).astype('int')
    n_convlstm_units = np.array([32]).astype('int')
    n_dense = np.array([2]).astype('int')
    n_dense_units = np.array([32]).astype('int')
    dropout = np.array([0.1,0.2])
    lrs = np.array([1e-5,5e-5,1e-4])
    l1s = np.array([0.02])
    
    possible_params = [batch_size, n_steps, n_convlstm, n_convlstm_units,
                    n_dense, n_dense_units, dropout, lrs, l1s, dl_alpha]
    
    out_ds = train_and_predict(loss_function,loss_function,possible_params,store_model, #execute
                      split_fractions,split_method,split_start_month,split_seed,
                      n_runs,n_iterations,n_epochs,patience,
                     input_dir,output_dir,
                     tgs,var_names,n_degrees,temp_freq,architecture)