import tensorflow as tf
import xarray as xr
import fnmatch
import numpy as np
import os
import sys
import keras
from surgeNN import io, preprocessing
from surgeNN.io import train_predict_output_to_ds, setup_output_dirs, add_loss_to_output
from surgeNN.evaluation import add_error_metrics_to_prediction_ds,rmse
from surgeNN.models import build_LSTM_stacked, build_ConvLSTM2D_with_channels
from surgeNN.losses import gevl,exp_negexp_mse,obs_squared_weighted_mse, obs_weighted_mse
from tqdm import tqdm
import itertools
import random

import gc #callback to clean up garbage after each epoch, not sure if strictly necessary (usage: callbacks = [GC_Callback()])
class GC_Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
def train_and_predict(model_architecture,loss_function,hyperparam_options,
                      split_fractions,strat_metric,strat_start_month,strat_seed,tgs,n_runs,n_iterations,n_epochs,patience,
                      predictor_path,predictor_vars,predictor_degrees,predictand_path,temp_freq,output_dir,store_model):
    '''
    Execute model training and prediction and evaluation on test split. Uses a stratified splitting scheme.
    
    Input:
        model_architecture:  neural network type to use STR: ['lstm','convlstm']
        loss function:       dictionary with a key containing the name of the loss function, and a value containing the loss function (tensorflow default or custom) [DICT: {STR:STR or func}]
        hyperparam_options:  length 10 list of vectors containing options for each hyperparameter in the order: 
                             [[batch_size],[n_steps],[n_layers],[n_layer_units],[n_dense_layers],[n_dense_layer_units],[dropout_rate],[learning_rate],[l1],[dl_alpha]] 
        split_fractions:     list of split fractions in the order [train,test,val], sum must be 1
        strat_metric:        metric based on which to stratify years (annual maximum or a percentile) ['amax' OR STR '<float>pct']
        strat_start_month:   starting month of each year for stratified sampling, from 1 to 12 [int]
        tgs:                 tide gauges to train for [list]
        strat_seed:          seed to control randomness in the stratified sampling scheme [int]
        n_runs:              number of runs with different hyperparameters to do [int]
        n_iterations:        number of repetitions to run each setting with [int]
        n_epochs:            maximum number of epochs to train with [int]
        patience:            patience parameter for early stopping [int]
        predictor_path:      path to load predictor data from (must point to directories organized per temporal frequency) [str] ---> expected to have regularly spaced and no missing data at the moment!
        predictor_vars:      predictor variables to use [list]
        predictor_degrees:   number of degrees of predictors around tide gauge to use [int/float]
        predictand_path:     path to load predictand data from (must point to directories organized per temporal frequency) [str]
        temp_freq: temporal  hourly frequency of input data to use [int]
        output_dir:          directory to store models and model performance to [str]
        store_models:        boolean indicating whether to output Tensorflow models [True/False]
        
    Output: 
        out_ds: xarray dataset containing the model settings, predictions and performance
    '''
    setup_output_dirs(output_dir,store_model,model_architecture)
    
    lf_name = list(loss_function.keys())[0]
    lf = list(loss_function.values())[0]
    
    try:
        lf = eval(lf)
    except:
        pass
        
    for tg in tqdm(tgs): #loop over TGs:
        
        ### (1) Load & prepare input data
        n_cells = int(predictor_degrees * (4/1)) #determine how many grid cells around TG to use (era5 resolution = 0.25 degree)
        
        predictors = io.Predictor(predictor_path)
        predictors.open_dataset(tg,predictor_vars,n_cells)
        predictors.trim_years(1979,2017)
        predictors.subtract_annual_means()
        predictors.deseasonalize()
 
        predictand = io.Predictand(predictand_path)
        predictand.open_dataset(tg)
        predictand.trim_dates(predictors.data.time.isel(time=0).values,predictors.data.time.isel(time=-1).values)
        predictand.deseasonalize()
        predictand.resample_fillna(str(temp_freq)+'h')
        '''
        import pandas as pd
        predictand.data.to_csv('test.csv') #save preprocessed predictand data
        '''
        ### (2) Configure sets of hyperparameters to run with
        all_settings = list(itertools.product(*hyperparam_options))
        n_settings = len(all_settings)

        if n_runs<n_settings:
            selected_settings = random.sample(all_settings, n_runs)
        else:
            selected_settings = all_settings
        
        ### (3) Execute training & evaluation (n_iterations * n_runs times):
        for it in np.arange(n_iterations): #for each iteration
            tg_datasets = [] #list to store output
            
            for i,these_settings in enumerate(selected_settings): #for each set of hyperparameters
                
                this_batch_size,this_n_steps,this_n_convlstm,this_n_convlstm_units,this_n_dense,this_n_dense_units,this_dropout,this_lr,this_l2,this_dl_alpha = these_settings #pick hyperparameters for this run
 
                #generate train, validation and test splits & prepare input for model
                model_input = preprocessing.trainingInput(predictors,predictand)
                if model_architecture == 'lstm':
                    model_input.stack_predictor_coords()
                    
                model_input.split_stratified(split_fractions,this_n_steps,strat_start_month,strat_seed,strat_metric)
                y_train_mean,y_train_sd = model_input.standardize()
        
                model_input.compute_denseloss_weights(this_dl_alpha) #generate the Denseloss weights for each split
               
                x_train,y_train,w_train = model_input.get_windowed_filtered_np_input('train',this_n_steps) #generate input for neural network model
                x_val,y_val,w_val       = model_input.get_windowed_filtered_np_input('val',this_n_steps)
                x_test,y_test,w_test    = model_input.get_windowed_filtered_np_input('test',this_n_steps)

                o_train,o_val,o_test = [model_input.y_train_sd * k + model_input.y_train_mean for k in [y_train,y_val,y_test]] #back-transform observations

                #build model
                if model_architecture == 'convlstm':
                    model = build_ConvLSTM2D_with_channels(this_n_convlstm, this_n_dense,(np.ones(this_n_convlstm)*this_n_convlstm_units).astype(int), 
                                              (np.ones(this_n_dense)*this_n_dense_units).astype(int),this_n_steps,
                                                           len(predictors.data.lat_around_tg),len(predictors.data.lon_around_tg),len(predictor_vars), 1,'convlstm0',this_dropout, this_lr, lf,l2=this_l2)
                elif model_architecture == 'lstm':
                    model = build_LSTM_stacked(this_n_convlstm, this_n_dense,(np.ones(this_n_convlstm)*this_n_convlstm_units).astype(int), 
                                      (np.ones(this_n_dense)*this_n_dense_units).astype(int),(this_n_steps,
                                                            len(predictors.data.lon_around_tg)**2 * len(predictor_vars)),1,'lstm0',this_dropout, this_lr, lf,l2=this_l2)

                #train model:
                if this_dl_alpha: #if using DenseLoss weights
                    train_history = model.fit(x=x_train,y=y_train,epochs=n_epochs,batch_size=this_batch_size,sample_weight=w_train,validation_data=(x_val,y_val,w_val),
                                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,restore_best_weights=True),GC_Callback()],verbose=2) #with numpy arrays input
                else: #else
                    train_history = model.fit(x=x_train,y=y_train,epochs=n_epochs,batch_size=this_batch_size,validation_data=(x_val,y_val),
                                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,restore_best_weights=True),GC_Callback()],verbose=2) #with numpy arrays input

                #make predictions & back-transform
                yhat_train = model.predict(x_train,verbose=0)*model_input.y_train_sd + model_input.y_train_mean#.flatten()*y_train_sd + y_train_mean
                yhat_val = model.predict(x_val,verbose=0)*model_input.y_train_sd + model_input.y_train_mean#.flatten()*y_train_sd + y_train_mean
                yhat_test = model.predict(x_test,verbose=0)*model_input.y_train_sd + model_input.y_train_mean#.flatten()*y_train_sd + y_train_mean
                
                #store results into xr dataset for current settings and iteration
                ds_train = train_predict_output_to_ds(o_train,yhat_train,model_input.t_train,these_settings,np.array([tg]),model_architecture,lf_name)
                ds_val = train_predict_output_to_ds(o_val,yhat_val,model_input.t_val,these_settings,np.array([tg]),model_architecture,lf_name)
                ds_test = train_predict_output_to_ds(o_test,yhat_test,model_input.t_test,these_settings,np.array([tg]),model_architecture,lf_name)
          
                ds_i = xr.concat((ds_train,ds_val,ds_test),dim='split',coords='different') #concatenate results for each split
                ds_i = ds_i.assign_coords(split = ['train','val','test'])

                ds_i = add_loss_to_output(ds_i,train_history,n_epochs)
                tg_datasets.append(ds_i) #append output of current iteration to list of all outputs
                    
                if store_model:
                    my_path = os.path.join(output_dir,'keras_models',model_architecture)
                    my_fn = model_architecture+'_'+str(temp_freq)+'h_'+tg.replace('.csv','')+'_'+lf_name+'_hp1_i'+str(i)+'_it'
                    
                    model.save(os.path.join(my_path,
                                     my_fn+str(len(fnmatch.filter(os.listdir(my_path),my_fn+'*')))+'.keras'))
                
                del model, model_input, train_history, ds_i #, x_train, x_val, x_test
                tf.keras.backend.clear_session()
                gc.collect()

            #concatenate across runs & compute statistics
            out_ds = xr.concat(tg_datasets,dim='i',coords='different')
            out_ds = add_error_metrics_to_prediction_ds(out_ds,[.95,.98,.99,.995],3) #optional third argument 'max_numT_between_isolated_extremes' to exclude extremes isolated by more than n timesteps from another extreme from evaluation (to avoid including extremes mainly due to semi-diurnal tides, see manuscript for more explanation)

            out_ds = out_ds.assign_coords(lon = ('tg',np.array([predictand.data['lon'].values[0]])))
            out_ds = out_ds.assign_coords(lat = ('tg',np.array([predictand.data['lat'].values[0]])))

            if len(n_steps) == 1: #if n_steps is constant across i, obs doesn't need to have i as a dimension. Saves storage.
                out_ds['o'] = out_ds['o'].isel(i=0,drop=True)
                
            out_ds.attrs['temp_freq'] = temp_freq
            out_ds.attrs['n_cells'] = n_cells
            out_ds.attrs['n_epochs'] = n_epochs
            out_ds.attrs['patience'] = patience
            out_ds.attrs['loss_function'] = lf_name
            out_ds.attrs['split_fractions'] = split_fractions
            out_ds.attrs['stratification'] = strat_metric+'_'+str(strat_start_month)+'_'+str(strat_seed)
            out_ds.attrs['predictor_vars'] = predictor_vars
            
            my_path = os.path.join(output_dir,'performance',model_architecture)
            my_fn = model_architecture+'_'+str(temp_freq)+'h_'+tg.replace('.csv','')+'_'+lf_name+'_hp1_ndeg'+str(predictor_degrees)+'_it'
            
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
        model_architecture = arguments[2] #architecture to use
        dl_alpha = eval(arguments[3]) #density-based weights tuning parameter
        predictor_degrees = eval(arguments[4])
    else:
        tgs = default_tgs
        model_architecture = default_architecture
        dl_alpha = default_alpha
        predictor_degrees = default_n_degrees
    
    print('Training & predicting for tide gauges: '+str(tgs))
    print('Architecture: '+model_architecture+'; dl_alpha: ' + str(dl_alpha))
    # --- 
    
    #configure standard settings if running main:
    
    #i/o
    predictor_path  = 'gs://leap-persistent/timh37/era5_predictors/3hourly/' #'gs://leap-persistent/timh37/HighResMIP/surgeNN_predictors/'#
    #predictand_path = '/home/jovyan/surgeNN/input/CoDEC_ERA5_at_gesla3_tgs_eu_hourly_anoms.nc'#'/home/jovyan/surgeNN/input/GTSM_HighResMIP_HadGEM3-GC31-HM_at_gesla3_tgs_stretched_3hourly_rounded10min_after2.nc'
    predictand_path = '/home/jovyan/surgeNN/input/t_tide_3h_hourly_deseasoned_predictands'
    output_dir = '/home/jovyan/surgeNN/results/nns/'
    store_model = 0#1 #whether to store the tensorflow models
    temp_freq = 3 # [hours] temporal frequency to use
    
    #training
    predictor_vars = ['msl','u10','v10','w'] #variables to use
    n_runs = 24 #how many hyperparameter combinations to run
    n_iterations = 1 #how many iterations to run per hyperparameter combination
    n_epochs = 100 #how many training epochs
    patience = 10 #early stopping patience
    loss_function = {'mse':'mse'} # default tensorflow loss function string or string of custom loss function of surgeNN.losses (e.g., 'gevl({gamma})')
    
    #splitting & stratified sampling
    split_fractions = [.6,.2,.2] #train, test, val
    strat_metric = '99pct'
    strat_start_month = 7
    strat_seed = 0

    #hyperparameters:
    #dl_alpha = np.array([0,1,3,5]).astype('int') #defined from command line
   
    batch_size = np.array([128]).astype('int')
    n_steps = np.array([9]).astype('int')
    n_convlstm = np.array([1]).astype('int')
    n_convlstm_units = np.array([32]).astype('int')
    n_dense = np.array([2]).astype('int')
    n_dense_units = np.array([32]).astype('int')
    dropout = np.array([0.1,0.2])
    lrs = np.array([1e-5,5e-5,1e-4])
    l1s = np.array([0.02])
    
    hyperparam_options = [batch_size, n_steps, n_convlstm, n_convlstm_units,
                    n_dense, n_dense_units, dropout, lrs, l1s, dl_alpha]
    
    out_ds = train_and_predict(model_architecture,loss_function,hyperparam_options, #execute
                      split_fractions,strat_metric,strat_start_month,strat_seed,tgs,n_runs,n_iterations,n_epochs,patience,
                     predictor_path,predictor_vars,predictor_degrees,predictand_path,temp_freq,output_dir,store_model)
    
