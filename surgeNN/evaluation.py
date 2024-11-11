import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_loss_evolution(model_history): 
    '''plot how the loss function of the neural network evolved during training'''
    f = plt.figure()
    
    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    
    return f

def rmse(y_obs,y_pred):
    return np.sqrt( np.mean( (y_obs-y_pred)**2 ) )

def r_pearson(y_obs,y_pred):
    return np.corrcoef(y_obs,y_pred)[0][1]

def compute_precision(true_pos,false_pos):
    return true_pos/(true_pos+false_pos)

def compute_recall(true_pos,false_neg):
    return true_pos/(true_pos+false_neg)

def compute_f1(precision,recall):
    return 2*recall*precision/(recall+precision)

def add_error_metrics_to_prediction_ds(prediction_ds,qnts,max_numT_between_isolated_extremes=None):
    '''
    Args:
        prediction_ds: xarray dataset with observations 'o' and predictions 'yhat' as function of 'time'.
        qnts: list of threshold quantiles above which to define extremes
        max_numT_between_isolated_extremes: integer-threshold to drop extremes that are isolated by more than max_numT_between_isolated_extremes time-steps
    
    Returns:
        prediction_ds with error statistics added
    '''
    prediction_ds['r_bulk'] = xr.corr(prediction_ds.o,prediction_ds.yhat,dim='time') #correlation of all predictions and observations
    prediction_ds['rmse_bulk'] = np.sqrt( ((prediction_ds.o-prediction_ds.yhat)**2 ).mean(dim='time')) #rmse of all predictions and observations

    where_observed_peaks_ = (prediction_ds.o>=prediction_ds.o.quantile(qnts,dim='time')) #find exceedances
    
    if max_numT_between_isolated_extremes: #possibly filter out isolated extremes using max_numT_between_isolated_extremes
        where_observed_peaks = ((where_observed_peaks_) & (where_observed_peaks_.rolling(time=1+2*int(max_numT_between_isolated_extremes),center='True').sum()>1))
    else:
        where_observed_peaks = where_observed_peaks_
        
    #error metrics where obs are extreme:    
    prediction_ds['r_extremes'] = xr.corr(prediction_ds.o.where(where_observed_peaks),
                                          prediction_ds.yhat.where(where_observed_peaks),dim='time') 
    prediction_ds['rmse_extremes'] = np.sqrt((( prediction_ds.o.where(where_observed_peaks)-prediction_ds.yhat.where(where_observed_peaks))**2 ).mean(dim='time'))

    #confusion matrix based on observational threshold
    prediction_ds['true_pos'] =  ((where_observed_peaks) & (prediction_ds.yhat>=prediction_ds.o.quantile(qnts,dim='time'))).where(np.isfinite(prediction_ds.o)).sum(dim='time')
    prediction_ds['false_neg'] =  ((where_observed_peaks) & ((prediction_ds.yhat>=prediction_ds.o.quantile(qnts,dim='time'))==False)).where(np.isfinite(prediction_ds.o)).sum(dim='time')
    prediction_ds['false_pos'] =  ((where_observed_peaks==False) & (prediction_ds.yhat>=prediction_ds.o.quantile(qnts,dim='time'))).where(np.isfinite(prediction_ds.o)).sum(dim='time')
    prediction_ds['true_neg'] =  ((where_observed_peaks==False) & ((prediction_ds.yhat>=prediction_ds.o.quantile(qnts,dim='time'))==False)).where(np.isfinite(prediction_ds.o)).sum(dim='time')

    #confusion matrix derivatives
    prediction_ds['precision'] = compute_precision(prediction_ds.true_pos,prediction_ds.false_pos)
    prediction_ds['recall'] = compute_recall(prediction_ds.true_pos,prediction_ds.false_neg)
    prediction_ds['f1'] = compute_f1(prediction_ds.precision,prediction_ds.recall)

    return prediction_ds
