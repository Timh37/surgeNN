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
    #plt.ylim([0.5, 1])
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

def compute_statistics_on_output_ds(out_ds,qnts):
    '''
    Args:
        out_ds: xarray dataset with observations 'o' and predictions 'yhat' as function of 'time'.
        qnts: threshold quantiles above which to define extremes

    Returns:
        out_ds with error statistics added
    '''
    out_ds['r_bulk'] = xr.corr(out_ds.o,out_ds.yhat,dim='time')
    out_ds['rmse_bulk'] = np.sqrt( ((out_ds.o-out_ds.yhat)**2 ).mean(dim='time'))

    out_ds['r_extremes'] = xr.corr(out_ds.o.where(out_ds.o>=out_ds.o.quantile(qnts,dim='time')),out_ds.yhat.where(out_ds.o>=out_ds.o.quantile(qnts,dim='time')),dim='time')
    out_ds['rmse_extremes'] = np.sqrt((( out_ds.o.where(out_ds.o>=out_ds.o.quantile(qnts,dim='time'))-out_ds.yhat.where(out_ds.o>=out_ds.o.quantile(qnts,dim='time')))**2 ).mean(dim='time'))

    #confusion matrix based on observational threshold
    out_ds['true_pos'] =  ((out_ds.o>=out_ds.o.quantile(qnts,dim='time')) & (out_ds.yhat>=out_ds.o.quantile(qnts,dim='time'))).where(np.isfinite(out_ds.o)).sum(dim='time')
    out_ds['false_neg'] =  ((out_ds.o>=out_ds.o.quantile(qnts,dim='time')) & ((out_ds.yhat>=out_ds.o.quantile(qnts,dim='time'))==False)).where(np.isfinite(out_ds.o)).sum(dim='time')
    out_ds['false_pos'] =  (((out_ds.o>=out_ds.o.quantile(qnts,dim='time'))==False) & (out_ds.yhat>=out_ds.o.quantile(qnts,dim='time'))).where(np.isfinite(out_ds.o)).sum(dim='time')
    out_ds['true_neg'] =  (((out_ds.o>=out_ds.o.quantile(qnts,dim='time'))==False) & ((out_ds.yhat>=out_ds.o.quantile(qnts,dim='time'))==False)).where(np.isfinite(out_ds.o)).sum(dim='time')

    #confusion matrix derivatives
    out_ds['precision'] = compute_precision(out_ds.true_pos,out_ds.false_pos)
    out_ds['recall'] = compute_recall(out_ds.true_pos,out_ds.false_neg)
    out_ds['f1'] = compute_f1(out_ds.precision,out_ds.recall)
    
    return out_ds