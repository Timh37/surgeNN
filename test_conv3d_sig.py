import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import torch.nn.functional as F
import torch.optim as optim
import signature_example_code.example_single.utils as utils
import signature_example_code.example_single.models as models
import xarray as xr
from sklearn.metrics import confusion_matrix
from surgeNN.io import load_predictand,load_predictors
from surgeNN.utils import get_train_test_val_idx, normalize_timeseries, rmse
from scipy import stats
from target_relevance import TargetRelevance #terminal->'mamba install kdepy' works
from statsmodels.distributions.empirical_distribution import ECDF
import scipy
import torch.utils.data as torchdata
import torch

#settings
max_epochs = 30
lr = 3e-5
optimizer_fn=optim.Adam
n_steps = 8
stopping_patience = 4 #epochs


def loss_fn(y_pre,y):
    return torch.sqrt(F.mse_loss(y_pre,y))

def mse_loss(y_pre,y):
    return F.mse_loss(y_pre,y)

#--->>
def weighted_mse_loss(y_pred,y):
    y_true,weights=y
    return (weights * (y_pred - y_true) ** 2).mean()

def weighted_mae_loss(y_pred,y):
    y_true,weights=y
    return (weights * torch.abs(y_pred - y_true)).mean()


def relentropy(y_pre,y):
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    outputs = F.log_softmax(y_pre/1)
    targets = F.softmax(y/1)  # normalize target as a distribution
    outputs_n = F.log_softmax(y_pre/-1)
    targets_n = F.softmax(y/-1)

    loss = 0.5*criterion(outputs, targets) + 0.5 * criterion(outputs_n, targets_n);
               
    return loss

tg = 'den_helder-denhdr-nld-rws.csv' #site to predict
predictand = load_predictand('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/t_tide_6h_anoms_deseasoned_predictands',tg) #open predictand csv
predictors = load_predictors('/Users/timhermans/Documents/Github/surgeNN/input/predictors_6hourly',tg,5) #open predictor xarray dataset
predictors = predictors.sel(time=slice('1980','2016')) #period for which we have hydrodynamic output as well

# only use predictands at timesteps for which we have predictor values:
predictand = predictand[(predictand['date']>=predictors.time.isel(time=0).values) & (predictand['date']<=predictors.time.isel(time=-1).values)] 

#preprocess predictors
predictors = predictors.sel(time=predictand['date'].values)
predictors = (predictors-predictors.mean(dim='time'))/predictors.std(dim='time',ddof=0) #normalize each variable in dataset
my_predictors = np.stack((predictors.msl.values,predictors.w.values,predictors.u10.values,predictors.v10.values),axis=-1) #load into memory

#preprocess observations
predictand['surge'] = predictand['surge'].rolling(window=5,min_periods=1,center=True).mean() #in the literature, 12h-rolling means are used to account for spurious tidal peaks. Trying this out at the moment.
predictand['surge'] = normalize_timeseries(predictand['surge']) #normalize predictands
surge_obs = predictand['surge'].values #get values as array

idx_train,idx_test,idx_val = get_train_test_val_idx(my_predictors,surge_obs,[.6,.2,.2],shuffle=False) #split data into training, validation and testing and get indices
my_predictors = np.moveaxis(my_predictors,-1,1) #needed for the input that the pytorch model takes

#independent:
#target_relevance = TargetRelevance(surge_obs[np.append(idx_train,idx_val)], alpha=3)
#weights = target_relevance.eval(surge_obs[np.append(idx_train,idx_val)]).flatten()
#for now:
target_relevance = TargetRelevance(surge_obs, alpha=3)
weights = target_relevance.eval(surge_obs).flatten()

t = scipy.stats.ecdf(surge_obs)
ecdf_vals = t.cdf.evaluate(surge_obs)
ecdf_weights = ecdf_vals#/np.sum(ecdf_vals)

#create dataloaders of windowed & batched data
train_dataloader,eval_dataloader,test_dataloader,example_batch_x,example_batch_y, example_batch_z = utils.generate_batched_windowed_data_from_timeseries(my_predictors,
                                                                                                                                        np.expand_dims(surge_obs,-1).astype('float32'), 
                                                                                                                                        idx_train,idx_val,idx_test,
                                                                                                                                        n_steps,128,sample_weights=np.expand_dims(weights,-1).astype('float32'))
    
#create model trainer class
model_trainer = utils.create_model_supervised_trainer(max_epochs=max_epochs, optimizer_fn=optimizer_fn,
                                                      loss_fn=weighted_mae_loss, weight_loss = True,patience = stopping_patience, train_dataloader=train_dataloader,
                                                      eval_dataloader=eval_dataloader, example_batch_x=example_batch_x, lr=lr)
#train model
history={}
deepsignet = models.conv3d_sig(augment_include_original=True, augment_include_time=True, T=1, signature_truncation=1, n_conv3d_kernels=24, n_predictor_vars=4,p_dropout3d=0.1)
model_trainer(deepsignet, 'DeepSigNet', history)


def make_inference(model,dataloader):
    with torch.no_grad():
        y_pred = np.array([])
        y_true =np.array([])
        for x_,y_ in dataloader:#len(test_windows)):
            y_pred = np.append(y_pred,model(x_).numpy().flatten())
            y_true = np.append(y_true,y_.numpy().flatten())
    return y_pred,y_true

          
surge_cnn_test,surge_obs_test = make_inference(deepsignet,test_dataloader) #predict with the test set predictors

### do some evaluation
threshold_pct = 98 #percentile of storm surge data to look at
threshold_value = np.percentile(surge_obs_test,threshold_pct) #threshold value

surge_obs_test_exceedances = (surge_obs_test>=threshold_value).flatten() #find where storm surges exceed threshold (extremes), for observations
surge_cnn_test_exceedances = (surge_cnn_test>=threshold_value).flatten() #find where storm surges exceed threshold (extremes), for predictions with CNN

print('---CNN---')
print('bulk correlation r='+str(np.corrcoef(surge_cnn_test,surge_obs_test)[0][1]))
print('bulk RMSE='+str(rmse(surge_cnn_test,surge_obs_test)))
print('Confusion matrix exceedances above {0}th percentile:'.format(threshold_pct))
print(confusion_matrix(surge_obs_test_exceedances,surge_cnn_test_exceedances))

print('Correlation at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('r=' + str(np.corrcoef(surge_cnn_test[surge_obs_test_exceedances],surge_obs_test[surge_obs_test_exceedances])[0][1]))
print('RMSE at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('RMSE=' + str(rmse(surge_cnn_test[surge_obs_test_exceedances],surge_obs_test[surge_obs_test_exceedances])))

'''
### also look at outputs of a hydrodynamic model
surge_codec = xr.open_dataset('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/CoDEC_ERA5_at_gesla3_tgs_eu_6hourly_anoms.nc')
surge_codec = surge_codec.sel(time=predictand['date'].values)
surge_codec['surge'] = surge_codec['surge'].rolling(time=5,min_periods=1,center=True).mean()
surge_codec['surge'] = (surge_codec['surge'] - surge_codec['surge'].mean(dim='time'))/surge_codec['surge'].std(dim='time',ddof=0) #normalize
surge_codec_test = surge_codec.sel(tg=tg).sel(time=predictand['date'].values[idx_test][n_steps-1:-1]).surge.values #select test timesteps
surge_codec_test_exceedances = (surge_codec_test>=threshold_value).flatten() #find where exceeding threshold
 
print('---CoDEC---')
print('bulk correlation r='+str(np.corrcoef(surge_codec_test,surge_obs_test)[0][1]))
print('bulk RMSE='+str(rmse(surge_codec_test,surge_obs_test)))
print('Confusion matrix exceedances above {0}th percentile:'.format(threshold_pct))
print(confusion_matrix(surge_obs_test_exceedances,surge_codec_test_exceedances))

print('Correlation at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('r=' + str(np.corrcoef(surge_codec_test[surge_obs_test_exceedances],surge_obs_test[surge_obs_test_exceedances])[0][1]))
print('RMSE at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('RMSE=' + str(rmse(surge_codec_test[surge_obs_test_exceedances],surge_obs_test[surge_obs_test_exceedances])))
'''

plt.figure()
plt.plot(surge_obs_test[surge_obs_test_exceedances],label='obs')
plt.plot(surge_cnn_test[surge_obs_test_exceedances],label='nn')
plt.legend()

plt.figure()
plt.plot(surge_obs[idx_test],label='obs')
plt.plot(surge_cnn_test,label='nn')
plt.legend()