from target_relevance import TargetRelevance #if starting with a clean environment, first, in terminal, do->'mamba install kdepy'
import numpy as np

def get_denseloss_weights(data,alpha):
    '''obtain sample weights on KDE following https://link.springer.com/article/10.1007/s10994-021-06023-5
    
    requires 'kdepy' package
    
    Input:
        data: samples of observations to assign weights to
        alpha: scaling factor for those weights
        
    Output
        weights: sample weights
    '''
    where_finite_data = np.isfinite(data)
    
    target_relevance = TargetRelevance(data[where_finite_data], alpha=alpha) #generate loss weights based on finite values in data
    
    weights = np.nan * np.zeros(len(data)) #initialize weights with the same length as data
    weights[where_finite_data] = target_relevance.eval(data[where_finite_data]).flatten()
    
    return weights