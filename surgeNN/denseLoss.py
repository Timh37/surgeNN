import functools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from KDEpy import FFTKDE

############################# from Steininger et al. (2021):
def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.
    From https://stackoverflow.com/a/41856629'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl

class TargetRelevance():

    def __init__(self, y, alpha=1.0):
        self.alpha = alpha
        #print('TargetRelevance alpha:', self.alpha)

        silverman_bandwidth = 1.06*np.std(y)*np.power(len(y), (-1.0/5.0))

        #print('Using Silverman Bandwidth', silverman_bandwidth)
        best_bandwidth = silverman_bandwidth

        self.kernel = FFTKDE(bw=best_bandwidth).fit(y, weights=None)

        x, y_dens_grid = self.kernel.evaluate(4096)  # Default precision is 1024
        self.x = x
        
        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens_grid = MinMaxScaler().fit_transform(y_dens_grid.reshape(-1, 1)).flatten()

        self.y_dens = np.vectorize(self.get_density)(y)

        self.eps = 1e-6
        w_star = np.maximum(1 - self.alpha * self.y_dens, self.eps)
        self.mean_w_star = np.mean(w_star)
        self.relevances = w_star / self.mean_w_star

    def get_density(self, y):
        idx = bisection(self.x, y)
        try:
            dens = self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            dens = self.y_dens_grid[idx]
        return dens

    @functools.lru_cache(maxsize=100000)
    def eval_single(self, y):
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w_star

    def eval(self, y):
        ys = y.flatten().tolist()
        rels = np.array(list(map(self.eval_single, ys)))[:, None]
        return rels

    def __call__(self, y):
        return self.eval(y)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def get_denseloss_weights(data,alpha):
    '''obtain sample weights on KDE following https://link.springer.com/article/10.1007/s10994-021-06023-5
    
    requires 'kdepy' package
    
    Input:
        data: samples of observations to assign weights to. if 2d, script loops over columns
        alpha: scaling factor for those weights
        
    Output
        weights: sample weights
    '''
    data_shape = data.shape
    
    if len(data_shape)>1:
        ncols = data.shape[-1]
    else:
        ncols = 1
        data = np.reshape(data,(-1,1))

    weights = np.nan * data #initialize weights with the same length as data
        
    for col in range(ncols):
        col_data = data[:,col]
     
        where_finite_data = np.isfinite(col_data)

        target_relevance = TargetRelevance(col_data[where_finite_data], alpha=alpha) #generate loss weights based on finite values in data

        weights[where_finite_data,col] = target_relevance.eval(col_data[where_finite_data]).flatten()
    
    weights = np.reshape(weights,data_shape)
        
    return weights