import tensorflow as tf
#various tensorflow loss functions

def obs_squared_weighted_mse(y_obs, y_pred):
    #MSE multiplied with squared observed value
    return tf.reduce_mean(tf.math.square(y_obs)*tf.math.square(y_obs - y_pred), axis=-1)  

def obs_weighted_mse(y_obs, y_pred):
    #MSE multiplied with absolute observed value
    return tf.reduce_mean(tf.math.abs(y_obs)*tf.math.square(y_obs - y_pred), axis=-1)  

def minmax_weighted_mse(y_obs, y_pred):
    #MSE multiplied with weight depending on where in min-max range of observations
    w = (y_obs - tf.reduce_min(y_obs) )/(tf.reduce_max(y_obs) - tf.reduce_min(y_obs))
    return tf.reduce_mean(w*(tf.math.square(y_obs - y_pred)), axis=-1)  

def gevl(gamma):
    #based on: https://ieeexplore.ieee.org/abstract/document/9527101
    #call as follows: loss_fn = gevl(gamma)
    gamma = tf.constant(gamma, dtype=tf.float64) #gumbel exponential
    
    def loss_fn(y_obs, y_pred):
        u = y_obs - y_pred
        lf = ((1-tf.math.exp(-tf.math.square(u)))**gamma)*tf.math.square(u) 
        return lf

    return loss_fn

def exp_negexp_mse(a,t):
    #based on: https://journals.ametsoc.org/view/journals/aies/2/1/AIES-D-22-0035.1.xml
    #call as follows: loss_fn = exp_negexp_mse(a,t)
    
    a = tf.constant(a, dtype=tf.float64) #float between 0-1, extent to which to weight positive vs negative
    t = tf.constant(t, dtype=tf.float64) #temperature of the exponential
    
    def loss_fn(y_true,y_pred):
        """Mean squared error of the exponential of the inputs and minus inputs.
      Args:
        y_true: True target.
        y_pred: Predicted value of target or forecast.

      Returns:
        The value of the loss.
      """
        mse = tf.keras.losses.MeanSquaredError()
        
        mse_exp = mse(tf.keras.activations.exponential(y_true/t),tf.keras.activations.exponential(y_pred/t))
        mse_negexp = mse(tf.keras.activations.exponential(-y_true/t),tf.keras.activations.exponential(-y_pred/t))
        
        return a * mse_exp  + (1-a) * mse_negexp
    return loss_fn


def relative_entropy(a,t):
    """Custom loss combining spatial softmax/min + channel-wise KL divergence.

  Custom loss that applies a spatial softmax and softmin to data on the cubed
  sphere, and then the Kullback-Liebler divergence. Follows Qi and Majda
  (PNAS, 2019, https://www.pnas.org/content/117/1/52.short). The loss
  assumes a channels_last keras configuration, such that axes (-4, -3, -2)
  correspond to the coordinates (face, height, width) on the cubed sphere.

  Args:
    y_true: True target.
    y_pred: Predicted value of target or forecast.

  Returns:
    The value of the loss.
  """
    a = tf.constant(a, dtype=tf.float64) #float between 0-1, extent to which to weight positive vs negative
    t = tf.constant(t, dtype=tf.float64) #temperature of the softmax
    
    def loss_fn(y_true,y_pred):
        true_sm_pos = tf.keras.activations.softmax(y_true/t)
        pred_sm_pos = tf.keras.activations.softmax(y_pred/t)

        true_sm_neg = tf.keras.activations.softmax(-y_true/t)
        pred_sm_neg = tf.keras.activations.softmax(-y_pred/t)

        kl = tf.keras.losses.KLDivergence()
    
        return alpha * kl(true_sm_pos, pred_sm_pos) + (1-alpha) * kl(true_sm_neg, pred_sm_neg)
    
    return loss_fn