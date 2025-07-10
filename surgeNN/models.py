#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:16:55 2024

@author: timhermans
"""
import tensorflow as tf
import keras
from keras import regularizers
from keras import layers
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, MaxPooling2D, MaxPooling3D,Dropout, SpatialDropout2D,SpatialDropout3D,BatchNormalization, ConvLSTM2D
from keras.models import Model
from keras import regularizers
from sklearn.decomposition import PCA
import statsmodels.api as sm
import numpy as np
import xarray as xr

#lstm models ---->
def build_LSTM_stacked(n_lstm, n_dense, n_lstm_units, n_neurons,
                     predictor_shape,n_output,
                     model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build an LSTM network where predictor variables are inputted as features
    
    Input:
        n_lstm: number of lstm layers
        n_dense: number of dense layers
        n_lstm_unis: list of number of units per lstm layer
        n_neurons: list of number of neurons per dense layer
        predictor_shape: shape of predictor data (n_leading_timesteps,n_predictors)
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    lstm_input = keras.Input(shape=predictor_shape)
    x = lstm_input
    for l in np.arange(n_lstm-1):
        x = layers.LSTM(n_lstm_units[l], return_sequences=True)(x)  
    x = layers.LSTM(n_lstm_units[n_lstm-1], return_sequences=False)(x)  
    
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout((dropout_rate))(x)
    x = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout((dropout_rate))(x)   
    
    outputs = []
    for k in np.arange(n_output):
        outputs.append(layers.Dense(1,activation='linear',dtype='float64')(x))

    model = keras.Model(inputs=lstm_input, outputs=outputs,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,weighted_metrics=[]) #compile the model
    return model

def build_LSTM_stacked_multioutput_static(n_lstm, n_dense, n_lstm_units, n_neurons,
                     predictor_shape,n_output,n_static,
                     model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build an LSTM network where predictor variables are inputted as features
    
    Input:
        n_lstm: number of lstm layers
        n_dense: number of dense layers
        n_lstm_unis: list of number of units per lstm layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    lstm_input = keras.Input(shape=predictor_shape)
    
    static_inputs = []
    for k in np.arange(n_output):
        static_inputs.append(keras.Input(shape=(n_static)))
    
    x = lstm_input
    for l in np.arange(n_lstm-1):
        x = layers.LSTM(n_lstm_units[l], return_sequences=True)(x)  
    x = layers.LSTM(n_lstm_units[n_lstm-1], return_sequences=False)(x)  
    
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout((dropout_rate))(x)
    x = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout((dropout_rate))(x)   
    
    outputs = []
    for k in np.arange(n_output):
        static = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(static_inputs[k])
        static_and_x = layers.concatenate((x,static))
        
        x_k = layers.Dense(n_neurons[n_dense-1],activation='linear',dtype='float64')(static_and_x)
        
        outputs.append(layers.Dense(1,activation='linear',dtype='float64')(x_k))

    model = keras.Model(inputs=[lstm_input]+static_inputs, outputs=outputs,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,weighted_metrics=[]) #compile the model
    return model

def build_LSTM_stacked_multioutput_cond(n_lstm, n_dense, n_lstm_units, n_neurons,
                     predictor_shape,n_static_cond,n_output,
                     model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build an LSTM network where predictor variables are inputted as features
    
    Input:
        n_lstm: number of lstm layers
        n_dense: number of dense layers
        n_lstm_unis: list of number of units per lstm layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    from cond_rnn import ConditionalRecurrent
    
    cond_input = keras.Input(shape=(n_static_cond))
    lstm_input = keras.Input(shape=predictor_shape)
    
    x = lstm_input
    for l in np.arange(n_lstm-1):
        if l==0:
            x = ConditionalRecurrent(layers.LSTM(n_lstm_units[l], return_sequences=True))([x,cond_input])  
        else:
            x = layers.LSTM(n_lstm_units[l], return_sequences=True)(x) 
            
    if n_lstm==1:
        x = ConditionalRecurrent(layers.LSTM(n_lstm_units[n_lstm-1], return_sequences=False))([x,cond_input])  
    else:
        x = layers.LSTM(n_lstm_units[n_lstm-1], return_sequences=False)(x)  
    
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout((dropout_rate))(x)
    x = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout((dropout_rate))(x)   
    
    outputs = []
    for k in np.arange(n_output):
        outputs.append(layers.Dense(1,activation='linear',dtype='float64')(x))

    model = keras.Model(inputs=[lstm_input,cond_input], outputs=outputs,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,weighted_metrics=[]) #compile the model
    
    return model

def build_LSTM_per_variable(n_lstm, n_dense, n_lstm_units, n_neurons,
                     n_timesteps,n_lats,n_lons,n_predictor_variables, 
                     model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build an LSTM network where predictor variables are fed to the vlstm layers separately and then merged
    
    Input:
        n_lstm: number of lstm layers
        n_dense: number of dense layers
        n_lstm_unis: list of number of units per lstm layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    input_shape = (n_timesteps,n_lats*n_lons) #channels last
    
    inputs = []
    lstmd_vars = []
    
    for var in np.arange(n_predictor_variables): #for each predictor variable, apply convlstm layers:
        lstm_input = keras.Input(shape=input_shape)
        x = lstm_input
        for l in np.arange(n_lstm-1):
            x = layers.LSTM(n_lstm_units[l], return_sequences=True)(x)        
        x = layers.LSTM(n_lstm_units[n_lstm-1], return_sequences=False)(x)
        
        lstmd_vars.append(x)
        inputs.append(lstm_input)
    
    concatenated = layers.concatenate(lstmd_vars) #concatenate layers for each variables
    x_ = concatenated
    
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x_ = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x_)
        x_ = layers.Dropout((dropout_rate))(x_)
    
    x_ = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x_)
    x_ = layers.Dropout((dropout_rate))(x_)   
    
    output = layers.Dense(1,activation='linear',dtype='float64')(x_)

    model = keras.Model(inputs=inputs, outputs=output,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,metrics=['accuracy']) #compile the model
    
    return model
#<----

#convlstm models ---->
def build_ConvLSTM2D_with_channels(n_convlstm, n_dense, n_kernels, n_neurons,
                     n_timesteps,n_lats,n_lons,n_predictor_variables,n_output, 
                     model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build a convolutional LSTM network where predictor variables are inputted as channels
    
    Input:
        n_convlstm: number of convlstm layers
        n_dense: number of dense layers
        n_kernels: list of number of kernels per convlstm layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    input_shape = (n_timesteps,n_lats,n_lons, n_predictor_variables) #channels last
    
    #add convlstm layers
    cnn_input = keras.Input(shape=input_shape)
    x = cnn_input
    for l in np.arange(n_convlstm-1):
        x = layers.ConvLSTM2D(n_kernels[l], kernel_size=(3, 3), return_sequences=True, padding='same',activation='relu')(x)        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")(x)
    
    x = layers.ConvLSTM2D(n_kernels[n_convlstm-1], kernel_size=(3, 3), return_sequences=False, padding='same',activation='relu')(x)        
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2),padding="same")(x)        
    #x = SpatialDropout2D(dropout_rate)(x)
    x = layers.Flatten()(x)
        
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout((dropout_rate))(x)
    x = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout((dropout_rate))(x)   
    
    outputs = []
    for k in np.arange(n_output):
        outputs.append(layers.Dense(1,activation='linear',dtype='float64')(x))
        
    model = keras.Model(inputs=cnn_input, outputs=outputs,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,metrics=['accuracy']) #compile the model
    
    return model

def build_ConvLSTM2D_per_variable(n_convlstm, n_dense, n_kernels, n_neurons,
                     n_timesteps,n_lats,n_lons,n_predictor_variables, 
                     model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build a convolutional LSTM network where predictor variables are fed to the convlstm layers separately and then merged
    
    Input:
        n_convlstm: number of convlstm layers
        n_dense: number of dense layers
        n_kernels: list of number of kernels per convlstm layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    input_shape = (n_timesteps,n_lats,n_lons, 1) #channels last
    
    inputs = []
    convoluted_vars = []
    
    #add convlstm layers
    for var in np.arange(n_predictor_variables): #for each predictor variable, apply convlstm layers:
        cnn_input = keras.Input(shape=input_shape)
        x = cnn_input
        for l in np.arange(n_convlstm-1):
            x = layers.ConvLSTM2D(n_kernels[l], kernel_size=(3, 3), return_sequences=True, padding='same',activation='relu')(x)        
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")(x)

        x = layers.ConvLSTM2D(n_kernels[n_convlstm-1], kernel_size=(3, 3), return_sequences=False, padding='same',activation='relu')(x)        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2),padding="same")(x)        
        #x = SpatialDropout2D(dropout_rate)(x)
        x = layers.Flatten()(x)
        
        convoluted_vars.append(x)
        inputs.append(cnn_input)
    
    concatenated = layers.concatenate(convoluted_vars) #concatenate layers for each variable
    x_ = concatenated
    
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x_ = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x_)
        x_ = layers.Dropout((dropout_rate))(x_)
    x_ = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x_)
    x_ = layers.Dropout((dropout_rate))(x_)   
    
    output = layers.Dense(1,activation='linear',dtype='float64')(x_)

    model = keras.Model(inputs=inputs, outputs=output,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,metrics=['accuracy']) #compile the model
    
    return model
#<----


#conv2d models ---->
def build_Conv2D_with_channels(n_conv, n_dense, n_kernels, n_neurons, 
                            n_steps, n_lats,n_lons,n_predictor_variables, 
                            model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build a convolutional network where predictor variables are inputted as channels
    
    Input:
        n_conv: number of con layers
        n_dense: number of dense layers
        n_kernels: list of number of kernels per conv2d layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    input_shape = (n_steps,n_lats,n_lons, n_predictor_variables)
    all_input = keras.Input(shape=input_shape)
    input_per_timestep = tf.unstack(all_input,axis=1)
    
    convoluted_steps = []
    for step_input in input_per_timestep: #for each timestep, apply convolutions:
        x=step_input
        for l in np.arange(n_conv):
            x = layers.Conv2D(n_kernels[l], kernel_size=(3, 3), padding='same', activation='relu')(x)        
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        
        #x = SpatialDropout2D(dropout_rate)(x)
        x = layers.Flatten()(x)
        convoluted_steps.append(x)
        
    concatenated = layers.concatenate(convoluted_steps) #concatenate convoluted data for each timestep
    x_ = concatenated
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x_ = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x_)
        x_ = layers.Dropout((dropout_rate))(x_)
        
    x_ = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x_)
    x_ = layers.Dropout((dropout_rate))(x_)   
    output = layers.Dense(1,activation='linear',dtype='float64')(x_)
    
    model = keras.Model(inputs=all_input, outputs=output,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,metrics=['accuracy']) #compile the model
    return model
#<----


#conv3d models ---->
def build_Conv3D_with_channels(n_conv, n_dense, n_kernels, n_neurons, 
                 n_steps, n_lats,n_lons,n_predictor_variables, 
                 model_name, dropout_rate, lr, loss_function,l2=0.01):
    '''build a convolutional3d network where predictor variables are inputted as channels
    
    Input:
        n_conv: number of conv3d layers
        n_dense: number of dense layers
        n_kernels: list of number of kernels per conv3d layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    input_shape = (n_steps,n_lats,n_lons,n_predictor_variables)
    cnn_input = keras.Input(shape=input_shape)
    x=cnn_input
    for l in np.arange(n_conv):
        x = layers.Conv3D(n_kernels[l], kernel_size=(3, 3,3), padding='same', activation='relu')(x)        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2,2), padding="same")(x)

    #x = SpatialDropout3D(dropout_rate)(x)
    x = layers.Flatten()(x)
        
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        x = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout((dropout_rate))(x)
        
    x = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout((dropout_rate))(x)   
    output = layers.Dense(1,activation='linear',dtype='float64')(x)
    
    model = keras.Model(inputs=cnn_input, outputs=output,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,metrics=['accuracy']) #compile the model
    return model
#<----


#Sequential convolutional lstm ---->
def build_Conv2D_then_LSTM_with_channels(n_conv,n_lstm, n_dense, 
                           n_kernels, n_lstm_units, n_neurons,
                             n_steps,n_lats,n_lons,n_predictor_variables, 
                             model_name, dropout_rate, lr, loss_function,l2=0.01): #to-do
    
    '''build a convolutional then LSTM network where predictor variables are inputted as channels
    
    Input:
        n_conv: number of conv2d layers
        n_lstm: number of lstm layers
        n_dense: number of dense layers
        n_kernels: list of number of kernels per conv layer
        n_lstm_units: list of number of units per lstm layer
        n_neurons: list of number of neurons per dense layer
        n_timesteps,n_lats,n_lons,n_predictor_variables: number of timesteps, latitude, longitude grid cells and predictor variables used
        model_name: tensorflow model name
        dropout_rate: dropout rate
        lr: learning rate
        loss_function: loss function to use
        l2: regularization rate
    Output:
        compiled tensorflow model
    '''
    input_shape = (n_steps,n_lats,n_lons, n_predictor_variables)
    all_input = keras.Input(shape=input_shape)
    input_per_timestep = tf.unstack(all_input,axis=1)
    
    convoluted_steps = []
    for step_input in input_per_timestep: #for each predictor input variable, apply convolution:
        #cnn_input = keras.Input(shape=input_shape)
        x=step_input
        for l in np.arange(n_conv):
            x = layers.Conv2D(n_kernels[l], kernel_size=(3, 3), padding='same', activation='relu')(x)        
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        
        #x = SpatialDropout2D(dropout_rate)(x)
        x = layers.Flatten()(x)
        
        convoluted_steps.append(x)
    
    lstm_input = tf.stack(convoluted_steps,axis=1)
    x_ = lstm_input
    
    for l in np.arange(n_lstm-1):
        x_ = layers.LSTM(n_lstm_units[l], return_sequences=True)(x_)        
        
    x_ = layers.LSTM(n_lstm_units[n_lstm-1], return_sequences=False)(x_)
        
    xd = x_
    #add densely connected layers:
    for l in np.arange(n_dense-1):
        xd = layers.Dense(n_neurons[l],activation='relu',activity_regularizer=regularizers.l2(l2))(xd)
        xd = layers.Dropout((dropout_rate))(xd)
        
    xd = layers.Dense(n_neurons[n_dense-1],activation='relu',activity_regularizer=regularizers.l2(l2))(xd)
    xd = layers.Dropout((dropout_rate))(xd)   
    output = layers.Dense(1,activation='linear',dtype='float64')(xd)
    
    model = keras.Model(inputs=all_input, outputs=output,name=model_name)  #construct the Keras model   
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,metrics=['accuracy']) #compile the model
    return model

'''#from elsewhere, not currently using this:
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
class Attention(Layer): 
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)
''' 
#<----

#MLR based on Tadesse et al. ---->
def train_gssr_mlr(predictors,predictand):
    '''training step, estimate mlr coefficients'''
    
    pca = PCA(.95) #find pcs explaining at least 95% of the variance
    pca.fit(predictors)
    X_pca = pca.transform(predictors)

    X_pca = sm.add_constant(X_pca) #add intercept for the regression
    est = sm.OLS(predictand, X_pca).fit() 
    
    return est.params,pca.components_

def predict_gssr_mlr(predictors,mlr_coefs,training_components,predictor_vars,n_steps):
    pca = PCA(len(mlr_coefs[np.isfinite(mlr_coefs)])-1) #get same number of pcs as used for regression coefficient estimation
    pca.fit(predictors)
    X_pca = pca.transform(predictors)
    prediction_components = pca.components_
    
    #sign check of the PCA components using pressure
    if 'msl' not in predictor_vars:
        raise Exception('No pressure ("msl") in predictor variables, cannot check the sign of the prediction PCA components against the trained PCA components.')
    
    #find pressure indices in predictor component matrix
    i_msl = np.where(np.array(predictor_vars)=='msl')[0][0] #where is "msl" in the list of predictor variables
    n_gridcells = int(prediction_components.shape[-1]/(n_steps*len(predictor_vars))) #total number of grid cells

    p_idx = []
    for k in np.arange(n_steps):
        p_idx.append(np.arange(0,n_gridcells)+k*len(predictor_vars)*n_gridcells + i_msl * n_gridcells)
    p_idx = np.hstack(p_idx)

    #compute rmses of pressure indices
    rmses = np.sqrt(np.mean((prediction_components[:,p_idx]-training_components[:,p_idx])**2,axis=-1))
    rmses_flipped = np.sqrt(np.mean((prediction_components[:,p_idx]--training_components[:,p_idx])**2,axis=-1))

    s = (rmses<rmses_flipped).astype('int') #flip pcs if rmse of flipped pc is lower
    s[s==0]=-1
    X_pca = X_pca * s

    prediction = np.sum(mlr_coefs * np.column_stack((np.ones(X_pca.shape[0]),X_pca)),axis=1)
    
    return prediction, prediction_components 