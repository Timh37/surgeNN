#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:00:43 2024

@author: timhermans
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tensorflow as tf

def generate_windowed_finite_numpy_input(x,y,n_steps,w=None):
    x_out = np.stack([x[k:k+n_steps,:] for k in np.arange(x.shape[0])][0:-(n_steps-1)],axis=0) #create windowed predictor array
    
    #filter where y is nan
    where_y_is_finite = np.isfinite(y)
    x_out = x_out[where_y_is_finite,...]
    y_out = y[where_y_is_finite]

    if w is not None: #do the same for the weights
        w_out = w[where_y_is_finite]
        return x_out,y_out,w_out
    else:
        return x_out,y_out

def create_batched_sequenced_datasets(x,y,n_steps,batch_size,weights=None):

    if weights is not None:
        x_ds = tf.data.Dataset.from_tensor_slices(x).window(n_steps, shift=1, drop_remainder=True) #create windowed dataset
        x_ds = x_ds.flat_map(lambda x: x).batch(n_steps)

        y_ds = tf.data.Dataset.from_tensor_slices(y).window(1, shift=1, drop_remainder=True)
        y_ds = y_ds.flat_map(lambda x: x).batch(1)

        w_ds = tf.data.Dataset.from_tensor_slices(weights).window(1, shift=1, drop_remainder=True)
        w_ds = w_ds.flat_map(lambda x: x).batch(1)
    
        ds = tf.data.Dataset.zip((x_ds,y_ds,w_ds))
        
        filter_nan = lambda x_ds, y_ds, w_ds: not tf.reduce_any(tf.math.is_nan(y_ds))
        ds_filtered =  ds.filter(filter_nan)
        
    else:
        x_ds = tf.data.Dataset.from_tensor_slices(x).window(n_steps, shift=1, drop_remainder=True) #create windowed dataset
        x_ds = x_ds.flat_map(lambda x: x).batch(n_steps)

        y_ds = tf.data.Dataset.from_tensor_slices(y).window(1, shift=1, drop_remainder=True)
        y_ds = y_ds.flat_map(lambda x: x).batch(1)
        
        ds = tf.data.Dataset.zip((x_ds,y_ds))
  
        filter_nan = lambda x_ds, y_ds: not tf.reduce_any(tf.math.is_nan(y_ds))
        ds_filtered =  ds.filter(filter_nan)

    return ds_filtered.batch(batch_size,drop_remainder=True)

def sequenced_dataset_from_dataset(dataset, input_sequence_length, batch_size): #does not filter out nans
    ds = dataset.window(input_sequence_length, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x).batch(input_sequence_length)
    
    return ds.batch(batch_size)