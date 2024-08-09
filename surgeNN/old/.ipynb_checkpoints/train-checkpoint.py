#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:23:34 2024

@author: timhermans
"""
import numpy as np

def train_model(mymodel, patience, save_file, verbose = 1):
    """
    Training loop

    Inputs:
        training_params  : epochs, and min_epochs
        gen              : an lstm_generator from generator.py
        patience         : number of epochs to wait for early stopping
        save_file        : path and filename for temporary weight files
        verbose          : if nonzero, will print train/val loss at each epoch

    """
    train_loss= []
    val_loss = []
    epochs = 20
    min_epochs = 10

    for epoch in range(epochs):

        losses = train_epoch(mymodel)
   
        train_loss.append(losses[0].numpy())
        val_loss.append(losses[1].numpy())
        
        if epoch==0 or (epoch+1) % verbose == 0: 
            print('Epoch: {}, Train Loss: {}, Val Loss: {}'.format(epoch+1, 
                    np.round(train_loss[-1],6), 
                    np.round(val_loss[-1],6)))

        # Save weights if val loss has improved
        if self.val_loss[-1] == np.min(self.val_loss):
            self.save_weights(save_file)
        
        if epoch > min_epochs:
            if np.argmin(self.val_loss) <= epoch - patience: break

    self.load_weights(save_file)
    self.test_loss = self.compute_test_loss(gen).numpy()
    print('Final test loss:', self.test_loss)

def train_epoch(mymodel):
        
        train_batches, val_batches = gen.batches_per_epoch()[:2]
        
        train_loss = tf.keras.metrics.Mean()
        for j in range(train_batches):
            X_hist,y,p_y = gen.next_train()
            train_loss(self.train_step(X_hist,y,p_y))
            
                    
        val_loss = tf.keras.metrics.Mean()
        for j in range(val_batches):
            X_hist, y, p_y = gen.next_val()
            val_loss(self.compute_loss(X_hist,y,p_y))
        
        return train_loss.result(), val_loss.result()