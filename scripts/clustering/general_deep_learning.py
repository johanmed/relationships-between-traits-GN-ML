#!/usr/bin/env python

"""
This script contains deep learning finetuning class for supervised learning
"""

import tensorflow as tf
import keras_tuner as kt # for finetuning


class MyUnsupervisedTaskTuning(kt.HyperModel):
    """
    Represents hyperparameter tuning for clustering task
    """
    
    def build(self, hp):
        """
        Finetuning at the model level
        """
        
        # Specify possible values of each hyperparameter for search
        
        n_hidden=hp.Int('n_hidden', min_value=1, max_value=10)
        n_neurons=hp.Int('n_neurons', min_value=1, max_value=2) # need to vary max value depending on the number of dimensions of type of modeling
        learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')
        optimizer=hp.Choice('optimizer', values=['sgd', 'adam', 'rmsprop', 'adam', 'adamax', 'adamw', 'nadam'])
        hidden_activation_func=hp.Choice('hidden_activation_func', values=['linear', 'relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish'])
        
        
        if optimizer=='sgd':
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer=='adam':
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer=optimizer
        
        stacked_encoder = tf.keras.Sequential()
        
        stacked_encoder.add(tf.keras.layers.Input(shape=(2,))) # need to vary max value depending on the number of dimensions of type of modeling
        
        for num in range(1, n_hidden+1):
            stacked_encoder.add(tf.keras.layers.Dense(units=n_neurons, kernel_initializer='he_normal', activation=hidden_activation_func))
            stacked_encoder.add(tf.keras.layers.BatchNormalization())
            
        stacked_encoder.add(tf.keras.layers.Dense(units=1, kernel_initializer='he_normal', activation=hidden_activation_func))
        
        stacked_decoder = tf.keras.Sequential()
        
        for num in range(1, n_hidden+1):
            stacked_decoder.add(tf.keras.layers.Dense(units=n_neurons, kernel_initializer='he_normal', activation=hidden_activation_func))
            stacked_decoder.add(tf.keras.layers.BatchNormalization())
            
        stacked_decoder.add(tf.keras.layers.Dense(units=2, kernel_initializer='he_normal', activation=hidden_activation_func))
        
        stacked_autoencoder = tf.keras.Sequential([stacked_encoder, stacked_decoder])
        
        stacked_autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['accuracy']) # compile model
        
        return stacked_autoencoder
        
        
    def fit(self, hp, autoencoder, X, y, **kwargs):
        """
        Finetuning of second preprocessing (normalization) and fit parameters
        """
        batch_size=hp.Int('batch_size', min_value=10, max_value=2000) # define possible values for batch size
        
        if hp.Boolean('normalize'): # allow normalization if yes
            normalization_layer=tf.keras.layers.Normalization()
            X=normalization_layer(X)
        
        return autoencoder.fit(X, y, batch_size=batch_size, **kwargs)



class MySupervisedTaskTuning(kt.HyperModel):

    """
    Represents hyperparameter tuning for clustering task
    """
    
    def build(self, hp):
        """
        Finetuning at the model level
        """
        
        # Specify possible values of each hyperparameter for search
        
        n_hidden=hp.Int('n_hidden', min_value=1, max_value=50, default=10)
        n_neurons=hp.Int('n_neurons', min_value=10, max_value=100)
        learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')
        optimizer=hp.Choice('optimizer', values=['sgd', 'adam', 'rmsprop', 'adam', 'adamax', 'adamw', 'nadam'])
        hidden_activation_func=hp.Choice('hidden_activation_func', values=['linear', 'relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish'])
        
        
        if optimizer=='sgd':
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'adam':
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizer
            
        
        neural_model_sup = tf.keras.Sequential()
        
        neural_model_sup.add(tf.keras.layers.Input(shape=(2,))) # need to vary max value depending on the number of dimensions of type of modeling
        
        for num in range(1, n_hidden+1):
            neural_model_sup.add(tf.keras.layers.Dense(units=n_neurons, kernel_initializer='he_normal', activation=hidden_activation_func))
            neural_model_sup.add(tf.keras.layers.BatchNormalization())
            
        neural_model_sup.add(tf.keras.layers.Dense(units=1, kernel_initializer='he_normal', activation=hidden_activation_func))
    
        
        neural_model_sup.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError']) # compile model
        
        return neural_model_sup
        
        
    def fit(self, hp, neural_model, X, y, **kwargs):
        """
        Finetuning of second preprocessing (normalization) and fit parameters
        """
        batch_size=hp.Int('batch_size', min_value=10, max_value=2000) # define possible values for batch size
        
        if hp.Boolean('normalize'): # allow normalization if yes
            normalization_layer=tf.keras.layers.Normalization()
            X=normalization_layer(X)
        
        return neural_model.fit(X, y, batch_size=batch_size, **kwargs)
