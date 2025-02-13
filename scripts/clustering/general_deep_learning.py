#!/usr/bin/env python

"""
This script contains deep learning finetuning class for supervised learning
"""

import tensorflow as tf
import keras_tuner as kt # for finetuning


class MyClusteringTaskTuning(kt.HyperModel):
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
        optimizer=hp.Choice('optimizer', values=['sgd', 'adam'])
        hidden_activation_func=hp.Choice('hidden_activation_func', values=['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish'])
        
        
        if optimizer=='sgd':
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        
        hidden_layers_dict={} # store hidden layers with parameters in order (Python) in dictionary for ease of reference later
        for h in range(1, n_hidden+1):
            hidden_layers_dict[h]=tf.keras.layers.Dense(units=n_neurons, kernel_initializer='he_normal', activation=hidden_activation_func, name=f'hidden{h}')
            
        concat_layer=tf.keras.layers.Concatenate() # define concatenation layer, no execution yet
        
        # Define output layers
        
        output_layer1=tf.keras.layers.Dense(units=2, activation='softmax', name='output_clusters') # need to vary units according to the modeling type; use the right number of clusters found for modeling by hits or QTL
        output_layer2=tf.keras.layers.Dense(1, name='output_distances')
        
        layers={} # store layers in order for actual construction of neural network
        
        # Pass executed layers in order
        
        layers['input']=tf.keras.layers.Input(shape=(2,))
        
        for k in hidden_layers_dict:
            layers[f'hidden{k}']=hidden_layers_dict[k](layers[list(layers.keys())[-1]]) # layer output passed to hidden is the previous one before the 2 above just added
            
        layers['concatenated']=concat_layer([layers['input'], layers[list(layers.keys())[-1]]]) # last layer of the dictionary passed with the input layer for concatenation
        output_clusters=output_layer1(layers['concatenated'])
        layers['output_clusters']=output_clusters
        output_distances=output_layer2(layers['concatenated'])
        layers['output_distances']=output_distances
        
        
        neural_model_unsup=tf.keras.Model(inputs=[layers['input']], outputs=[output_clusters, output_distances]) # specify model inputs and outputs
        
        neural_model_unsup.compile(optimizer=optimizer, loss=('sparse_categorical_crossentropy', 'mse'), metrics=['accuracy', 'RootMeanSquaredError']) # compile model
        
        return neural_model_unsup
        
        
    def fit(self, hp, neural_model, X, y, **kwargs):
        """
        Finetuning of second preprocessing (normalization) and fit parameters
        """
        batch_size=hp.Int('batch_size', min_value=10, max_value=2000) # define possible values for batch size
        
        if hp.Boolean('normalize'): # allow normalization if yes
            normalization_layer=tf.keras.layers.Normalization()
            X=normalization_layer(X)
        
        return neural_model.fit(X, y, batch_size=batch_size, **kwargs)


