"""models

Module for model architecture definitions
"""
from pathlib import Path
import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras import layers
from keras import initializers

class CDAE:
    """Convolutional Denoising Autoencoder
    
    Refer to Soto & Ashley 2020 (https://www.nature.com/articles/s41746-020-00320-4)
    For model architecutre, refer to the supplementary material here:
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41746-020-00320-4/MediaObjects/41746_2020_320_MOESM1_ESM.pdf
    """

    def __init__(self, input_size: int, models_path: Path):
        # Try loading existing model
        self.model_path = Path(models_path, "cdae.keras").resolve()
        self.input_size = input_size

        self.model = self.load_model()

    def load_model(self) -> keras.Model:
        model = keras.Model()
        if self.model_path.exists():
            model = keras.models.load_model(str(self.model_path))
        else:
            model = self.build_architecture()
            model.save(str(self.model_path))
        return model
    
    def build_architecture(self) -> keras.Model:
        # Input Layer
        input_layer = layers.Input(shape=(self.input_size, 1))

        # Encoder
        x = layers.Conv1D(filters=64, kernel_size=10, activation='relu', padding='same', kernel_initializer=initializers.he_normal())(input_layer)
        x = layers.MaxPooling1D(pool_size=3)(x)  
        x = layers.Conv1D(filters=45, kernel_size=8, activation='relu', padding='same', kernel_initializer=initializers.he_normal())(x)
        x = layers.MaxPooling1D(pool_size=3)(x)  
        x = layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding='same', kernel_initializer=initializers.he_normal())(x)
        cdae_encoder = layers.MaxPooling1D(pool_size=2)(x)

        # Decoder
        x = layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding='same', kernel_initializer=initializers.he_normal())(cdae_encoder)
        x = layers.UpSampling1D(size=2)(x) 
        x = layers.Conv1D(filters=45, kernel_size=8, activation='relu', padding='same', kernel_initializer=initializers.he_normal())(x)
        x = layers.UpSampling1D(size=3)(x)  
        x = layers.Conv1D(filters=64, kernel_size=10, activation='relu', padding='same', kernel_initializer=initializers.he_normal())(x)
        x = layers.UpSampling1D(size=3)(x)  

        # Flatten
        x = layers.Flatten()(x)  # (None, 792, 64) -> (None, 792 * 64) = (None, 50688)

        # Dense
        x = layers.Dense(units=self.input_size, activation='relu', kernel_initializer=initializers.he_normal())(x)  # (None, 50688) -> (None, self.input_size)

        # Reshape to match input shape
        x = layers.Reshape((self.input_size, 1))(x)  # (None, self.input_size) -> (None, self.input_size, 1)

        # Build CDAE model
        cdae = keras.Model(inputs=input_layer, outputs=x)
        cdae.compile(optimizer='adam', loss='mean_squared_error')

        return cdae

        
