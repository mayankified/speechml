import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler

# Define label mapping
label_mapping = {'happyness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'boredom': 5, 'disgust': 6}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Recreate model architecture
ANN_model = Sequential([
    Dense(999, input_shape=(90,), activation='elu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(785, activation='elu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(865, activation='elu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(672, activation='elu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile the model
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
ANN_model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['SparseCategoricalAccuracy'])

# Load saved weights
checkpoint_path = 'Ann_EMODB_90feature.weights.h5'
ANN_model.load_weights(checkpoint_path)
print("Model weights loaded successfully!")
