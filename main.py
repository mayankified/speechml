from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa
import tensorflow as tf
import os
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from vmdpy import VMD
import EntropyHub as eh
from pyentrp import entropy as ent
import scipy

# Initialize FastAPI
app = FastAPI()

# Emotion Mapping (same as in training)
emotion_mapping = {
    0: 'happyness', 1: 'neutral', 2: 'anger', 
    3: 'sadness', 4: 'fear', 5: 'boredom', 6: 'disgust'
}

# Variational Mode Decomposition Parameters (same as training)
alpha = 5000
tau = 0
K = 3
DC = 0
init = 1
tol = 1e-7
sr = 16000
q = scipy.signal.windows.hanning(800)

# Define Model Architecture (same as training)
def build_model():
    model = Sequential()
    model.add(Dense(999, input_shape=(90,), activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(785, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(865, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(672, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(7, activation='softmax'))  # 7 emotion classes
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load Model and Weights
ANN_model = build_model()
ANN_model.load_weights("Ann_EMODB_90feature.weights.h5")
print("✅ Model Loaded Successfully!")

# Feature Extraction Function (same as training)
def extract_audio_features(file_path, sr=16000):
    signal, sr = librosa.load(file_path, sr=sr)
    signal = librosa.effects.preemphasis(signal, coef=0.97)
    signal = librosa.util.normalize(signal)

    hop_length = 400
    frame_length = 800
    feature = []

    for i in range(0, len(signal), hop_length):
        current_frame = signal[i:i + frame_length]
        current_frame1 = np.zeros(800)

        for index in range(len(current_frame)):
            current_frame1[index] = current_frame[index] * q[index]

        # Apply Variational Mode Decomposition (VMD)
        u, u_hat, omega = VMD(current_frame1, alpha, tau, K, DC, init, tol)

        data = []
        for j in range(len(u)):
            MFCCs = librosa.feature.mfcc(y=u[j], sr=sr, n_mfcc=30).T
            MFCCs_mean = np.mean(MFCCs, axis=0)
            data.extend(MFCCs_mean)  # Append MFCCs (90 features)

        feature.append(data)

    feature = np.array(feature)
    feature_avg = np.mean(feature, axis=0)

    return feature_avg

# FastAPI Endpoint to Predict Emotion from Audio File
@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract features
        features = extract_audio_features(temp_file)
        features = np.array(features).reshape(1, -1)  # Reshape for model input

        # Predict Emotion
        predictions = ANN_model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_emotion = emotion_mapping[predicted_class]

        # Delete temp file
        os.remove(temp_file)

        return {"emotion": predicted_emotion, "confidence": float(np.max(predictions))}
    
    except Exception as e:
        return {"error": str(e)}

# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "🎙️ Emotion Detection API is running!"}
