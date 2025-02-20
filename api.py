from flask import Flask, request, jsonify
import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import joblib
from scipy.stats import entropy
from scipy import signal
from vmdpy import VMD
import soundfile as sf

app = Flask(__name__)

# Define label mapping
label_mapping = {'happyness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'boredom': 5, 'disgust': 6}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Load pre-trained model weights
checkpoint_path = "Ann_EMODB_90feature.weights.h5"

# Recreate the ANN model architecture
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

    Dense(7, activation='softmax')
])

# Compile the model
ANN_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['SparseCategoricalAccuracy'])

# Load model weights
if os.path.exists(checkpoint_path):
    ANN_model.load_weights(checkpoint_path)
    print("✅ Model weights loaded successfully!")
else:
    raise FileNotFoundError("❌ Error: Model weights file not found!")

# Load StandardScaler
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded successfully!")
else:
    scaler = None  # If missing, use a dummy scaler

# Define processing functions
def extract_features(audio_file):
    sr = 16000  # Sample rate
    alpha = 5000
    tau = 0
    K = 3
    DC = 0
    init = 1
    tol = 1e-7
    frame_length = 800
    hop_length = 400

    try:
        # Load audio file
        signal, sample_rate = librosa.load(audio_file, sr=sr)
        signal = librosa.effects.preemphasis(signal, coef=0.97)

        # Normalize the signal
        signal = librosa.util.normalize(signal)

        # Hann Window
        q = signal.windows.hanning(frame_length)

        # Feature extraction
        feature = []
        for i in range(0, len(signal), hop_length):
            current_frame = signal[i:i + frame_length]
            current_frame1 = np.zeros(frame_length)
            for index in range(len(current_frame)):
                current_frame1[index] = current_frame[index] * q[index]

            # Apply VMD
            u, u_hat, omega = VMD(current_frame1, alpha, tau, K, DC, init, tol)

            # Extract MFCCs
            data = []
            for i in range(len(u)):
                MFCCs = librosa.feature.mfcc(y=u[i], n_fft=frame_length, hop_length=hop_length, sr=sr, n_mfcc=30).T
                MFCCs_mean = np.mean(MFCCs, axis=0)  # Average across frames
                data.extend(MFCCs_mean)

            feature.append(data)

        # Average features
        feature = np.array(feature)
        feature_avg = np.mean(feature, axis=0)

        return feature_avg.tolist()
    
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# API route for emotion prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    # Save file temporarily
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    # Extract features
    features = extract_features(audio_path)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    # Scale features if scaler exists
    if scaler:
        features = scaler.transform([features])

    # Predict emotion
    prediction = ANN_model.predict(features)
    predicted_label = np.argmax(prediction)
    predicted_emotion = reverse_label_mapping[predicted_label]

    # Clean up temp file
    os.remove(audio_path)

    return jsonify({
        "predicted_emotion": predicted_emotion,
        "confidence_scores": prediction.tolist()
    })

# Run Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
