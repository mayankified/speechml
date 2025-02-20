import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
import csv
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# Load dataset with 90 features
df = pd.read_csv('/content/data.csv')

# Select features and labels
x = df.iloc[:, 1:91]  # Assuming first column is filename, selecting 90 features
y = df.iloc[:, -1]  # Last column contains emotion labels

# Encode Labels into Numerical Format
label_mapping = {'happyness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'boredom': 5, 'disgust': 6}
y = y.map(label_mapping)

# Split into training and testing sets (80-20 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights to handle imbalances
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

# Standardize the features
scaler = StandardScaler()
scaler.fit(x_train)
X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Define ANN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

ANN_model = Sequential()
ANN_model.add(Dense(999, input_shape=(90,), activation='elu'))
ANN_model.add(BatchNormalization())
ANN_model.add(Dropout(0.1))

ANN_model.add(Dense(785, activation='elu'))
ANN_model.add(BatchNormalization())
ANN_model.add(Dropout(0.2))

ANN_model.add(Dense(865, activation='elu'))
ANN_model.add(BatchNormalization())
ANN_model.add(Dropout(0.2))

ANN_model.add(Dense(672, activation='elu'))
ANN_model.add(BatchNormalization())
ANN_model.add(Dropout(0.3))

ANN_model.add(Dense(7, activation='softmax'))
ANN_model.summary()

# Save initial model weights
import tempfile
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
ANN_model.save_weights(initial_weights)

# Compile the model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
ANN_model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['SparseCategoricalAccuracy'])

# Define Model Checkpoints and Early Stopping
checkpoint_path = 'Ann_EMODB_90feature.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_sparse_categorical_accuracy', verbose=1,
                                               save_best_only=True, save_weights_only=True)
callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=300, verbose=1,
                                             restore_best_weights=True)

callbacks_list = [callback1, callback2]

# Train the model
history = ANN_model.fit(X_train_scaled, y_train,
                        validation_data=(X_test_scaled, y_test),
                        batch_size=64, epochs=1000, verbose=1,
                        class_weight=class_weights,
                        callbacks=callbacks_list)

# Plot accuracy over epochs
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Load best model weights
ANN_model.load_weights(checkpoint_path)

# Evaluate Model Performance
train_score = ANN_model.evaluate(X_train_scaled, y_train, verbose=1)
test_score = ANN_model.evaluate(X_test_scaled, y_test, verbose=1)

print("Training Accuracy: {:.2f}%".format(train_score[1] * 100))
print("Testing Accuracy: {:.2f}%".format(test_score[1] * 100))

# Get Predictions
y_test_predictions = ANN_model.predict(X_test_scaled)
y_test_predictions = np.argmax(y_test_predictions, axis=1)

# Convert Predictions and True Labels back to Text Labels
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
y_pred_labels = np.array([reverse_label_mapping[i] for i in y_test_predictions])
y_true_labels = np.array([reverse_label_mapping[i] for i in y_test])

# Evaluate Model Performance
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print("Final Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# Plot Confusion Matrix
def plot_cm(y_true, y_pred, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_df, cmap="Blues", annot=annot, fmt='', ax=ax)

plot_cm(y_true_labels, y_pred_labels)