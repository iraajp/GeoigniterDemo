import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
X = np.load("data/X.npy")  # shape: (1024, 1024, 2)
Y = np.load("data/Y.npy")  # shape: (1024, 1024)

# Flatten spatial dimensions
H, W, C = X.shape
X_flat = X.reshape(-1, C)           # shape: (1024*1024, 2)
Y_flat = Y.flatten()                # shape: (1024*1024,)

# Optional: convert to binary class
Y_flat = (Y_flat > 0).astype(np.uint8)

# Build simple Dense model
model = Sequential([
    Input(shape=(2,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_flat, Y_flat, epochs=5, batch_size=64, validation_split=0.2)

# Save the trained model
model.save("models/fire_model.h5")
print("✅ Model saved to models/fire_model.h5")
