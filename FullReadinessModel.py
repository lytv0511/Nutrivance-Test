import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, MultiHeadAttention, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D, Add
)
from keras.optimizers import Adam
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import coremltools as ct

# Constants
NUM_USERS = 5000
NUM_DAYS = 42
WINDOW_SIZE = 7
PREDICTION_STEPS = 10

METRICS = {
    "workout_duration": (20, 120),
    "avg_hr": (60, 180),
    "active_energy": (200, 800),
    "exercise_time": (15, 90),
    "steps": (3000, 20000),
    "distance": (2, 15),
    "resting_hr": (50, 90),
    "sleep_duration": (360, 540),
    "training_load": (50, 200),
    "workout_intensity": (110, 170),
    "vo2_max": (30, 60),
    "hrv": (20, 100),
    "mindfulness_minutes": (0, 30),
}

# Data Generation
def generate_training_data():
    all_sequences, all_targets = [], []
    for _ in range(NUM_USERS):
        sequence = np.array([
            np.random.uniform(min_val, max_val, NUM_DAYS + PREDICTION_STEPS)
            for min_val, max_val in METRICS.values()
        ]).T
        for i in range(NUM_DAYS):
            all_sequences.append(sequence[i:i+WINDOW_SIZE])
            all_targets.append(sequence[i+WINDOW_SIZE:i+WINDOW_SIZE+1])
    return np.array(all_sequences), np.array(all_targets)

X, y = generate_training_data()

# Transformer Block
def transformer_block(inputs, num_heads=4, key_dim=32, ff_dim=128, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(inputs, inputs)
    attn_output = Add()([inputs, attn_output])
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
    ff_output = Dense(ff_dim, activation="relu")(attn_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    return LayerNormalization(epsilon=1e-6)(Add()([attn_output, ff_output]))

# Build Transformer Model
def build_transformer_model(window_size, num_features, num_layers=3):
    inputs = Input(shape=(window_size, num_features))
    x = inputs
    for _ in range(num_layers):
        x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_features)(x)
    return Model(inputs, outputs)

# Train and Save Model
train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

model = build_transformer_model(WINDOW_SIZE, len(METRICS))
model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

tf.saved_model.save(model, "transformer_model")
coreml_model = ct.convert("transformer_model", source="tensorflow", inputs=[ct.TensorType(shape=(1, WINDOW_SIZE, len(METRICS)))], minimum_deployment_target=ct.target.iOS14)
coreml_model.save("transformer_model.mlpackage")