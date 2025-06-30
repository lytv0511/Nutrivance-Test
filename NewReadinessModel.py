import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, MultiHeadAttention, LayerNormalization, Dense, 
    Dropout, GlobalAveragePooling1D, Conv1D, Concatenate, Add
)
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

# Generate fresh data
NUM_USERS = 5000
NUM_DAYS = 84  # Doubled to accommodate larger window
WINDOW_SIZE = 42  # Changed from 7 to 42 days
PREDICTION_STEPS = 10  # Number of future days to predict

METRICS = {
    "workout_duration": ("minutes", 20, 120, False),
    "avg_hr": ("BPM", 60, 180, True),
    "active_energy": ("kcal", 200, 800, False),
    "exercise_time": ("minutes", 15, 90, False),
    "steps": ("count", 3000, 20000, False),
    "distance": ("km", 2, 15, False),
    "resting_hr": ("BPM", 50, 90, True),
    "sleep_duration": ("minutes", 360, 540, False),
    "training_load": ("arbitrary", 50, 200, False),
    "workout_intensity": ("BPM", 110, 170, True),
    "vo2_max": ("mL/kg/min", 30, 60, True),
    "hrv": ("ms", 20, 100, True),
    "mindfulness_minutes": ("minutes", 0, 30, False),
}

def generate_training_data():
    all_sequences, all_targets = [], []
    for _ in range(NUM_USERS):
        sequence = np.array([
            np.random.uniform(min_val, max_val, NUM_DAYS + PREDICTION_STEPS)
            for _, (_, min_val, max_val, _) in METRICS.items()
        ]).T
        
        for i in range(NUM_DAYS - WINDOW_SIZE):
            window = sequence[i:i+WINDOW_SIZE]
            target = sequence[i+WINDOW_SIZE:i+WINDOW_SIZE+1]
            all_sequences.append(window)
            all_targets.append(target)
    
    return np.array(all_sequences), np.squeeze(np.array(all_targets))

def transformer_block(inputs, num_heads=8, ff_dim=256):  # Increased capacity
    attn_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=inputs.shape[-1]  # Using shape attribute directly
    )(inputs, inputs)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)  # Using shape attribute directly
    ff_output = Dropout(0.1)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

def build_hybrid_model(window_size, num_features):
    inputs = Input(shape=(window_size, num_features))
    
    # Enhanced CNN branch for longer sequences
    conv_branch = Conv1D(256, kernel_size=7, activation='relu', padding='same')(inputs)
    conv_branch = Conv1D(128, kernel_size=5, activation='relu', padding='same')(conv_branch)
    conv_branch = Conv1D(num_features, kernel_size=3, activation='relu', padding='same')(conv_branch)
    
    # Enhanced transformer branch
    trans_branch = inputs
    for _ in range(4):  # Added more transformer blocks
        trans_branch = transformer_block(trans_branch, ff_dim=256)
    
    merged = Concatenate()([conv_branch, trans_branch])
    
    x = GlobalAveragePooling1D()(merged)
    x = Dense(512, activation='relu')(x)  # Increased layer size
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    outputs = Dense(num_features)(x)
    return Model(inputs=inputs, outputs=outputs)

def predict_sequence(model, initial_sequence, num_steps):
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(num_steps):
        next_step = model.predict(current_sequence[np.newaxis, :, :], verbose=0)
        predictions.append(next_step[0])
        current_sequence[:-1] = current_sequence[1:]
        current_sequence[-1] = next_step[0]
    
    return np.array(predictions)

def examine_data(data, model):
    print(data.shape)
    print(data[0])
    print (data[-1])
    tf.keras.utils.plot_model(model)

# Generate and prepare data
X, y = generate_training_data()
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-5,  # Reduced for stability with larger window
    decay_steps=10000,
    decay_rate=0.96
)

# Use fixed learning rate instead of schedule
model = build_hybrid_model(WINDOW_SIZE, len(METRICS))
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Fixed learning rate
    loss='mse',
    metrics=['mae']
)

# Train with adjusted batch size
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,  # Reduced batch size for larger sequences
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
)

# Example prediction
initial_sequence = X_test[0]
predictions = predict_sequence(model, initial_sequence, PREDICTION_STEPS)

# Save models
tf.saved_model.save(model, "readiness_prediction_model")

# Convert to CoreML
import coremltools as ct
input_shape = (1, WINDOW_SIZE, len(METRICS))
coreml_model = ct.convert(
    "readiness_prediction_model",
    source="tensorflow",
    inputs=[ct.TensorType(shape=input_shape)],
    minimum_deployment_target=ct.target.iOS14
)
coreml_model.save("readiness_prediction_model.mlpackage")