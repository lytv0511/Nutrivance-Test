import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import coremltools as ct

# Load the dataset
data = pd.read_csv('synthetic_health_data.csv')

# Fill missing values
data.ffill(inplace=True)

# Convert 'reason_1', 'reason_2', 'reason_3' to numeric
for col in ['reason_1', 'reason_2', 'reason_3']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.ffill(inplace=True)

# Features and target columns
input_features = [
    'workout_duration', 'avg_hr', 'active_energy', 'exercise_time',
    'steps', 'distance', 'resting_hr', 'sleep_duration',
    'training_load', 'workout_intensity', 'vo2_max', 'hrv', 'mindfulness_minutes'
]
output_features = ['TSB', 'ATL', 'CTL', 'reason_1', 'reason_2', 'reason_3']

# Normalize inputs
scaler = MinMaxScaler()
data[input_features] = scaler.fit_transform(data[input_features])

# Update sequence length to 42 days
sequence_length = 42  # Change from 7 to 42

def create_sequences(data, input_features, output_features, sequence_length):
    X, y = [], []
    for user_id in data['user_id'].unique():
        user_data = data[data['user_id'] == user_id].reset_index(drop=True)
        for i in range(len(user_data) - sequence_length):
            seq_x = user_data.loc[i:i+sequence_length-1, input_features].values
            seq_y = user_data.loc[i+sequence_length, output_features].values
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

X, y = create_sequences(data, input_features, output_features, sequence_length)

# Print shapes for debugging
print(f"X shape: {X.shape}, y shape: {y.shape}")  # Should be (samples, 42, 13)

# Split into training and test sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Ensure correct shape before training
assert X_train.shape[1:] == (42, 13), f"Unexpected input shape: {X_train.shape}"

# Build the transformer model using Functional API
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    
    x = x + x_ff
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# Model architecture
input_layer = Input(shape=(sequence_length, len(input_features)))

x = transformer_encoder(input_layer, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

x = GlobalAveragePooling1D()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)

output_layer = Dense(len(output_features))(x)  

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), 
    loss='mse', 
    metrics=['mae']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32
)

# Save model in TensorFlow SavedModel format
tf.saved_model.save(model, "readiness_prediction_model")

# Convert the SavedModel to CoreML
input_shape = (1, sequence_length, len(input_features))

coreml_model = ct.convert(
    "readiness_prediction_model",
    source="tensorflow",
    inputs=[ct.TensorType(shape=input_shape)],
    minimum_deployment_target=ct.target.iOS14
)
coreml_model.save("readiness_prediction_model.mlpackage")