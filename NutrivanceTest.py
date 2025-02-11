import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import coremltools as ct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("health_nutrition_dataset_refined.csv")

# Define feature and label columns
feature_columns = ["age", "tdee", "steps", "walking_running_minutes", "flights_climbed", "cardio_vo2max", "cardio_recovery_bpm"]
label_columns = ["protein_needed_g", "fats_needed_g", "carbs_needed_g", "water_needed_liters"]

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Standardize the features
preprocessor = StandardScaler()
X_train = preprocessor.fit_transform(train_df[feature_columns])
X_test = preprocessor.transform(test_df[feature_columns])
y_train = train_df[label_columns].values
y_test = test_df[label_columns].values

# Define the model
preprocessor_layer = keras.layers.Lambda(
    lambda x: (x - preprocessor.mean_) / preprocessor.scale_
)

full_model = keras.Sequential([
    preprocessor_layer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_columns), activation='linear')
])

# Compile the model
full_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = full_model.fit(
    train_df[feature_columns].values, 
    y_train,
    validation_data=(test_df[feature_columns].values, y_test),
    epochs=500,
    batch_size=32
)

# Evaluate the model
eval_results = full_model.evaluate(test_df[feature_columns].values, y_test)
print("Test Loss, Test MAE:", eval_results)

# Input values for prediction
input_data = np.array([[28.0, 2400.0, 12000.0, 45.0, 15.0, 45.0, 65.0]])

# Standardize the input data
input_data_standardized = preprocessor.transform(input_data)

# Make a prediction
predicted_output = full_model.predict(input_data_standardized)

# Create DataFrames for input and output
input_df = pd.DataFrame(input_data, columns=feature_columns)
predicted_df = pd.DataFrame(predicted_output, columns=label_columns)

print("Input Data:")
print(input_df)

print("\nPredicted Output:")
print(predicted_df)

# Save the model in TensorFlow SavedModel format
tf.saved_model.save(full_model, "nutrition_prediction_model")

# Convert the SavedModel to CoreML
coreml_model = ct.convert(
    "nutrition_prediction_model",
    source="tensorflow",
    inputs=[ct.TensorType(shape=(1, len(feature_columns)))],
    minimum_deployment_target=ct.target.iOS14
)
coreml_model.save("nutrition_prediction_model.mlpackage")