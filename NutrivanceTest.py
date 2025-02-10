import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import coremltools as ct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("health_nutrition_dataset_refined.csv")

feature_columns = ["age", "tdee", "steps", "walking_running_minutes", "flights_climbed", "cardio_vo2max", "cardio_recovery_bpm"]
label_columns = ["tdee", "protein_needed_g", "fats_needed_g", "carbs_needed_g", "water_needed_liters"]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

preprocessor = StandardScaler()
X_train = preprocessor.fit_transform(train_df[feature_columns])
X_test = preprocessor.transform(test_df[feature_columns])
y_train = train_df[label_columns].values
y_test = test_df[label_columns].values

preprocessor_layer = keras.layers.Lambda(
    lambda x: (x - preprocessor.mean_) / preprocessor.scale_
)

full_model = keras.Sequential([
    preprocessor_layer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_columns), activation='linear')
])

full_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = full_model.fit(
    train_df[feature_columns].values, 
    y_train,
    validation_data=(test_df[feature_columns].values, y_test),
    epochs=1000,
    batch_size=32
)

eval_results = full_model.evaluate(test_df[feature_columns].values, y_test)
print("Test Loss, Test MAE:", eval_results)

coreml_model = ct.convert(
    full_model,
    inputs=[ct.TensorType(shape=(1, len(feature_columns)))]
)
coreml_model.save("nutrition_prediction_model.mlpackage")