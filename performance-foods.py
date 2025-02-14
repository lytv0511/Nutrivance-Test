import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import coremltools as ct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_v2 = pd.read_csv("performance_food_ml_dataset_v2.csv")
df_v3 = pd.read_csv("performance_food_ml_dataset_v3.csv")
df_v1 = pd.read_csv("performance_food_ml_dataset.csv")

df = pd.concat([df_v2, df_v3, df_v1], ignore_index=True)

feature_columns = ["workout_type", "duration_planned", "intensity_level", "time_of_day", "previous_workout_strain", 
                   "current_macronutrients_carbs", "current_macronutrients_proteins", "current_macronutrients_fats", 
                   "hydration_status", "recent_meal_timing", "caloric_balance", "micronutrient_levels_calcium", 
                   "micronutrient_levels_iron", "micronutrient_levels_vitamin_d", "recovery_score", "sleep_quality", 
                   "heart_rate_variability", "body_fat_percentage", "lean_mass_kg", "training_age", 
                   "food_category", "timing_window", "portion_size_macro_ratio", "portion_size_total_calories", 
                   "portion_size_nutrient_timing", "portion_size_hydration"]

label_columns = ["food_category", "timing_window", "portion_size_macro_ratio", "portion_size_total_calories", "portion_size_nutrient_timing", "portion_size_hydration"]

df['workout_type'] = df['workout_type'].astype('category').cat.codes
df['time_of_day'] = df['time_of_day'].astype('category').cat.codes
df['recent_meal_timing'] = df['recent_meal_timing'].astype('category').cat.codes
df['food_category'] = df['food_category'].astype('category').cat.codes
df['timing_window'] = df['timing_window'].astype('category').cat.codes
df['portion_size_nutrient_timing'] = df['portion_size_nutrient_timing'].astype('category').cat.codes

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

preprocessor = StandardScaler()
X_train = preprocessor.fit_transform(train_df[feature_columns])
X_test = preprocessor.transform(test_df[feature_columns])
y_train = train_df[label_columns].values
y_test = test_df[label_columns].values

model = keras.Sequential([
    layers.Input(shape=(len(feature_columns),)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_columns), activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=32)

eval_results = model.evaluate(X_test, y_test)
print("Test Loss, Test MAE:", eval_results)

tf.saved_model.save(model, "movement_prediction_sequential_model")

coreml_model = ct.convert("movement_prediction_sequential_model", source="tensorflow", inputs=[ct.TensorType(shape=(1, len(feature_columns)))], minimum_deployment_target=ct.target.iOS14)
coreml_model.save("movement_prediction_sequential_model.mlpackage")
