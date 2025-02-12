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

print(df.isnull().sum())

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

preprocessor_layer = layers.Lambda(lambda x: (x - preprocessor.mean_) / preprocessor.scale_)

def resnet_block(x, units):
    shortcut = layers.Dense(units)(x)
    x = layers.Dense(units, activation='relu')(x)
    x = layers.Dense(units)(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

input_layer = layers.Input(shape=(len(feature_columns),))
x = preprocessor_layer(input_layer)
x = resnet_block(x, 128)
x = layers.Dropout(0.2)(x)
x = resnet_block(x, 128)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(len(label_columns), activation='linear')(x)

full_model = keras.Model(inputs=input_layer, outputs=output)
full_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = full_model.fit(train_df[feature_columns].values, y_train, validation_data=(test_df[feature_columns].values, y_test), epochs=500, batch_size=32)

eval_results = full_model.evaluate(test_df[feature_columns].values, y_test)
print("Test Loss, Test MAE:", eval_results)

input_data = np.array([[28.0, 30.0, 5, 2, 10.0, 200.0, 100.0, 50.0, 80.0, 2, 350.0, 150.0, 20.0, 75.0, 80.0, 55.0, 30.0, 15.0, 22.0, 12.0, 5, 1, 4, 3.0, 2.0, 1.5]])
input_data_standardized = preprocessor.transform(input_data)
predicted_output = full_model.predict(input_data_standardized)

input_df = pd.DataFrame(input_data, columns=feature_columns)
predicted_df = pd.DataFrame(predicted_output, columns=label_columns)

print("Input Data:")
print(input_df)
print("\nPredicted Output:")
print(predicted_df)

tf.saved_model.save(full_model, "nutrition_prediction_model")

coreml_model = ct.convert("nutrition_prediction_model", source="tensorflow", inputs=[ct.TensorType(shape=(1, len(feature_columns)))], minimum_deployment_target=ct.target.iOS14)
coreml_model.save("movement_prediction_model.mlpackage")