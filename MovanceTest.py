import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import coremltools as ct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the datasets
df_v2 = pd.read_csv("performance_food_ml_dataset_v2.csv")
df_v3 = pd.read_csv("performance_food_ml_dataset_v3.csv")
df_v1 = pd.read_csv("performance_food_ml_dataset.csv")

# Merge the datasets into one DataFrame
df = pd.concat([df_v2, df_v3, df_v1], ignore_index=True)

# Check for any missing values or discrepancies in the dataset
print(df.isnull().sum())

# Define feature and label columns
feature_columns = ["workout_type", "duration_planned", "intensity_level", "time_of_day", "previous_workout_strain", 
                   "current_macronutrients_carbs", "current_macronutrients_proteins", "current_macronutrients_fats", 
                   "hydration_status", "recent_meal_timing", "caloric_balance", "micronutrient_levels_calcium", 
                   "micronutrient_levels_iron", "micronutrient_levels_vitamin_d", "recovery_score", "sleep_quality", 
                   "heart_rate_variability", "body_fat_percentage", "lean_mass_kg", "training_age", 
                   "food_category", "timing_window", "portion_size_macro_ratio", "portion_size_total_calories", 
                   "portion_size_nutrient_timing", "portion_size_hydration"]

# Update the label columns to match the actual output columns in the dataset
label_columns = [
    "food_category",
    "timing_window", 
    "portion_size_macro_ratio",
    "portion_size_total_calories",
    "portion_size_nutrient_timing",
    "portion_size_hydration"
]

# Handle categorical columns by encoding them (assuming some columns are categorical)
df['workout_type'] = df['workout_type'].astype('category').cat.codes
df['time_of_day'] = df['time_of_day'].astype('category').cat.codes
df['recent_meal_timing'] = df['recent_meal_timing'].astype('category').cat.codes
df['food_category'] = df['food_category'].astype('category').cat.codes
df['timing_window'] = df['timing_window'].astype('category').cat.codes
df['portion_size_nutrient_timing'] = df['portion_size_nutrient_timing'].astype('category').cat.codes

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Standardize the features
preprocessor = StandardScaler()
X_train = preprocessor.fit_transform(train_df[feature_columns])
X_test = preprocessor.transform(test_df[feature_columns])
y_train = train_df[label_columns].values
y_test = test_df[label_columns].values

# Define the preprocessor layer for input standardization
preprocessor_layer = layers.Lambda(
    lambda x: (x - preprocessor.mean_) / preprocessor.scale_
)

# Define the Residual Block
def resnet_block(x, units):
    # Project input to match output dimensions if needed
    shortcut = layers.Dense(units)(x)
    
    # Main path
    x = layers.Dense(units, activation='relu')(x)
    x = layers.Dense(units)(x)
    
    # Add shortcut connection
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Input Layer
input_layer = layers.Input(shape=(len(feature_columns),))
x = preprocessor_layer(input_layer)

# First residual block
x = resnet_block(x, 128)
x = layers.Dropout(0.2)(x)

# Second residual block
x = resnet_block(x, 128)
x = layers.Dropout(0.2)(x)

# Third hidden layer
x = layers.Dense(64, activation='relu')(x)

# Final Output Layer
output = layers.Dense(len(label_columns), activation='linear')(x)

# Build the model
full_model = keras.Model(inputs=input_layer, outputs=output)

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
# Update input_data to include all 26 features in the correct order
input_data = np.array([[
    28.0,  # workout_type
    30.0,  # duration_planned
    5,     # intensity_level
    2,     # time_of_day
    10.0,  # previous_workout_strain
    200.0, # current_macronutrients_carbs
    100.0, # current_macronutrients_proteins
    50.0,  # current_macronutrients_fats
    80.0,  # hydration_status
    2,     # recent_meal_timing
    350.0, # caloric_balance
    150.0, # micronutrient_levels_calcium
    20.0,  # micronutrient_levels_iron
    75.0,  # micronutrient_levels_vitamin_d
    80.0,  # recovery_score
    55.0,  # sleep_quality
    30.0,  # heart_rate_variability
    15.0,  # body_fat_percentage
    22.0,  # lean_mass_kg
    12.0,  # training_age
    5,     # food_category
    1,     # timing_window
    4,     # portion_size_macro_ratio
    3.0,   # portion_size_total_calories
    2.0,   # portion_size_nutrient_timing
    1.5    # portion_size_hydration
]])

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
coreml_model.save("movement_prediction_model.mlpackage")