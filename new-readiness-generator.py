import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

np.random.seed(42)

NUM_USERS = 5000
NUM_DAYS = 42
CONSISTENT_USER_RATIO = 0.7

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

def generate_user_data(user_id, consistent=True):
    user_data = []
    base_values = {
        metric: np.random.uniform(min_val, max_val)
        for metric, (_, min_val, max_val, _) in METRICS.items()
    }

    for day in range(NUM_DAYS):
        daily_data = {"user_id": user_id, "day": day + 1}
        for metric, (unit, min_val, max_val, is_avg) in METRICS.items():
            if consistent:
                variation = np.random.normal(0, 0.05 * (max_val - min_val))
                value = np.clip(base_values[metric] + variation, min_val, max_val)
            else:
                variation = np.random.normal(0, 0.3 * (max_val - min_val))
                value = np.clip(np.random.uniform(min_val, max_val) + variation, min_val, max_val)

            value = round(value, 1) if is_avg else int(value)
            daily_data[metric] = value

        user_data.append(daily_data)
    
    return user_data

def compute_loads(df):
    df = df.sort_values(by=["user_id", "day"]).reset_index(drop=True)

    df["ATL"] = df.groupby("user_id")["training_load"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["CTL"] = df.groupby("user_id")["training_load"].transform(lambda x: x.rolling(42, min_periods=1).mean())
    df["TSB"] = df["CTL"] - df["ATL"]

    outputs = []
    
    for user_id, user_data in df.groupby("user_id"):
        last_day = user_data.iloc[-1]
        
        contributions = {}
        for metric in METRICS:
            value = last_day[metric]
            min_val, max_val = METRICS[metric][1], METRICS[metric][2]
            midpoint = (min_val + max_val) / 2
            contributions[metric] = value - midpoint

        top_reasons = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        reason_1, reason_2, reason_3 = [metric for metric, _ in top_reasons]
        
        outputs.append({
            "user_id": user_id,
            "TSB": last_day["TSB"],
            "reason_1": reason_1,
            "reason_2": reason_2,
            "reason_3": reason_3
        })

    output_df = pd.DataFrame(outputs)

    for col in ["reason_1", "reason_2", "reason_3"]:
        le = LabelEncoder()
        output_df[col] = le.fit_transform(output_df[col])

        with open(f"{col}_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    merged_df = df.merge(output_df, on="user_id")
    return merged_df

def generate_sequence_dataset(window_size=7, stride=1):
    daily_data = pd.DataFrame([])
    for user_id in range(1, NUM_USERS + 1):
        consistent = np.random.rand() < CONSISTENT_USER_RATIO
        user_data = generate_user_data(user_id, consistent)
        daily_data = pd.concat([daily_data, pd.DataFrame(user_data)])
        
        if user_id % 500 == 0:
            print(f"Generated data for {user_id} users...")

    processed_data = compute_loads(daily_data)
    
    features = []
    labels = []
    
    for user_id, user_data in processed_data.groupby('user_id'):
        user_sequence = user_data.drop(['user_id', 'day', 'reason_1', 'reason_2', 'reason_3'], axis=1).values
        
        for i in range(0, len(user_sequence) - window_size, stride):
            window = user_sequence[i:i + window_size]
            target = user_sequence[i + window_size]
            
            features.append(window)
            labels.append(target)
    
    features = np.array(features)
    labels = np.array(labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    tf_writer = tf.io.TFRecordWriter('training_data.tfrecord')
    
    for x, y in dataset:
        feature = {
            'input_sequence': tf.train.Feature(
                float_list=tf.train.FloatList(value=x.numpy().flatten())),
            'target_sequence': tf.train.Feature(
                float_list=tf.train.FloatList(value=y.numpy()))
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_writer.write(example.SerializeToString())
    
    tf_writer.close()
    
    metadata = {
        'window_size': window_size,
        'stride': stride,
        'feature_shape': features.shape,
        'label_shape': labels.shape,
        'metrics': list(METRICS.keys())
    }
    
    with open('dataset_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    return dataset

def build_hybrid_model(window_size, num_features):
    inputs = Input(shape=(window_size, num_features))
    
    # Improved CNN branch with residual connections
    conv_branch = Conv1D(128, kernel_size=3, activation='relu', padding='same')(inputs)
    conv_branch = Conv1D(64, kernel_size=3, activation='relu', padding='same')(conv_branch)
    conv_branch = Conv1D(32, kernel_size=3, activation='relu', padding='same')(conv_branch)
    
    # Enhanced transformer branch
    trans_branch = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    trans_branch = LayerNormalization(epsilon=1e-6)(trans_branch)
    
    merged = Concatenate()([conv_branch, trans_branch])
    
    x = GlobalAveragePooling1D()(merged)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    outputs = Dense(num_features)(x)
    return Model(inputs=inputs, outputs=outputs)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_loss'
    )
]

def main():
    print("Generating sequence dataset...")
    dataset = generate_sequence_dataset()
    print("Dataset saved as TFRecord file with metadata.")

if __name__ == "__main__":
    main()
