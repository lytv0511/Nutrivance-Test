import numpy as np
import pandas as pd
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

    # Compute ATL and CTL
    df["ATL"] = df.groupby("user_id")["training_load"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["CTL"] = df.groupby("user_id")["training_load"].transform(lambda x: x.rolling(42, min_periods=1).mean())
    df["TSB"] = df["CTL"] - df["ATL"]

    # Extract outputs (TSB + top 3 reasons) per user
    outputs = []
    
    for user_id, user_data in df.groupby("user_id"):
        last_day = user_data.iloc[-1]  # Last day's data (Day 42)
        
        # Contributions for top 3 reasons
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

    # Label encode reason columns
    for col in ["reason_1", "reason_2", "reason_3"]:
        le = LabelEncoder()
        output_df[col] = le.fit_transform(output_df[col])

        with open(f"{col}_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    # Merge outputs back into the main dataframe (outputs only for last day of each user)
    merged_df = df.merge(output_df, on="user_id")
    return merged_df

def generate_dataset():
    all_data = []

    for user_id in range(1, NUM_USERS + 1):
        consistent = np.random.rand() < CONSISTENT_USER_RATIO
        user_data = generate_user_data(user_id, consistent)
        all_data.extend(user_data)

        if user_id % 500 == 0:
            print(f"Generated data for {user_id} users...")

    daily_data = pd.DataFrame(all_data)
    result = compute_loads(daily_data)
    
    return result

def main():
    print("Generating dataset...")
    dataset = generate_dataset()
    
    # Save to CSV; one row per day per user, outputs only for the last day
    dataset.to_csv("synthetic_health_data.csv", index=False)
    print("Dataset saved to 'synthetic_health_data.csv'.")

if __name__ == "__main__":
    main()