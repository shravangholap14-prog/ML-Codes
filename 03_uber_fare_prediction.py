"""
Practical 3: Uber Ride Fare Prediction using Linear, Ridge and Lasso Regression.

Optional real dataset:
Download an Uber fare dataset CSV (for example, the Kaggle "uber.csv") and place it
beside this script as uber.csv. It should contain fare_amount, pickup_longitude,
pickup_latitude, dropoff_longitude, dropoff_latitude and optionally passenger_count
and pickup_datetime.

If uber.csv is not found, a small synthetic demonstration dataset is generated so
the practical runs immediately. Synthetic results are for demonstration only.
Run: python 03_uber_fare_prediction.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def load_data():
    path = "uber.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        required = ["fare_amount", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"uber.csv is missing required columns: {missing}")
        print("Using real dataset:", path)
    else:
        print("uber.csv not found. Using synthetic demonstration data.")
        rng = np.random.default_rng(42)
        n = 1200
        plon = rng.uniform(-74.05, -73.75, n)
        plat = rng.uniform(40.60, 40.90, n)
        dlon = plon + rng.normal(0, 0.025, n)
        dlat = plat + rng.normal(0, 0.025, n)
        distance = haversine_km(plat, plon, dlat, dlon)
        passengers = rng.integers(1, 5, n)
        fare = 3.0 + 2.2 * distance + 0.35 * passengers + rng.normal(0, 1.2, n)
        df = pd.DataFrame({
            "fare_amount": np.maximum(fare, 2.5),
            "pickup_longitude": plon, "pickup_latitude": plat,
            "dropoff_longitude": dlon, "dropoff_latitude": dlat,
            "passenger_count": passengers
        })

    df = df.copy()
    df = df.dropna(subset=["fare_amount"])
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]
    for col in ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]:
        df = df[df[col].between(-180 if "longitude" in col else -90,
                                180 if "longitude" in col else 90)]
    df["distance_km"] = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"])
    if "passenger_count" not in df:
        df["passenger_count"] = 1
    if "pickup_datetime" in df:
        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["pickup_hour"] = dt.dt.hour
        df["pickup_day"] = dt.dt.dayofweek
    else:
        df["pickup_hour"] = 12
        df["pickup_day"] = 0
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def main():
    df = load_data()
    features = ["distance_km", "passenger_count", "pickup_hour", "pickup_day"]
    X, y = df[features], df["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000)
    }
    results = {}
    for name, model in models.items():
        pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        }
        print(f"\n{name}")
        print(f"MAE:  {results[name]['MAE']:.3f}")
        print(f"RMSE: {results[name]['RMSE']:.3f}")
        print(f"R2:   {results[name]['R2']:.3f}")

    pd.DataFrame(results).T.plot(kind="bar", figsize=(9, 5))
    plt.title("Uber Fare Prediction: Model Metric Comparison")
    plt.ylabel("Metric value (R2 is on a different scale)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
