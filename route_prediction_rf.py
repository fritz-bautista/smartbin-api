from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pymysql
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # load DB credentials from .env

app = FastAPI(title="SmartBin Route Prediction API")

# Request model
class PredictionRequest(BaseModel):
    bin_id: int
    days_ahead: int = 1  # predict for next N days

# Connect to database
def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE"),
        port=int(os.getenv("DB_PORT"))
    )

# Fetch past waste data for a bin
def fetch_bin_data(bin_id):
    conn = get_db_connection()
    query = f"SELECT weight, level, created_at FROM waste_levels WHERE bin_id={bin_id} ORDER BY created_at"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Predict overflow using Random Forest
def predict_overflow(df, days_ahead=1):
    """
    Predicts future overflow probabilities for the next `days_ahead` days.
    Uses historical weight changes to simulate realistic future weights.
    """
    from sklearn.ensemble import RandomForestClassifier
    from datetime import datetime, timedelta
    import numpy as np

    if df.empty:
        # No data → return empty predictions
        return [
            {"date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
             "predicted_overflow": 0.0,
             "collection_needed": False}
            for i in range(days_ahead)
        ]

    # --- Prepare features ---
    df['day'] = pd.to_datetime(df['created_at']).dt.dayofyear
    X = df[['day', 'weight']].values
    y = (df['weight'] >= 50).astype(int)  # overflow threshold 50kg

    # --- Train model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- Compute historical daily increments ---
    df = df.sort_values('created_at')
    df['diff'] = df['weight'].diff().fillna(0)
    
    # Use a rolling window of last 7 increments for more stability
    increments = df['diff'].rolling(7, min_periods=1).mean().tolist()

    # --- Start simulation from last known weight and day ---
    last_weight = df['weight'].iloc[-1]
    last_day = df['day'].iloc[-1]

    predictions = []
    for i in range(1, days_ahead + 1):
        future_day = last_day + i

        # Use the most recent increment from history
        increment = increments[-1] if increments else 0

        # Simulate realistic weight growth
        future_weight = last_weight + increment
        future_weight = min(max(future_weight, 0), 50)  # clamp between 0 and capacity

        # Predict overflow probability
        pred_prob = model.predict_proba([[future_day, future_weight]])[0][1]

        predictions.append({
            "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_overflow": float(pred_prob),
            "collection_needed": pred_prob >= 0.5
        })

        # Update last_weight for next day simulation
        last_weight = future_weight

        # Optionally update increments list if you want dynamic simulation
        increments.append(increment)

    return predictions


@app.post("/predict")
def route_prediction(req: PredictionRequest):
    try:
        df = fetch_bin_data(req.bin_id)
        predictions = predict_overflow(df, req.days_ahead)
        return {"bin_id": req.bin_id, "predictions": predictions}
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Always return predictions key, even if empty
        return {"bin_id": req.bin_id, "predictions": [], "error": str(e)}


