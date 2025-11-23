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
def predict_overflow(df, days_ahead=7):
    import numpy as np

    if df.empty:
        return [
            {"date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
             "predicted_overflow": 0.0,
             "collection_needed": False}
            for i in range(days_ahead)
        ]

    df['day'] = pd.to_datetime(df['created_at']).dt.dayofyear
    X = df[['day', 'weight']].values
    y = (df['weight'] >= 50).astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    df = df.sort_values('created_at')
    df['diff'] = df['weight'].diff().fillna(0)
    increments = df['diff'].rolling(7, min_periods=1).mean().tolist()
    last_weight = df['weight'].iloc[-1]
    last_day = df['day'].iloc[-1]

    predictions = []
    for i in range(1, days_ahead + 1):
        future_day = last_day + i
        increment = increments[-1] if increments else 0
        noise = np.random.uniform(-0.5, 0.5)
        future_weight = min(max(last_weight + increment + noise, 0), 50)

        # Safe overflow probability
        if model.classes_.size == 1:
            pred_prob = 0.0 if model.classes_[0] == 0 else 1.0
        else:
            pred_prob = float(model.predict_proba([[future_day, future_weight]])[0][1])

        predictions.append({
            "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_overflow": pred_prob,
            "collection_needed": pred_prob >= 0.5
        })

        last_weight = future_weight
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


