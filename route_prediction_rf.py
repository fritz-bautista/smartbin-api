from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pymysql
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()  # load DB credentials from .env

app = FastAPI(title="SmartBin Route Prediction API")

# Request model
class PredictionRequest(BaseModel):
    bin_id: int
    days_ahead: int = 7  # predict for next N days
    bin_max: float = 50  # max capacity of bin in kg

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

# Predict future weights and overflow probability
def predict_overflow(df, days_ahead=7, bin_max=50):
    if df.empty:
        return [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "predicted_overflow": 0.0,
                "collection_needed": False,
                "predicted_weight": 0.0,
                "predicted_level": 0.0
            }
            for i in range(days_ahead)
        ]

    # Calculate historical daily increments (realistic)
    df_sorted = df.sort_values('created_at')
    df_sorted['diff'] = df_sorted['weight'].diff().fillna(0)
    daily_increment = float(df_sorted['diff'].mean())
    
    # Take last known values
    last_weight = float(df_sorted['weight'].iloc[-1])
    
    predictions = []
    for i in range(1, days_ahead + 1):
        # Predict next day weight using historical daily increment
        future_weight = min(last_weight + daily_increment, bin_max)
        future_level = (future_weight / bin_max) * 100

        # Overflow probability is 1 if near full, else scaled 0-1
        overflow_probability = min(future_level / 100, 1.0)
        collection_needed = overflow_probability >= 0.8  # Only collect if 80% full

        predictions.append({
            "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_overflow": round(overflow_probability * 100, 2),  # as %
            "collection_needed": collection_needed,
            "predicted_weight": round(future_weight, 2),
            "predicted_level": round(future_level, 2)
        })

        last_weight = future_weight

    return predictions

@app.post("/predict")
def route_prediction(req: PredictionRequest):
    try:
        df = fetch_bin_data(req.bin_id)
        predictions = predict_overflow(df, req.days_ahead, req.bin_max)
        return {"bin_id": req.bin_id, "predictions": predictions}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"bin_id": req.bin_id, "predictions": [], "error": str(e)}
