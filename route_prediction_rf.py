from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pymysql
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # Load DB credentials from .env

app = FastAPI(title="SmartBin Route Prediction API")

# Request model
class PredictionRequest(BaseModel):
    bin_id: int
    days_ahead: int = 7  # predict next N days
    bin_max: float = 50  # max bin capacity (kg)

# DB connection
def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE"),
        port=int(os.getenv("DB_PORT"))
    )

# Fetch bin data
def fetch_bin_data(bin_id):
    conn = get_db_connection()
    query = f"""
        SELECT weight, level, created_at 
        FROM waste_levels 
        WHERE bin_id={bin_id} 
        ORDER BY created_at
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Predict future weights, levels, and overflow probability
def predict_overflow(df, days_ahead=7, bin_max=50):
    if df.empty:
        return [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "predicted_weight": 0.0,
                "predicted_level": 0.0,
                "predicted_overflow": 0.0,
                "collection_needed": False
            }
            for i in range(days_ahead)
        ]

    df['created_at'] = pd.to_datetime(df['created_at'])
    df_sorted = df.sort_values('created_at')

    # Aggregate per day (use last reading per day)
    daily_df = df_sorted.groupby(df_sorted['created_at'].dt.date).agg({'weight': 'last', 'level': 'last'}).reset_index()

    # Daily increments (median to avoid spikes)
    weight_inc = daily_df['weight'].diff().median()
    level_inc = daily_df['level'].diff().median()

    weight_inc = float(weight_inc if not pd.isna(weight_inc) else 0)
    level_inc = float(level_inc if not pd.isna(level_inc) else 0)

    last_weight = float(daily_df['weight'].iloc[-1])
    last_level = float(daily_df['level'].iloc[-1])

    predictions = []
    for i in range(1, days_ahead + 1):
        # Predict future weight and level
        future_weight = min(max(last_weight + weight_inc, 0), bin_max)
        future_level = min(max(last_level + level_inc, 0), 100)  # percent

        overflow_probability = min(future_level / 100, 1.0)  # 0–1
        collection_needed = overflow_probability >= 0.8

        predictions.append({
            "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_weight": round(future_weight, 2),
            "predicted_level": round(future_level, 2),
            "predicted_overflow": round(overflow_probability * 100, 2),
            "collection_needed": collection_needed
        })

        last_weight = future_weight
        last_level = future_level

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
