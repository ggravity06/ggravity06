
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import pandas as pd
import os
import pickle
import numpy as np

# ---- แพตช์ NumPy ให้มี np.NaN เพื่อไม่ให้ lib เก่าพังกับ NumPy 2 ----
if not hasattr(np, "NaN"):
    np.NaN = np.nan

from saleluzaa.pipeline_utils import (
    clean_raw_data,
    build_daily_from_raw,
    add_features,
    FEATURE_COLS,
)

app = FastAPI(title="Coffee Sales FLAML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ภายหลังค่อยล็อก origin ให้เหลือโดเมนเว็บเรา
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- โหลดโมเดลจากไฟล์ในโฟลเดอร์ saleluzaa ----- #
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "coffee_sales_flaml.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Coffee Sales Prediction API is running!"}


@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)):
    df_raw = pd.read_csv(file.file)

    # 1) clean raw
    df_clean = clean_raw_data(df_raw)

    # 2) aggregate daily
    df_daily = build_daily_from_raw(df_clean)

    # 3) feature engineering
    df_feat = add_features(df_daily)

    # 4) เลือกฟีเจอร์เหมือนตอน train
    X_new = df_feat[FEATURE_COLS]

    # 5) predict
    preds = model.predict(X_new)

    df_feat["Predicted_Amount_of_Sale"] = preds

    return df_feat[["Date", "Menu_Name", "Amount_of_Sale", "Predicted_Amount_of_Sale"]].to_dict(
        orient="records"
    )
