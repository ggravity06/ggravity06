
import pandas as pd

# ฟีเจอร์ที่ใช้เทรนโมเดล (ต้องตรงกับตอน train ใน notebook)
FEATURE_COLS = [
    "Weeksort",
    "Monthsort",
    "Menu_Name",
    "Total_Revenue",
    "day_number",
    "sale_yesterday",
    "sale_last_week",
    "is_weekend",
]

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    ทำความสะอาด raw coffee sales data

    คาดว่ามีคอลัมน์ประมาณ:
      - Date
      - Time
      - coffee_name
      - money
      - Weekdaysort (optional)
      - Monthsort (optional)
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # แปลง Date เป็น datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
    else:
        raise ValueError("Raw data ต้องมีคอลัมน์ 'Date'")

    # แปลงเงิน
    if "money" in df.columns:
        df["money"] = (
            df["money"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .astype(float)
        )
    else:
        raise ValueError("Raw data ต้องมีคอลัมน์ 'money'")

    # ทำชื่อเมนู
    if "coffee_name" in df.columns:
        df["coffee_name"] = df["coffee_name"].astype(str).str.strip()
    else:
        raise ValueError("Raw data ต้องมีคอลัมน์ 'coffee_name'")

    # เติม Weekdaysort / Monthsort ถ้ายังไม่มี
    if "Weekdaysort" not in df.columns:
        df["Weekdaysort"] = df["Date"].dt.weekday + 1  # 1–7
    if "Monthsort" not in df.columns:
        df["Monthsort"] = df["Date"].dt.month

    return df


def build_daily_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    aggregate เป็นรายวัน/เมนู:
    Date, Weeksort, Monthsort, Menu_Name, Amount_of_Sale, Total_Revenue
    """
    daily = (
        raw_df
        .groupby(["Date", "Weekdaysort", "Monthsort", "coffee_name"])
        .agg(
            Amount_of_Sale=("coffee_name", "count"),
            Total_Revenue=("money", "sum"),
        )
        .reset_index()
    )

    daily = daily.rename(columns={
        "coffee_name": "Menu_Name",
        "Weekdaysort": "Weeksort",
        "Monthsort": "Monthsort",
    })

    return daily


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่มฟีเจอร์:
      - day_number
      - sale_yesterday
      - sale_last_week
      - is_weekend
    """
    df = daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # เรียงตามเมนู + วันที่ก่อน
    df = df.sort_values(["Menu_Name", "Date"])

    # day_number
    first_day = df["Date"].min()
    df["day_number"] = (df["Date"] - first_day).dt.days

    # sale_yesterday
    df["sale_yesterday"] = df.groupby("Menu_Name")["Amount_of_Sale"].shift(1)

    # sale_last_week (7 วันก่อน)
    df["sale_last_week"] = df.groupby("Menu_Name")["Amount_of_Sale"].shift(7)

    # เติม NaN ด้วย 0
    df[["sale_yesterday", "sale_last_week"]] = df[["sale_yesterday", "sale_last_week"]].fillna(0)

    # is_weekend จาก Weeksort
    df["is_weekend"] = df["Weeksort"].isin([6, 7]).astype(int)

    return df
