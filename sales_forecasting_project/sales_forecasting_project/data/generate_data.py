"""
generate_data.py
----------------
Generates a realistic synthetic daily e-commerce sales dataset with:
  - Long-term upward trend
  - Weekly seasonality (weekend spikes)
  - Yearly seasonality (holiday season boost)
  - Promotional spikes (random flash sales + fixed campaigns like BFCM, New Year)
  - Random noise

Output: data/sales_data.csv
Columns: date, sales, units_sold, is_promotion, is_holiday_season
"""

import numpy as np
import pandas as pd

np.random.seed(42)

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


def generate_sales_data(start=START_DATE, end=END_DATE):
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)
    t = np.arange(n)

    # 1. Base level + long-term trend (business growing ~25% over 3 years)
    base_level = 1000
    trend = base_level + t * 0.35

    # 2. Weekly seasonality: weekends (Fri/Sat/Sun) sell more
    weekday = dates.dayofweek  # Mon=0 ... Sun=6
    weekly_pattern = np.where(np.isin(weekday, [4, 5, 6]), 1.25, 1.0)

    # 3. Yearly seasonality: smooth sinusoid + strong Nov-Dec holiday boost
    day_of_year = dates.dayofyear
    yearly_wave = 1 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    holiday_boost = np.where(dates.month.isin([11, 12]), 1.35, 1.0)
    holiday_boost = np.where(
        (dates.month == 12) & (dates.day.isin(range(20, 32))), 1.6, holiday_boost
    )

    # 4. Promotions: fixed campaigns + random flash sales
    is_promotion = np.zeros(n, dtype=int)
    fixed_promo_windows = [
        ("2022-11-24", "2022-11-28"),  # Black Friday / Cyber Monday
        ("2022-12-31", "2023-01-01"),  # New Year
        ("2023-07-10", "2023-07-12"),  # Summer flash sale (Prime Day style)
        ("2023-11-24", "2023-11-27"),
        ("2023-12-31", "2024-01-01"),
        ("2024-07-08", "2024-07-10"),
        ("2024-11-29", "2024-12-02"),
        ("2024-12-31", "2025-01-01"),
    ]
    for s, e in fixed_promo_windows:
        mask = (dates >= s) & (dates <= e)
        is_promotion[mask] = 1

    # Random surprise flash sales (~1.5% of days)
    random_promo_idx = np.random.choice(n, size=int(n * 0.015), replace=False)
    is_promotion[random_promo_idx] = 1

    promo_multiplier = np.where(is_promotion == 1, 1.5 + np.random.uniform(0, 0.3, n), 1.0)

    # 5. Combine all components
    sales = trend * weekly_pattern * yearly_wave * holiday_boost * promo_multiplier

    # 6. Add realistic noise (multiplicative + a little additive)
    noise = np.random.normal(loc=1.0, scale=0.05, size=n)
    sales = sales * noise
    sales = np.maximum(sales, 0)  # no negative sales

    # Derive units sold from an average order value that drifts slightly over time
    avg_order_value = 45 + 5 * np.sin(2 * np.pi * t / 365.25) + np.random.normal(0, 1.5, n)
    avg_order_value = np.clip(avg_order_value, 20, None)
    units_sold = np.round(sales / avg_order_value).astype(int)

    is_holiday_season = dates.month.isin([11, 12]).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": np.round(sales, 2),
            "units_sold": units_sold,
            "is_promotion": is_promotion,
            "is_holiday_season": is_holiday_season,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    out_path = "data/sales_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
    print(df.describe())
