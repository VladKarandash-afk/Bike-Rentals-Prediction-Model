import pandas as pd
import numpy as np
import holidays

from encoding import encoding

# FUNCTION FOR ADDING FEATURES

def preprocessing(data):
    df = pd.DataFrame(data)
    pl_holidays = holidays.PL()

    df['date'] = pd.to_datetime(df['date'])
    df['weekend'] = (df['date'].dt.dayofweek > 4) | (df['date'].isin(pl_holidays))
    df['weekend'] = df['weekend'].astype('int')
    cat_weather_type = pd.CategoricalDtype(categories=sorted(df['wcond'].dropna().unique()), ordered=True)
    df['wcond'] = df['wcond'].astype(cat_weather_type)

    df = df.replace(r'^\s*$', np.nan, regex=True)
    empty_rows_count = df.isna().any(axis=1).sum()
    print(f"Rows with empty values: {empty_rows_count}\n")

    df["precipitation"] = df["wcond"] == 3
    conditions = [
        (df['atemp'] < 10),
        (df['atemp'] >= 10) & (df['atemp'] <= 25),
        (df['atemp'] > 25)
    ]
    choices = [1, 2, 3]
    df['temp_rank'] = np.select(conditions, choices, default=0)

    for i in range(1, 4):
        df[f'count_lag_{i}'] = df['count'].shift(i)

    df['wcond_lag1'] = df['wcond'].shift(1)
    df['atemp_lag1'] = df['temp'].shift(1)
    df = df.dropna()
    df['extreme_wcond'] = (df['wind'] > df['wind'].quantile(0.9)) | (df['hum'] > df['hum'].quantile(0.9)) | df[
        'precipitation']
    df['perfect_temp'] = (df['atemp'] >= 18) & (df['atemp'] <= 25)
    df['temp_diff'] = df['atemp'] - df['atemp_lag1']
    wcond_codes = df['wcond'].astype('float')
    wcond_lag1_codes = df['wcond_lag1'].astype('float')
    df['weather_improved'] = (wcond_codes - wcond_lag1_codes) < 0
    df['weekend_and_bad_weather'] = df['weekend'].astype('int') * df['precipitation'].astype('int')

    month = df['date'].dt.month
    conditions = [
        (month == 12) | (month <= 2),
        (month >= 3) & (month <= 5),
        (month >= 6) & (month <= 8),
        (month >= 9) & (month <= 11)
    ]
    choices = [1, 2, 3, 4]
    df['month'] = pd.CategoricalDtype(categories=sorted(month.dropna().unique()), ordered=True)
    df['season'] = np.select(conditions, choices, default=np.nan)

    indices = df.index + 1
    df["day_number"] = pd.CategoricalDtype(categories=sorted(indices.dropna().unique()), ordered=True)

    encoded_data = encoding(df)

    return encoded_data