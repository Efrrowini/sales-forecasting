import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Year']        = df['Date'].dt.year
    df['Month']       = df['Date'].dt.month
    df['DayOfMonth']  = df['Date'].dt.day
    df['Week']        = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter']     = df['Date'].dt.quarter
    df['IsWeekend']   = (df['DayOfWeek'] >= 6).astype(int)
    df['IsMonthStart'] = (df['DayOfMonth'] <= 5).astype(int)
    df['IsMonthEnd']   = (df['DayOfMonth'] >= 25).astype(int)
    return df


def add_lag_features(df: pd.DataFrame,
                      lags: list = [7, 14, 21, 28]) -> pd.DataFrame:
    """Sales N days ago for each store — captures weekly patterns."""
    df = df.sort_values(['Store', 'Date']).copy()
    for lag in lags:
        df[f'Sales_lag_{lag}'] = (
            df.groupby('Store')['Sales'].shift(lag)
        )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std — captures recent sales trend per store."""
    df = df.sort_values(['Store', 'Date']).copy()
    grp = df.groupby('Store')['Sales']
    for w in [7, 14, 30]:
        df[f'Sales_roll_mean_{w}'] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'Sales_roll_std_{w}'] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # StateHoliday has mixed types — fix first
    df['StateHoliday'] = df['StateHoliday'].astype(str).map(
        {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    ).fillna(0).astype(int)
    df['StoreType']  = df['StoreType'].map(
        {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    ).fillna(0).astype(int)
    df['Assortment'] = df['Assortment'].map(
        {'a': 0, 'b': 1, 'c': 2}
    ).fillna(0).astype(int)
    return df


def build_features(
    train_path: str = 'data/raw/train.csv',
    store_path: str = 'data/raw/store.csv',
    out_path:   str = 'data/processed/features.csv'
) -> pd.DataFrame:

    print('Loading data...')
    train = pd.read_csv(train_path, parse_dates=['Date'],
                        low_memory=False)
    store = pd.read_csv(store_path)
    df = train.merge(store, on='Store', how='left')

    # Filter closed / zero-sales days
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()
    df = df.drop(columns=['Customers', 'Open'])

    print('Encoding categoricals...')
    df = encode_categoricals(df)

    print('Adding calendar features...')
    df = add_calendar_features(df)

    print('Adding lag features (slowest step)...')
    df = add_lag_features(df)

    print('Adding rolling features...')
    df = add_rolling_features(df)

    # Fill NaN from lags at start of each store history
    df = df.fillna(0)

    print(f'Saving to {out_path}...')
    df.to_csv(out_path, index=False)

    print(f'Done! Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    return df


if __name__ == '__main__':
    build_features()