import pandas as pd
import numpy as np
import json

df = pd.read_csv('data/Sample - Superstore.csv', encoding='latin-1')

print(f"Dataframe shape: {df.shape}")
print(f"Dataframe columns: {df.columns}")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df['Order Month Name'] = df['Order Date'].dt.strftime('%B')
df['Order Quarter'] = df['Order Date'].dt.quarter

for col in ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode']:
    df[col] = df[col].str.strip().str.title()

df['Profit Margin'] = df.apply(
    lambda row: round(row['Profit'] / row['Sales'],
                      4) if row['Sales'] != 0 else 0,
    axis=1
)

df.to_csv('data/superstore_clean.csv', index=False)
print("Preprocessing done. Clean file saved.")
print(f"New columns added: Order Year, Order Month, Order Quarter, Profit Margin")
