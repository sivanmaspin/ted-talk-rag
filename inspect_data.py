import pandas as pd

df = pd.read_csv('ted_talks_en.csv', nrows=5)

print("--- Columns ---")
print(df.columns.tolist())
print("\n--- First Talk Transcript Preview ---")
print(df.iloc[0]['transcript'][:500] + "...")