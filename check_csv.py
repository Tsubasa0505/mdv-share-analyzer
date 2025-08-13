import pandas as pd

# Shift-JISエンコーディングでCSVを読み込み
df = pd.read_csv('ダミーデータ（CSV）.csv', encoding='shift-jis')

print("列名:")
for col in df.columns:
    print(f"  - {col}")

print(f"\nデータ形状: {df.shape}")
print("\nデータ型:")
print(df.dtypes)