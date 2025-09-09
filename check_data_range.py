import pandas as pd

# 检查原始数据
print("=== 检查原始 deposits_withdrawals_operating_cash 数据 ===")
df = pd.read_csv('data/raw/deposits_withdrawals_operating_cash.csv')
df['record_date'] = pd.to_datetime(df['record_date'])

print(f"原始数据日期范围: {df['record_date'].min().date()} 到 {df['record_date'].max().date()}")
print(f"总记录数: {len(df):,}")
print(f"唯一日期数: {df['record_date'].dt.date.nunique():,}")
print(f"按年份统计:")
print(df['record_date'].dt.year.value_counts().sort_index())

print("\n=== 检查建模数据 ===")
modeling_df = pd.read_csv('data/processed/treasury_modeling_data_20250903_174415.csv')
modeling_df['Date'] = pd.to_datetime(modeling_df['Date'])

print(f"建模数据日期范围: {modeling_df['Date'].min().date()} 到 {modeling_df['Date'].max().date()}")
print(f"建模数据记录数: {len(modeling_df):,}")
print(f"按年份统计:")
print(modeling_df['Date'].dt.year.value_counts().sort_index())

print("\n=== 检查TGA数据过滤 ===")
tga_data = df[df['account_type'] == 'Treasury General Account (TGA)']
print(f"TGA数据记录数: {len(tga_data):,}")
print(f"TGA数据日期范围: {tga_data['record_date'].min().date()} 到 {tga_data['record_date'].max().date()}")
print(f"TGA按年份统计:")
print(tga_data['record_date'].dt.year.value_counts().sort_index())


