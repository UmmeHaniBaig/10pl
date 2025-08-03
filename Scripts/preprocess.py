import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Create output folder
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/weather_data.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df.dropna(subset=['timestamp'], inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Print basic info
print("📊 Initial Shape:", df.shape)
print("\n🧹 Missing Values:\n", df.isnull().sum())

# Drop rows with missing values (optional: use fillna instead)
df.dropna(inplace=True)

# Print summary statistics
print("\n📈 Summary Statistics:")
print(df.describe())

# Columns to visualize and scale
num_cols = ['temperature', 'feels_like', 'humidity', 'wind_speed', 'pressure', 'uv_index']

# ========== 📊 VISUALIZATIONS ==========

# Distribution plots
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.savefig(f"plots/dist_{col}.png")
    plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# Boxplots for outliers
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"plots/box_{col}.png")
    plt.close()

# Time series trend of temperature and humidity
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['temperature'], label='Temperature', color='tomato')
plt.plot(df['timestamp'], df['humidity'], label='Humidity', color='steelblue')
plt.legend()
plt.title("Temperature & Humidity Over Time")
plt.xlabel("Time")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/temp_humidity_over_time.png")
plt.close()

# ========== ⚙️ SCALING NUMERICAL FEATURES ==========
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

# Save cleaned & scaled data
df_scaled.to_csv("data weather_data_clean.csv", index=False)
print("\n✅ Preprocessing complete. Cleaned data saved to 'data/weather_data_clean.csv'")
print("📁 Plots saved in 'plots/' folder.")
