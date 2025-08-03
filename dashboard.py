import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ========== Page Setup ==========
st.set_page_config(page_title="Weather Dashboard", layout="wide")
st.title("🌤️ Weather & Pollution Dashboard")

# ========== Load Cleaned Data ==========
@st.cache_data
def load_data():
    return pd.read_csv("data/weather_data_clean.csv", parse_dates=["timestamp"])

df = load_data()

# ========== Load Trained Models ==========
linear_model = joblib.load("models/linear_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")

# ========== Filters ==========
with st.sidebar:
    st.header("🔍 Filter Data")

    date_range = st.date_input(
        "Select Date Range",
        [df['timestamp'].min().date(), df['timestamp'].max().date()]
    )

    weather_options = st.multiselect(
        "Select Weather Types",
        options=df['weather'].unique(),
        default=list(df['weather'].unique())
    )

# Apply filters
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
filtered_df = df[
    (df['timestamp'] >= start_date) &
    (df['timestamp'] <= end_date) &
    (df['weather'].isin(weather_options))
]

st.success(f"Filtered {len(filtered_df)} rows between {start_date.date()} and {end_date.date()}.")

# ========== Summary Stats ==========
st.subheader("📊 Summary Statistics")
st.dataframe(filtered_df.describe(), use_container_width=True)

# ========== Plots ==========
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Temperature & Humidity Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(filtered_df['timestamp'], filtered_df['temperature'], label='Temp (°C)', color='tomato')
    ax1.plot(filtered_df['timestamp'], filtered_df['humidity'], label='Humidity (%)', color='steelblue')
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Value")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("📊 Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(filtered_df[["temperature", "feels_like", "humidity", "wind_speed", "pressure", "uv_index"]].corr(),
                annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ========== Distribution Plots ==========
st.subheader("📊 Distributions")

for col in ["temperature", "humidity", "wind_speed", "pressure", "uv_index"]:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

# ========== Model Predictions ==========
st.subheader("🔮 Model Predictions")

input_features = ["temperature", "humidity", "wind_speed", "pressure", "uv_index"]

if all(feature in filtered_df.columns for feature in input_features):
    X = filtered_df[input_features].dropna()
    linear_preds = linear_model.predict(X)
    rf_preds = rf_model.predict(X)

    filtered_df = filtered_df.loc[X.index]  # Align with prediction rows
    filtered_df["Linear Regression Prediction"] = linear_preds
    filtered_df["Random Forest Prediction"] = rf_preds

    st.dataframe(filtered_df[["timestamp", *input_features,
                              "Linear Regression Prediction", "Random Forest Prediction"]].head())

    st.line_chart(filtered_df.set_index("timestamp")[[
        "Linear Regression Prediction", "Random Forest Prediction"
    ]])
else:
    st.warning("Required input features are missing from the dataset.")
