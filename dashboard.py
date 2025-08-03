# ========== Model Predictions ==========
st.subheader("🧠 Air Quality Predictions")

# Select model
model_option = st.selectbox("Choose a model for prediction:", ("Linear Regression", "Random Forest"))

# Features used for prediction (ensure these match the model training)
feature_cols = ["temperature", "humidity", "wind_speed", "pressure", "uv_index"]
input_data = filtered_df[feature_cols].dropna()

# Make predictions
if model_option == "Linear Regression":
    predictions = linear_model.predict(input_data)
else:
    predictions = rf_model.predict(input_data)

# Display predictions
filtered_df = filtered_df.loc[input_data.index]  # align rows
filtered_df["Predicted AQI"] = predictions

st.dataframe(filtered_df[["timestamp", "weather", "Predicted AQI"]].head(10), use_container_width=True)

# Plot predictions
st.subheader("📉 Predicted AQI Over Time")
fig3, ax3 = plt.subplots()
ax3.plot(filtered_df["timestamp"], filtered_df["Predicted AQI"], label="Predicted AQI", color="purple")
ax3.set_xlabel("Timestamp")
ax3.set_ylabel("AQI")
ax3.set_title("Predicted AQI Over Time")
st.pyplot(fig3)
