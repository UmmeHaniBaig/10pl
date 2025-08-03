# model2.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load preprocessed data
df = pd.read_csv('data/weather_data_clean.csv')

# Features and target
X = df.drop(columns=['AirQualityIndex'])
y = df['AirQualityIndex']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
print("Random Forest R2:", r2_score(y_test, predictions))
print("Random Forest RMSE:", mean_squared_error(y_test, predictions, squared=False))

# Save model
joblib.dump(model, 'models/rf_model.pkl')
