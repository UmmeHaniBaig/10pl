# model2_rf.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load data
df = pd.read_csv('data/weather_data_clean.csv')

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Optional: drop timestamp if you don't want it
if 'timestamp' in categorical_cols:
    categorical_cols.remove('timestamp')
    X = X.drop(columns=['timestamp'])

# Preprocessing pipeline
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict and evaluate
predictions = model_pipeline.predict(X_test)
print("Random Forest R2:", r2_score(y_test, predictions))
import numpy as np
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# Save model pipeline
joblib.dump(model_pipeline, 'models/rf_model_pipeline.pkl')
