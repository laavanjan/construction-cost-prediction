import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def train_model(data_path, model_path="cost_model.pkl", preprocessor_path="preprocessor.pkl"):
    # Load and preprocess data
    data = pd.read_csv(data_path)
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    processed_data = preprocessor.fit_transform(data)
    X = processed_data[:, :-1]  # Features
    y = processed_data[:, -1]   # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regressor': SVR(),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate adjusted R²
        n = len(y_test)  # number of samples
        p = X_test.shape[1]  # number of features
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        results.append((name, r2, adjusted_r2, mae, rmse))

        print(f"{name} - R²: {r2:.4f}, Adjusted R²: {adjusted_r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Select the best model
    best_model = max(results, key=lambda x: x[1])  # Select model with highest R²
    final_model = models[best_model[0]]

    print("\nModel Evaluation Summary:")
    for name, r2, adjusted_r2, mae, rmse in results:
        print(f"{name}: R² = {r2:.4f}, Adjusted R² = {adjusted_r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    # Save the best model
    joblib.dump(final_model, model_path)
    print(f"Best model saved as {model_path}")

    # Save the preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved as {preprocessor_path}")

    return final_model, preprocessor


if __name__ == "__main__":
    # Example usage
    train_model("construction_data.csv")
