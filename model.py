import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataloader import load_data, preprocess_data

def train_model(data_path, model_path='cost_model.pkl'):
    # Load and preprocess data
    data = load_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)

    # Define Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Build pipeline (preprocessing + model)
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_model)
    ])

    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Save the model
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    return pipeline

if __name__ == "__main__":
    # Example usage
    train_model("construction_data.csv")
