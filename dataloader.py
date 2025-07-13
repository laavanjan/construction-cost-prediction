import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load construction project data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or unreadable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("The provided CSV file is empty.")
        logger.info(
            f"Data loaded successfully from {file_path} with shape {data.shape}"
        )
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(data: pd.DataFrame):
    """
    Preprocess the data by scaling numeric features and encoding categorical features.

    Args:
        data (pd.DataFrame): The raw data.

    Returns:
        tuple: Preprocessed train and test sets, and the preprocessor object.
    """
    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = data.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    processed_data = preprocessor.fit_transform(data)
    X = processed_data[:, :-1]  # Features
    y = processed_data[:, -1]   # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Preprocessing complete. Data split into training and testing sets.")

    return X_train, X_test, y_train, y_test, preprocessor


def generate_sample_data(
    n_samples: int = 100, output_file: str = "construction_data.csv"
) -> pd.DataFrame:
    """
    Generate synthetic construction project data for testing purposes.

    Args:
        n_samples (int): Number of samples to generate.
        output_file (str): Path to save the generated CSV.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    np.random.seed(42)

    data = {
        "building_type": np.random.choice(
            ["Residential", "Commercial", "Industrial"], n_samples
        ),
        "area_sqm": np.random.uniform(100, 10000, n_samples),
        "floors": np.random.randint(1, 50, n_samples),
        "location": np.random.choice(["Urban", "Suburban", "Rural"], n_samples),
        "foundation_type": np.random.choice(["Concrete", "Pile", "Slab"], n_samples),
        "roof_type": np.random.choice(["Flat", "Pitched", "Dome"], n_samples),
        "has_basement": np.random.choice([0, 1], n_samples),
        "has_parking": np.random.choice([0, 1], n_samples),
        "labor_rate": np.random.uniform(20, 50, n_samples),
    }

    df = pd.DataFrame(data)

    # Cost Calculation
    base_cost = 500
    df["total_cost"] = (
        base_cost
        * df["area_sqm"]
        * df["building_type"].map(
            {"Residential": 1.0, "Commercial": 1.3, "Industrial": 1.1}
        )
        * df["location"].map({"Urban": 1.2, "Suburban": 1.0, "Rural": 0.8})
        * (1 + 0.1 * df["floors"])
        * (1 + 0.05 * df["has_basement"])
        * (1 + 0.02 * df["has_parking"])
    )

    # Add noise
    df["total_cost"] *= np.random.normal(1, 0.1, n_samples)

    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Synthetic data generated and saved to {output_file}")

    return df


if __name__ == "__main__":
    generate_sample_data()
