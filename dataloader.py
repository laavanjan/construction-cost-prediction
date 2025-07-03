import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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


def preprocess_data(
    data: pd.DataFrame,
    target_column: str = "total_cost",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Preprocess the construction data for model training.

    Args:
        data (pd.DataFrame): Raw construction data.
        target_column (str): Name of the target column.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed.

    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
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
