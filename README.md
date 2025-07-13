
# 🏗️ Construction Cost Prediction



This project delivers a robust solution for predicting the total construction cost of building projects. By leveraging machine learning models, specifically Random Forest and XGBoost regressors, it analyzes key features such as building type, location, area, number of floors, foundation type, roof type, and labor rates to provide accurate cost estimations. The project includes a user-friendly web interface built with Streamlit for interactive predictions.

![Construction Cost Estimation](Screenshot%202025-07-13%20230543.png)
---

## ✨ Key Features

-   **Accurate Cost Prediction**: Utilizes Random Forest and XGBoost regression models with 96%+ accuracy for high-precision cost estimations.
-   **Realistic Data Processing**: Works with construction costs ranging from $14M to $500M using MinMaxScaler preprocessing.
-   **Comprehensive Data Preprocessing**: Implements robust data preprocessing pipelines, including feature scaling and encoding, to prepare data for machine learning.
-   **Interactive Web Interface**: Features a Streamlit-based frontend, allowing users to input project parameters and receive real-time cost predictions.
-   **Modular and Extensible Codebase**: Designed with a clear separation of concerns, making it easy to understand, maintain, and extend with new features or models.
-   **Multiple ML Models**: Compares Linear Regression, Random Forest, Gradient Boosting, SVR, and XGBoost to select the best performer.
-   **Binary Feature Handling**: Effectively manages binary input features (e.g., "Yes/No" options for basement, parking).

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/laavanjan/construction-cost-prediction.git
cd construction-cost-prediction
```

### ✅ 2. Create and Activate a Virtual Environment

```bash
conda create --name cost-env python=3.11
conda activate cost-env
```

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 4. Generate Realistic Dataset

```bash
python generate_dataset.py
```

### ✅ 5. Train the Models

Run the Jupyter notebook `model_comparison.ipynb` to train and compare multiple ML models, or run individual cells to:
- Load and preprocess the data
- Train multiple algorithms (Random Forest, XGBoost, etc.)
- Compare model performance
- Save the best model

### ✅ 6. Launch the Streamlit Application

```bash
streamlit run predict.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

---

## 💡 Input Features for Prediction

The Streamlit application accepts the following input parameters:

* **Building Type**: Residential, Commercial, Industrial
* **Area (sq.m)**: 250 - 15,000 square meters
* **Number of Floors**: 1 - 20 floors
* **Location**: Urban, Suburban, Rural
* **Foundation Type**: Concrete, Pile, Slab
* **Roof Type**: Flat, Pitched, Dome
* **Has Basement**: Yes / No
* **Has Parking**: Yes / No
* **Labor Rate**: 3,000 - 15,000 (per hour)

**Example Prediction Output**: $453,026,328 (Construction cost ranging from $14M to $500M)

---

## �️ Technologies & Libraries

* **Python**: 3.11+
* **Pandas**: Data manipulation and analysis
* **NumPy**: Numerical operations
* **Scikit-learn**: Machine learning algorithms and preprocessing (MinMaxScaler, Random Forest)
* **XGBoost**: Gradient boosting framework
* **Streamlit**: Interactive web application
* **Joblib**: Model serialization
* **Matplotlib**: Data visualization

*(See `requirements.txt` for exact versions)*

---

## 📊 Model Performance

The project compares multiple machine learning algorithms:

| Model | R² Score | Adjusted R² | MAE | RMSE |
|-------|----------|-------------|-----|------|
| **Random Forest** | **0.9603** | **0.9596** | **7.49M** | **21.24M** |
| XGBoost | 0.9507 | 0.9498 | 9.51M | 23.69M |
| Gradient Boosting | 0.9493 | 0.9484 | 12.18M | 24.02M |
| Linear Regression | 0.3580 | 0.3469 | 65.36M | 85.45M |

**Best Model**: Random Forest with 96.03% accuracy

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Laavanjan**  
GitHub: [@laavanjan](https://github.com/laavanjan)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
````


