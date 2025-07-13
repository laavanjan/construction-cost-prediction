


# 🏗️ Construction Cost Prediction

This project predicts the total construction cost of a building project based on various features such as building type, location, area, floors, quality, material index, and labor rate. The model uses machine learning algorithms like Random Forest and XGBoost for regression, and includes a simple and interactive Streamlit web app.

---

## 📁 Project Structure

```

cost-prediction/
│
├── data/
│   └── construction\_data.csv        # Dataset
│
├── model/
│   └── model.py                     # Model creation and training logic
│
├── app/
│   └── predict.py                   # Streamlit app for user predictions
│
├── utils/
│   └── dataloader.py                # Data generation and preprocessing
│
├── main.py                          # Script to train and save the model
├── requirements.txt                 # Required libraries
├── README.md                        # Project overview

````

---

## 🧠 Features

- 🔍 Predicts total cost using realistic synthetic data
- 📊 Uses Random Forest and/or XGBoost regressors
- 🧹 Includes data preprocessing pipelines
- 🌐 Streamlit frontend for real-time prediction
- 📦 Modular code structure for easy extension

---

## 🚀 Getting Started

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/your-username/cost-prediction.git
cd cost-prediction
````

### ✅ 2. Create & Activate a Conda Environment

```bash
conda create -n cost-env python=3.11
conda activate cost-env
```

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 4. Generate Data (Optional)

```bash
python -m utils.dataloader --output data/construction_data.csv
```

### ✅ 5. Train the Model

```bash
python main.py --data data/construction_data.csv --output model/cost_model.pkl
```

### ✅ 6. Launch the Streamlit App

```bash
streamlit run app/predict.py
```

---

## 💡 Example Input Fields

* Building Type: Residential / Commercial / Industrial
* Location: Urban / Suburban / Rural
* Area (sq.m): 100 – 10000
* Floors: 1 – 50
* Quality Grade: Standard / Premium / Luxury
* Labor Rate: in LKR
* Material Cost Index: 0.8 – 1.5
* Basement, Elevator, Parking: Yes / No

---

## 📚 Dependencies

* Python 3.11+
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

(See `requirements.txt` for exact versions.)

---

## 👨‍💻 Author

**RAM**
A self-taught machine learning and AI enthusiast working on real-world data-driven applications.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).




# 🏗️ Construction Cost Prediction

This project delivers a robust solution for predicting the total construction cost of building projects. By leveraging machine learning models, specifically Random Forest and XGBoost regressors, it analyzes key features such as building type, location, area, number of floors, quality grade, material cost index, and labor rates to provide accurate cost estimations. The project includes a user-friendly web interface built with Streamlit for interactive predictions.

---

## 📁 Project Structure

A well-organized project structure is crucial for maintainability and scalability. This project follows a modular design:

```

cost-prediction/
│
├── data/
│   └── construction\_data.csv     \# Sample or generated construction dataset
│
├── model/
│   └── model.py                   \# Contains model definition, training, and evaluation logic
|               
├── predict.py                    \# Streamlit application for user interaction and predictions
├── dataloader.py                 \# Utilities for data loading, generation, and preprocessing
├── main.py                       \# Main script to orchestrate model training and saving
├── requirements.txt              \# List of Python dependencies for reproducibility
├── README.md                     \# This file: Project overview and instructions
└── .gitignore                    \# Specifies intentionally untracked files by Git

````

---

## ✨ Key Features

-   **Accurate Cost Prediction**: Utilizes Random Forest and XGBoost regression models for high-precision cost estimations.
-   **Synthetic Data Generation**: Includes a utility to generate realistic synthetic construction data for model training and testing when real-world data is scarce.
-   **Comprehensive Data Preprocessing**: Implements robust data preprocessing pipelines, including feature scaling and encoding, to prepare data for machine learning.
-   **Interactive Web Interface**: Features a Streamlit-based frontend, allowing users to input project parameters and receive real-time cost predictions.
-   **Modular and Extensible Codebase**: Designed with a clear separation of concerns, making it easy to understand, maintain, and extend with new features or models.
-   **Support for Localized Parameters**: Capable of incorporating region-specific data, such as labor rates in Sri Lankan Rupees (LKR).
-   **Binary Feature Handling**: Effectively manages binary input features (e.g., "Yes/No" options for basement, elevator, parking).

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### ✅ 1. Clone the Repository

First, clone the project repository from GitHub to your local system:

```bash
git clone [https://github.com/laavanjan/cost-prediction.git](https://github.com/your-username/cost-prediction.git)
cd cost-prediction
````

### ✅ 2. Create and Activate a Conda Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.

```bash
conda create --name cost-env python=3.11
conda activate cost-env
```

### ✅ 3. Install Dependencies

Install all the necessary Python libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### ✅ 4. Generate Synthetic Data (Optional)

If you don't have a dataset, you can generate synthetic data using the provided script. This data will be saved in the `data/` directory.

```bash
python -m utils.dataloader --output data/construction_data.csv
```

### ✅ 5. Train the Prediction Model

Train the machine learning model using the available dataset (either your own or the one generated in the previous step). The trained model will be saved in the `model/` directory.

```bash
python main.py --data data/construction_data.csv --output model/cost_model.pkl
```

### ✅ 6. Launch the Streamlit Application

Start the interactive Streamlit web application to make predictions:

```bash
streamlit run app/predict.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

-----

## 💡 Example Input Fields for Prediction

The Streamlit application will prompt for the following (or similar) input features:

  * **Building Type**: e.g., Residential, Commercial, Industrial
  * **Location**: e.g., Urban, Suburban, Rural
  * **Total Area (sq.m)**: Numerical value (e.g., 100 – 10000)
  * **Number of Floors**: Integer value (e.g., 1 – 50)
  * **Quality Grade**: e.g., Standard, Premium, Luxury
  * **Labor Rate (LKR)**: Local labor cost per unit
  * **Material Cost Index**: A factor representing material cost fluctuations (e.g., 0.8 – 1.5)
  * **Basement**: Yes / No
  * **Elevator**: Yes / No
  * **Parking**: Yes / No

-----

## 🛠️ Core Technologies & Libraries

This project leverages the following technologies:

  * **Python**: Version 3.11+
  * **Pandas**: For data manipulation and analysis.
  * **NumPy**: For numerical operations.
  * **Scikit-learn**: For machine learning algorithms (Random Forest) and preprocessing tools.
  * **Streamlit**: For creating the interactive web application.
  * **Joblib**: For saving and loading trained machine learning models.

*(Refer to `requirements.txt` for specific versions.)*

-----


## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, please feel free to fork the repository, make your changes, and submit a pull request. You can also open an issue to report bugs or suggest new features.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----
````


