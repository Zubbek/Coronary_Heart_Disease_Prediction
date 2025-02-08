# Coronary Heart Disease (CHD) Prediction

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zubbek/Coronary_Heart_Disease_Prediction/blob/main/CHD_Prediction.ipynb)

## 📖 Project Overview
This project focuses on predicting Coronary Heart Disease (CHD) using machine learning techniques. The dataset used is `Z-Alizadeh-sani-dataset.xlsx`, which contains medical records and clinical features of patients. The goal is to develop a predictive model that can classify individuals at risk of CHD based on these features.

## 📂 Dataset
- **Source**: [`Z-Alizadeh-Sani dataset`](https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani)
- **Size**: Various clinical parameters of patients
- **Target Variable**: CHD diagnosis (Binary Classification: Presence or Absence)

## 🔧 Installation & Requirements
Ensure you have Python installed along with the required dependencies. You can install the necessary packages using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap optuna 
```

## 🚀 Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/Zubbek/Coronary_Heart_Disease_Prediction.git
   ```
2. Open the Jupyter Notebook or Google Colab.
3. Load the dataset from the provided Excel file.
4. Run the notebook cells to explore data, preprocess it, and train the model.

## 📊 Methodology
1. **Data Preprocessing**
   - Handling missing values
   - Handling Outliers
   - Feature selection and encoding
2. **Exploratory Data Analysis (EDA)**
   - Distribution of key features
   - Correlation analysis
3. **Model Training & Evaluation**
   - Machine learning classifiers used:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - LightGBM
     - XGBoost
     - GradientBoosting Classifier
     - NuSVC
     - ExtraTrees Classifier
     - Decision Trees
     - Random Forest
     - MLP Classifier
   - Hyperparameter tuning using:
     - Randomized Search
     - Optuna optimization framework
   - Performance metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
4. **Model Explainability**
   - SHAP (SHapley Additive exPlanations) values to interpret feature importance and individual predictions
5. **Principal Component Analysis**
- Dimensionality reduction for 70%, 80% and 90% for an explanation of the total variance using Principal Component Analysis (PCA)

## 📈 Results
- Model performance metrics will be displayed in the notebook.
- Best Models with hihgest Accuracy
- Visualizations of key insights from the dataset.
- Feature importance analysis using SHAP values.
- Accuracy after PCA

## 🤝 Contribution
Feel free to contribute by submitting issues or pull requests. Suggestions for model improvements and additional features are welcome!

---

### ✨ Future Work
- Advanced hyperparameter tuning strategies.
- Implementing deep learning models.
