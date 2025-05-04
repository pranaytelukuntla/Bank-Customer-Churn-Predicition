
# Bank Customer Churn Prediction

A modular, end-to-end Python pipeline for predicting bank customer churn using Scikit-learn and XGBoost.

## 📖 Project Overview

Customer churn is a key challenge in retail banking, where retaining customers is significantly more cost-effective than acquiring new ones. This project offers a complete machine learning workflow—from data ingestion to model evaluation—to help identify customers at risk of leaving, enabling informed retention strategies.

## 🗂️ Repository Structure

```
Bank-Customer-Churn-Prediction/
├── data/  
│   ├── raw/                  # Original input datasets  
│   └── processed/            # Cleaned and transformed data  
├── notebooks/  
│   └── Churn_Exploration.ipynb  # Exploratory data analysis and modeling  
├── src/  
│   ├── data_preprocessing.py    # Data loading and cleaning  
│   ├── feature_engineering.py   # Encoding, scaling, and derived features  
│   ├── model_training.py        # Model training and hyperparameter tuning  
│   └── evaluation.py            # Model evaluation and visualization  
├── requirements.txt             # Dependency list  
└── README.md                    # Project documentation  
```

## 🔧 Module Overview

### `data_preprocessing.py`
- `load_data(path)`: Loads CSV files into a DataFrame.
- `clean_data(df)`: Handles missing values and duplicates.
- `split_data(df, target)`: Splits dataset into training and test sets.

### `feature_engineering.py`
- `encode_categoricals(df, cols)`: Label encodes categorical columns.
- `scale_numericals(df, cols)`: Standardizes numeric features.
- `engineer_features(df)`: Adds derived features such as ratios.

### `model_training.py`
- `get_models()`: Returns baseline models.
- `tune_model(model, grid, X, y)`: Performs grid search.
- `train_all(X, y)`: Trains and returns all models.

### `evaluation.py`
- `evaluate_model(model, X, y)`: Returns evaluation metrics.
- `plot_roc(model, X, y)`: Plots ROC curves with AUC.

## 🚀 Getting Started

1. **Clone the repository and install dependencies**

   ```bash
   git clone https://github.com/your-org/Bank-Customer-Churn-Prediction.git
   cd Bank-Customer-Churn-Prediction
   python3 -m venv venv
   source venv/bin/activate       # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Run the pipeline**

   ```python
   from src.data_preprocessing import load_data, clean_data, split_data
   from src.feature_engineering import encode_categoricals, scale_numericals, engineer_features
   from src.model_training import train_all
   from src.evaluation import evaluate_model, plot_roc

   df = load_data('data/raw/customer_churn.csv')
   df = clean_data(df)
   df = engineer_features(df)
   df = encode_categoricals(df, ['Geography', 'Gender'])
   df = scale_numericals(df, ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Balance_Age_Ratio'])
   X_train, X_test, y_train, y_test = split_data(df, target='Exited')

   models = train_all(X_train, y_train)
   for name, model in models.items():
       metrics = evaluate_model(model, X_test, y_test)
       print(f"{name}:", metrics)
       plot_roc(model, X_test, y_test)
   ```

## 📈 Example Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.82     | 0.78      | 0.80   | 0.79     | 0.87    |
| Random Forest       | **0.85** | **0.82**  | **0.83** | **0.83** | **0.90** |
| XGBoost             | 0.84     | 0.80      | 0.82   | 0.81     | 0.89    |

## ⚙️ Dependencies

Refer to `requirements.txt` for full list. Core packages include:

- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
