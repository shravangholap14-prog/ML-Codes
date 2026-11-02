"""
Practical 1: Basic Feature Engineering Techniques
Works with Python 3.9+, Jupyter, Anaconda, VS Code and Google Colab.
Run: python 01_basic_feature_engineering.py
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def main():
    # Sample mixed-type dataset. Replace this DataFrame with pd.read_csv("your_file.csv")
    data = {
        "Age": [22, 25, None, 35, 28, None, 42, 31],
        "Salary": [25000, 30000, 28000, None, 42000, 39000, 52000, 36000],
        "City": ["Pune", "Mumbai", "Pune", "Delhi", None, "Mumbai", "Delhi", "Pune"],
        "Education": ["UG", "PG", "UG", "PG", "UG", None, "PG", "UG"],
        "Purchased": [0, 1, 0, 1, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    print("Original Dataset:\n", df)
    print("\nMissing values before preprocessing:\n", df.isnull().sum())

    numeric_features = ["Age", "Salary"]
    categorical_features = ["City", "Education"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Compatibility with older scikit-learn versions
    try:
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])
        transformed = preprocessor.fit_transform(df)
    except TypeError:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])
        transformed = preprocessor.fit_transform(df)

    feature_names = preprocessor.get_feature_names_out()
    result = pd.DataFrame(transformed, columns=feature_names)
    result["Purchased"] = df["Purchased"].values

    print("\nDataset after handling missing values, encoding and scaling:")
    print(result.round(3))
    print("\nMissing values after preprocessing:\n", result.isnull().sum())
    print("\nFeature engineering completed successfully.")

if __name__ == "__main__":
    main()
