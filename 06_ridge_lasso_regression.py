"""
Practical 6: Ridge and Lasso Regression on a multivariate dataset.
Dataset: Diabetes dataset built into scikit-learn.
Run: python 06_ridge_lasso_regression.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def main():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)
    }
    rows = []
    for name, estimator in models.items():
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        rows.append([name, mse, mse ** 0.5, r2])
    results = pd.DataFrame(rows, columns=["Model", "MSE", "RMSE", "R2"])
    print(results.to_string(index=False))

    results.set_index("Model")[["RMSE", "R2"]].plot(kind="bar", figsize=(9, 5))
    plt.title("Comparison of Linear, Ridge and Lasso Regression")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
