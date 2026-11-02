"""
Practical 5: Polynomial Regression with different polynomial degrees.
Run: python 05_polynomial_regression.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(-3, 3, 100)).reshape(-1, 1)
    y = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 0] ** 2 + 0.25 * X[:, 0] ** 3
    y += rng.normal(0, 2.0, size=len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    degrees = [1, 2, 3, 5, 10]
    x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    plt.figure(figsize=(9, 6))
    plt.scatter(X, y, color="black", s=22, label="Data")
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"Degree {degree}: MSE={mean_squared_error(y_test, pred):.3f}, "
              f"R2={r2_score(y_test, pred):.3f}")
        plt.plot(x_plot, model.predict(x_plot), label=f"Degree {degree}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Polynomial Regression for Different Degrees")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
