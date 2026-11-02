"""
Practical 8: Logistic Regression for Binary Classification.
Dataset: Breast Cancer Wisconsin (built into scikit-learn).
Run: python 08_logistic_regression.py
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Accuracy: ", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred), 4))
    print("Recall:   ", round(recall_score(y_test, pred), 4))
    print("F1-score: ", round(f1_score(y_test, pred), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, target_names=data.target_names))

    cm = confusion_matrix(y_test, pred)
    print("Confusion Matrix:\n", cm)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names).plot(cmap="Blues")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
