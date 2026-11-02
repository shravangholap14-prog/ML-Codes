"""
Practical 4: SVM classification using Linear and RBF kernels.
Dataset: Breast Cancer Wisconsin (built into scikit-learn).
Run: python 04_svm_breast_cancer.py
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

    models = {
        "Linear Kernel": SVC(kernel="linear", C=1.0),
        "RBF Kernel": SVC(kernel="rbf", C=1.0, gamma="scale")
    }
    for name, model in models.items():
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        print(f"\n{name}")
        print("Accuracy:", round(accuracy_score(y_test, pred), 4))
        print(classification_report(y_test, pred, target_names=data.target_names))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=data.target_names, cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
