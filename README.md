# Machine Learning Practical Codes

This repository contains working Python programs for all 8 practical assignments.

## Practicals
1. Basic Feature Engineering Techniques
2. Principal Component Analysis (PCA) using Wine dataset
3. Uber Ride Fare Prediction using Linear, Ridge and Lasso Regression
4. SVM using Linear and RBF kernels on Breast Cancer dataset
5. Polynomial Regression with different polynomial degrees
6. Ridge and Lasso Regression on a multivariate dataset
7. K-Means Clustering on Iris dataset using the Elbow Method
8. Logistic Regression for Binary Classification

## Requirements
- Python 3.9 or newer recommended
- numpy
- pandas
- matplotlib
- scikit-learn

Install all dependencies:
```bash
python -m pip install -r requirements.txt
```

## Run a practical
Open a terminal in this folder and run, for example:
```bash
python 02_pca_wine.py
```
You can also open any `.py` file in Jupyter, Spyder (Anaconda), VS Code or Google Colab.

## Dataset notes
Most practicals use datasets included with scikit-learn, so no separate dataset download is needed.
Practical 3 uses `uber.csv` if it is placed in this folder. The CSV should contain:
- `fare_amount`
- `pickup_longitude`
- `pickup_latitude`
- `dropoff_longitude`
- `dropoff_latitude`
- optional: `passenger_count`, `pickup_datetime`

If `uber.csv` is not present, Practical 3 creates a synthetic sample dataset so the code can still be executed and demonstrated. For actual assignment results, use your instructor-approved Uber dataset.

## Important
The scripts print evaluation results and display graphs where appropriate. Results can vary slightly with library versions and dataset versions.
