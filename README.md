# Machine Learning Algorithms Coded from Scratch

## Project Description

This project implements 13 machine learning algorithms (supervised and unsupervised) from scratch using only NumPy. Its purpose is to replicate a scikit-learn style ML algorithm library, and is thus structured as an installable python package.

## Available Algorithms

**Supervised:** Linear Regression, Logistic Regression, Softmax Regression, Decision Tree, Perceptron, Naive Bayes, Gaussian Discriminant Analysis

**Unsupervised:** K-Means, PCA, Gaussian Mixture Model, Factor Analysis, ICA, Collaborative Filtering

## Using this Package

```bash
pip install -e .
```

```python
from ml.supervised import LogisticRegressor
from ml.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
model  = LogisticRegressor(learning_rate=0.1, num_iterations=1000).fit(scaler.transform(X_train), y_train)

print(model.score(scaler.transform(X_test), y_test))
```

