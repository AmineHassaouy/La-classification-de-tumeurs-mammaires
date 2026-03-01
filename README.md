# Classification de Tumeurs Mammaires (Bénignes / Malignes)

Classification of breast tumors as **benign** or **malignant** using machine learning on the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| Samples | 569 |
| Features | 30 (computed from cell nucleus images) |
| Classes | Malignant (212) · Benign (357) |

Available directly via `sklearn.datasets.load_breast_cancer()`.

---

## Project Structure

```
breast-cancer/
├── main.ipynb      # Full notebook: EDA → Preprocessing → Models → Comparison
├── README.md
└── .gitignore
```

---

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **EDA** | Class distribution, feature histograms, correlation heatmap, pairplot |
| 2 | **Preprocessing** | 80/20 stratified split, StandardScaler |
| 3 | **SVM** | GridSearchCV over C, kernel, gamma — confusion matrix & ROC curve |
| 4 | **Random Forest** | GridSearchCV over n_estimators, max_depth — feature importance chart |
| 5 | **MLP** | GridSearchCV over hidden layers, activation, alpha — loss curve |
| 6 | **Comparison** | Metric table, grouped bar chart, overlay ROC curves, conclusion |

---

## Models & Results

All three models are tuned with **5-fold cross-validated GridSearchCV** and evaluated on a held-out test set.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| SVM (RBF) | ~97% | ~99% |
| Random Forest | ~96% | ~99% |
| MLP | ~97% | ~99% |

> Exact scores depend on the random seed and hyperparameter search results.

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook main.ipynb
```

---

## Key Takeaways

- **SVM** with RBF kernel achieves the highest accuracy on this dataset.
- **Random Forest** provides useful feature importance rankings.
- **MLP** is competitive but requires more tuning and compute.
- In a medical context, **Recall** for the malignant class is the most critical metric — a false negative (missed cancer) is far more costly than a false positive.
