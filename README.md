# Machine Learning Final Project — Human Activity Recognition

Code and data for my CS 7830 Machine Learning final project (Wright State University, Fall 2026).

**Dataset:** [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
(UCI ML Repository #240; Anguita et al., ESANN 2013) — 10,299 windows of smartphone
accelerometer/gyroscope features (561 per window) from 30 subjects performing 6 activities,
pre-split by subject into 7,352 train / 2,947 test windows.

**Goal:** six-class activity classification with regularized one-vs-all logistic regression,
evaluated with a confusion matrix and precision/recall/F1 on the held-out subjects, plus an
unsupervised look at the feature space (PCA, K-means / fuzzy C-means).

## Layout

- `src/` — model and training code
- `notebooks/` — EDA and experiments
- `data/` — dataset (not committed; see below)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib jupyter
python src/download_data.py        # fetches and unzips the UCI HAR dataset into data/
```

The paper (IEEE format) is kept in a separate private course repository.
