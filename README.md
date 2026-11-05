# Human Activity Recognition

Code and data for CS 7830 Machine Learning final project

**Dataset:** [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
(UCI ML Repository #240; Anguita et al., ESANN 2013) — 10,299 windows of smartphone
accelerometer/gyroscope features (561 per window) from 30 subjects performing 6 activities.

## Layout

- `src/` — shared code: `har_data.py` loads the dataset, `preprocess.py` (duplicate-column removal, standardization, PCA; fit on train only), `clustering.py` (HW1 K-means and fuzzy C-means, prediction and metrics), `plot_style.py` (figure style)
- `notebooks/` — one notebook per stage, run in order: `01_eda.ipynb`, `02_preprocessing_pca.ipynb`, `03_clustering.ipynb` (six-activity and moving/still clustering; PCA comparison)
- `figures/` — saved figures (PDF + PNG) in `eda/`, `preprocessing/`, and `clustering/`
- `tests/` — synthetic clustering checks, Hungarian matching, and shared preprocessing checks
- `data/` — dataset (not committed; see below)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py        # fetches and unzips the UCI HAR dataset into data/
```

The paper (IEEE format) is kept in a separate private course repository.

## Run the analysis

Open the notebooks from `notebooks/` using the project environment and run them in
order. Save changes to `STANDARDIZE` and `N_COMPONENTS` in 02 before rerunning 03;
03 reads those literal settings and rebuilds the shared train-only preprocessing
pipeline. The current `N_COMPONENTS = 0.95` remains a placeholder.

03 uses ten seeded restarts per algorithm, selects by the training objective, and
assigns test windows to fixed training centres. Its matched accuracy uses the
training Hungarian mapping; split-optimal matching is reported separately.

To run the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
