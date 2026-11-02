"""Load the UCI HAR dataset (561-feature version) into pandas objects.

Usage (from a notebook in notebooks/):
    import sys; sys.path.append("../src")
    from har_data import load_har
    d = load_har()
    d.X_train, d.y_train, d.subject_train, d.X_test, ...
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "UCI HAR Dataset"


@dataclass
class HAR:
    X_train: pd.DataFrame
    y_train: pd.Series        # integer labels 1..6
    subject_train: pd.Series  # subject IDs 1..30
    X_test: pd.DataFrame
    y_test: pd.Series
    subject_test: pd.Series
    activities: dict[int, str]  # label id -> activity name

    def names(self, y: pd.Series) -> pd.Series:
        """Map integer labels to activity names."""
        return y.map(self.activities)


def feature_names(data_dir: Path = DATA_DIR) -> list[str]:
    """Return the 561 feature names, made unique.

    features.txt repeats 42 names: the 14 bandsEnergy() features of fBodyAcc,
    fBodyAccJerk and fBodyGyro each appear in three consecutive blocks of 14.
    Following the axis order used everywhere else in the file, the blocks are
    taken to be the X, Y and Z axes, and "-X"/"-Y"/"-Z" is appended.
    """
    raw = (
        pd.read_csv(data_dir / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
        ["name"].tolist()
    )
    counts = Counter(raw)
    seen: Counter = Counter()
    names = []
    for n in raw:
        if counts[n] == 1:
            names.append(n)
        else:
            assert counts[n] == 3, f"unexpected duplicate count for {n}"
            names.append(f"{n}-{'XYZ'[seen[n]]}")
            seen[n] += 1
    assert len(set(names)) == len(names) == 561
    return names


def activity_labels(data_dir: Path = DATA_DIR) -> dict[int, str]:
    df = pd.read_csv(data_dir / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"])
    return dict(zip(df["id"], df["name"]))


def _load_split(split: str, cols: list[str], data_dir: Path):
    d = data_dir / split
    X = pd.read_csv(d / f"X_{split}.txt", sep=r"\s+", header=None, names=cols)
    y = pd.read_csv(d / f"y_{split}.txt", header=None).iloc[:, 0].rename("activity")
    s = pd.read_csv(d / f"subject_{split}.txt", header=None).iloc[:, 0].rename("subject")
    assert len(X) == len(y) == len(s)
    return X, y, s


def load_har(data_dir: Path | str = DATA_DIR) -> HAR:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found - run `python src/download_data.py` first")
    cols = feature_names(data_dir)
    Xtr, ytr, str_ = _load_split("train", cols, data_dir)
    Xte, yte, ste = _load_split("test", cols, data_dir)
    return HAR(Xtr, ytr, str_, Xte, yte, ste, activity_labels(data_dir))
