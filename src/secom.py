"""Shared data loading, split and preprocessing for the SECOM project.

Every notebook uses these functions so that the split and the preprocessing
are identical across experiments. All cleaning steps are sklearn transformers:
they are fitted inside each training fold, never on validation or test data.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UCI_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SEED = 42


def load_secom(data_dir=DATA_DIR):
    """Return X (1567 x 590 DataFrame) and y (Series, 1 = Fail, 0 = Pass).

    Downloads the UCI files into data/ the first time.
    """
    data_dir = Path(data_dir)
    if not (data_dir / "secom.data").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UCI_URL, timeout=60) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(data_dir)

    X = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]
    labels = pd.read_csv(data_dir / "secom_labels.data", sep=r"\s+", header=None)
    y = (labels.iloc[:, 0] == 1).astype(int).rename("fail")  # UCI: -1 = Pass, 1 = Fail
    return X, y


def get_split(X, y):
    """Stratified 80/20 split with a fixed seed. The test set is used once, in Week 4."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


class DropHighMissing(BaseEstimator, TransformerMixin):
    """Drop features whose missing rate (on the training fold) exceeds max_missing."""

    def __init__(self, max_missing=0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        self.keep_ = np.isnan(np.asarray(X, dtype=float)).mean(axis=0) <= self.max_missing
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class DropCorrelated(BaseEstimator, TransformerMixin):
    """Drop one feature of each pair with |Pearson r| >= threshold (computed on the training fold).

    Scans features left to right and keeps a feature only if it is not
    highly correlated with a feature already kept. Missing values are
    ignored pairwise; a pair needs at least min_periods common rows,
    otherwise very sparse columns can show |r| = 1 by chance.
    """

    def __init__(self, threshold=0.98, min_periods=100):
        self.threshold = threshold
        self.min_periods = min_periods

    def fit(self, X, y=None):
        corr = pd.DataFrame(np.asarray(X, dtype=float)).corr(min_periods=self.min_periods).abs().to_numpy()
        keep = np.ones(corr.shape[0], dtype=bool)
        for j in range(corr.shape[0]):
            if keep[j]:
                dup = corr[j, j + 1:] >= self.threshold  # NaN (constant column) compares False
                keep[j + 1:][dup] = False
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def make_preprocessor(max_missing=None, corr_threshold=None, missing_indicator=False):
    """Leak-free preprocessing steps.

    Default (B1): median imputation -> drop constant features -> standardization.
    Optional cleaning: drop features with > max_missing missing values,
    drop near-duplicates (|r| >= corr_threshold), add missing-value indicators.
    """
    steps = []
    if max_missing is not None:
        steps.append(("drop_missing", DropHighMissing(max_missing)))
    if corr_threshold is not None:
        steps.append(("drop_correlated", DropCorrelated(corr_threshold)))
    steps += [
        ("impute", SimpleImputer(strategy="median", add_indicator=missing_indicator,
                                 keep_empty_features=True)),
        ("drop_constant", VarianceThreshold()),
        ("scale", StandardScaler()),
    ]
    return steps


def make_pipeline(model, **prep_kwargs):
    """Preprocessing + model in one Pipeline, ready for cross-validation."""
    return Pipeline(make_preprocessor(**prep_kwargs) + [("model", model)])
