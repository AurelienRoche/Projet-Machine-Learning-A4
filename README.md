# Semiconductor Failure Prediction (SECOM)

ESILV A4 Machine Learning project: Armand Bussolino, Aurélien Roche, Jérémie Leroy-Dubief.

Predict whether a production sample passes or fails line testing from 590 process sensor measurements
([UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), CC BY 4.0).

## Run

```bash
pip install -r requirements.txt
```

Open the notebooks from the `notebooks/` folder and run them in order. The data is downloaded automatically into `data/` the first time.

## Structure

| Path | Content |
|---|---|
| `src/secom.py` | Shared code: data loading, train/test split, leak-free preprocessing |
| `notebooks/01_eda.ipynb` | Week 1: exploratory data analysis |
| `notebooks/02_data_preparation.ipynb` | Week 2: missing values, redundancy, cleaning decisions |
| `notebooks/03_baselines.ipynb` | Week 2: B0 / B1 baselines and cleaning variants (cross-validation) |
| `ai_governance/` | Module ownership, prompt log, AI audit |

## Method rules

- Stratified 80/20 train/test split with a fixed seed (`get_split`). The test set is only used for the final evaluation.
- Every fitted step (imputation, cleaning, scaling, selection, resampling) is inside a scikit-learn Pipeline and fitted on the training folds only.
- Main metric: average precision (Fail class). Accuracy is not used.
