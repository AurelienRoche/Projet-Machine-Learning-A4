# AI audit

Concrete flaws found in AI outputs (at least 4, at least one per student) and how they were fixed. Updated every week.

| # | Student | Week | Flaw in the AI output | How it was caught | Fix |
|---|---|---|---|---|---|
| 1 | Armand | 1 | The specification draft stated 591 features and called the samples "wafer lots". | Team review: compared with the EDA output (590 columns; the 591st UCI attribute is a timestamp stored with the labels) and with the UCI description ("production entities"). | 590 features, "samples". |
| 2 | Aurélien | 1 | The EDA notebook said the data was loaded with `ucimlrepo`, while the code read local files. | Team review: code compared with its description. | Description corrected; loading centralized in `load_secom()`. |
| 3 | Jérémie | 2 | The redundancy filter computed correlations between very sparse sensors on only 2–3 common samples. Two points always give \|r\| = 1, so a sensor could be dropped as a "duplicate" by chance. | Code review before running it. | Correlation computed only when at least 100 samples are shared (`min_periods=100`). |
| 4 | Jérémie | 2 | Generated code called `np.fill_diagonal` on an array returned by `DataFrame.to_numpy()`, which is read-only with pandas 3: the notebook crashed. | Running the notebook end to end. | `to_numpy(copy=True)`. |
