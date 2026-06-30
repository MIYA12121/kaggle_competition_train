# ROGII v3 solution

This folder contains a stronger Kaggle-friendly version.

Files:

- `rogii_v3_solution.py`: training, validation calibration, model saving, and submission generation.
- `rogii_v3.ipynb`: notebook version for Kaggle.
- `submission.csv`: generated local submission when run here.
- `models/model_artifacts.joblib`: saved model bundle after training.
- `models/model_metadata.json`: model parameters and validation metrics.

Main optimization:

- Use each well's last known `TVT_input` as a strong baseline.
- Train only on rows where `TVT_input` is missing, matching the hidden prediction zone.
- Predict residuals: `TVT - last_known_tvt`.
- Use a validation well split to choose how much model residual should be blended back into the baseline.
- Prefer LightGBM/XGBoost when available; fall back to sklearn models or baseline-only if needed.

Run:

```bash
python 3/rogii_v3_solution.py --data-dir rogii-wellbore-geology-prediction --output 3/submission.csv --model-dir 3/models
```

For Kaggle, open `rogii_v3.ipynb`, run all cells, then submit the generated `/kaggle/working/submission.csv`.
