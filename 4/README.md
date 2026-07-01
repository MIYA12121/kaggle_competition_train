# ROGII v4 solution

This is the stronger fourth version. It keeps the v3 residual model idea, then adds several ideas inspired by strong public notebooks:

- last-known `TVT_input` baseline;
- GR/typewell particle-filter tracker;
- spatial formation prior from train wells;
- residual boosting model with LightGBM/XGBoost/CatBoost/sklearn fallback;
- validation-calibrated blend of model, particle filter, and spatial candidates;
- robust per-well projection smoothing;
- guarded train/test overlap contact override;
- saved model artifacts and metadata.

Run locally from the repository root:

```bash
python 4/rogii_v4_solution.py --data-dir rogii-wellbore-geology-prediction --output 4/submission.csv --model-dir 4/models
```

For a quick smoke test:

```bash
python 4/rogii_v4_solution.py --data-dir rogii-wellbore-geology-prediction --output 4/submission.csv --model-dir 4/models --max-train-wells 30 --max-train-rows 20000 --calibration-rows 5000 --pf-train-particles 24 --pf-train-seeds 1 --pf-test-particles 64 --pf-test-seeds 2
```

Kaggle outputs:

- `/kaggle/working/submission.csv`
- `/kaggle/working/submission_raw.csv`
- `/kaggle/working/models/model_artifacts.joblib`
- `/kaggle/working/models/model_metadata.json`
- `/kaggle/working/models/test_features.parquet`

Open `rogii_v4.ipynb` on Kaggle and run all cells.
