# Stroke Risk Screening – IT2011 Group 21

Streamlit web app for the group's calibrated XGBoost stroke-risk model.
Course prototype only – not for clinical use.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – the web app
- `model_bundle.joblib` – trained model + decision threshold
- `iqr_capper.py` – custom preprocessing class needed to load the model
- `requirements.txt` – exact library versions the model was saved with (do not change them)
