# Stroke Risk Screening – IT2011 Group 21

Streamlit web app for the group's calibrated XGBoost stroke-risk model.
Choose how the result is shown: **risk percentage**, **Risk / No risk label**, or **both**.
Course prototype only – not for clinical use.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Upload these files (not the folder) to a public GitHub repository.
2. On share.streamlit.io choose Create app → your repo → `app.py`; under Advanced settings choose Python 3.12.
3. Deploy.

## Files
- `app.py` – the web app
- `model_bundle.joblib` – trained model + decision threshold
- `iqr_capper.py` – custom preprocessing class needed to load the model
- `requirements.txt` – exact library versions the model was saved with (do not change them)
