# Multimodal Construction Intelligence Platform

Skill Wallet Capstone Project 1.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. On share.streamlit.io, choose New app → select the repo → `app.py`.
3. Optional: in App settings → Secrets add `GEMINI_API_KEY = "your-key"`.

Without a key the app runs in demonstration mode.
