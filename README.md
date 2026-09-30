# Football IQ - Unified App (3 tabs)

## How to run (Windows)
1. Unzip this folder, open it in File Explorer, click the address bar, type `cmd`, press Enter.
2. Install once:  `python -m pip install -r requirements.txt`
3. Launch:        `python -m streamlit run app.py`
4. Opens at http://localhost:8501  (Stop with Ctrl+C in the black window)

## Tabs
1. **Match Predictor** - pick two teams: win/draw/loss probabilities + predicted corners and fouls (your trained XGBoost + Random Forest models).
2. **Player Dashboard** - type a name (typos OK) or pick from the list: profile, per-season table, goals/assists trend, per-90 vs league average.
3. **Sentiment Tracker** - chat assistant (goal, who scored, score, sentiment, yes/no, or any fan tweet) + Qatar 0-2 Ecuador timeline with the real goal times.
