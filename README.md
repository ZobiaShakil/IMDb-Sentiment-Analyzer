# IMDb-Sentiment-Analyzer
Built and evaluated a high accuracy sentiment classifier on the IMDb movie review dataset. By fine tuning a pre trained DistilBERT model and using Optuna for hyperparameter search, achieved 91.15 % test accuracy with balanced precision and recall across positive and negative classes.


### Highlights
Fine-tuned DistilBERT model on IMDb dataset

Hyperparameter tuning with Optuna

Interactive Streamlit UI

Movie info fetched via OMDb API

91.15% test accuracy


### How to Run

pip install -r requirements.txt

streamlit run streamlit_app/app.py
