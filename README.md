# 🛠️ Predictive Maintenance: Machine Failure Prediction

Machine learning project that predicts whether a machine is likely to fail from its operating conditions (temperature, rotational speed, torque, tool wear, etc.). Four classifiers are trained and compared, and the results are served through an interactive **Streamlit** web app.

## Features

- **Full EDA:** class balance, distributions, outliers, correlations, categorical analysis
- **Preprocessing:** leakage and ID column removal, feature engineering, stratified split, scaling and one-hot encoding inside a scikit-learn `Pipeline`
- **Four models:** Random Forest, SVM, K-Nearest Neighbors, Decision Tree
- **Imbalance-aware training:** class weights, F1-based hyperparameter tuning (`GridSearchCV`, stratified 5-fold CV)
- **Evaluation:** precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC and precision-recall curves
- **Streamlit app:** single prediction, batch CSV prediction, model switcher, adjustable failure threshold

## Project structure

```
├── notebook.py                    # EDA, preprocessing, training, evaluation
├── app.py                         # Streamlit frontend
├── model.pkl                      # Trained models + metadata used by the app
├── requirements.txt
└── README.md
```

## Dataset

Predictive maintenance dataset from Kaggle (binary target: machine failure / no failure).
Failure is rare (a few percent of rows), so accuracy alone is misleading and models are compared mainly on **recall, F1 and PR-AUC**.

> Failure-cause columns (e.g. `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) directly encode the target, so they are dropped to avoid data leakage.

## Methodology

1. **EDA:** inspect distributions, missing values, duplicates, outliers and class imbalance
2. **Cleaning:** drop IDs and leakage columns, standardise column names
3. **Feature engineering:** temperature difference, mechanical power (`torque × angular speed`), wear × torque
4. **Split:** stratified 80/20 train/test split
5. **Modelling:** each model sits in a `Pipeline` with the preprocessor, so scaling is fitted on training folds only
6. **Tuning:** `GridSearchCV` optimising F1
7. **Comparison:** all models evaluated on the same held-out test set

## Results

Fill in with your own numbers from the notebook.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | | | | | | |
| SVM | | | | | | |
| KNN | | | | | | |
| Decision Tree | | | | | | |

## Run locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Using the app

- **Single prediction:** enter machine readings and get a failure probability with a clear verdict
- **Batch prediction:** upload a CSV (column names with units such as `Torque [Nm]` are accepted) and download the predictions
- **Model performance:** view the test-set metrics for every model
- **Sidebar:** switch models and tune the failure threshold. Lowering it catches more failures at the cost of more false alarms

> The SVM outputs a decision score rather than a probability, so the threshold slider does not apply to it.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Streamlit · joblib

## Notes

- The model file must be loaded with the same scikit-learn version used for training (pinned in `requirements.txt`).
- If you change the feature engineering in the notebook, update `add_features()` in `app.py` to match.

## License

MIT
