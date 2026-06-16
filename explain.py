import shap
import joblib
import pandas as pd

model = joblib.load("models/heart_model.pkl")

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)

explainer = shap.TreeExplainer(model)

def get_shap_values(patient):

    shap_values = explainer.shap_values(patient)

    return shap_values