import xgboost  as xgb
import numpy as np
import pandas as pd
import shap

def load_model(model_path :str):
  model = xgb.XGBClassifier()
  model = model.load_model(model_path)

  return model

def report_tuple(prob,shap):
  return prob, shap


def predict_with_shap(test_df_path: str, inference_df_path:str, model_path:str, top_features:int):

  test_df = pd.read_csv(test_df)
  inference_df = pd.read_csv(inference_df_path)
  model = xgb.XGBClassifier.load_model(model_path)

  shap_explainer = shap.TreeExplainer(model)
  shap_values = shap_explainer(test_df)

  inf_values = inference_df.values
  loan_prob = float(model.predict_proba(inf_values)[:, 1])

  inf_shap = shap_explainer(inference_df)
  values = inf_shap.values
  feature_names = inference_df.columns.tolist()

  feature_impacts = sorted( 
    zip(feature_names, values, inf_values),
    key= lambda x: abs(x[1]), reverse = True
    )[:top_features]

  drivers = [
    {
      "feature":feat,
      "feat_value": str(val),
      "shap_impact": round(float(impact), 4),
      "push_dir": "low_risk" if impact > 0 else "high_risk"
    } 
    for feat, impact, val in feature_impacts
  ]

  report = report_tuple(loan_prob, drivers)

  return report, feature_impacts
