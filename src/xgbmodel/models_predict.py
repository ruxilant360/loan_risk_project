import xgboost  as xgb
import pandas as pd
import shap

def load_model(model_path :str):
  model = xgb.XGBClassifier(enable_categorical=True)
  model.load_model(model_path)

  return model

def report_tuple(prob,shap):
  return prob, shap


def predict_with_shap(inference_df: pd.DataFrame, model_path: str, top_features: int):

  model = load_model(model_path)

  shap_explainer = shap.TreeExplainer(model)
  inf_shap = shap_explainer(inference_df)

  loan_prob = float(model.predict_proba(inference_df)[:, 1][0])

  feature_names = inference_df.columns.tolist()
  shap_row = inf_shap.values[0]
  value_row = inference_df.values[0]

  feature_impacts = sorted(
    zip(feature_names, shap_row, value_row),
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
