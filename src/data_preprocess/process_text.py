import numpy as np
import os
from pathlib import Path
import json
import re
import pandas as pd

#df = pd.read_csv(r"C:\Users\USER\Downloads\Projects\credit_risk_scoring\notebooks\data\loan_data.csv")


def open_and_read_text(path : str):

  with open(path, "r") as f:
    raw_text = f.read

  return raw_text


def extract_json(llm_output:str):

  match = re.search(r"\{.*\}", llm_output, re.DOTALL)
  if not match:
    raise ValueError("No JSON object found in model output")
  json_str = match.group(0)
  record = json.loads(json_str)

  with open("feature_dtypes.json" , "r") as f:
    feature_dtypes = json.load(f)

  feature_order = list(feature_dtypes.keys())

  return record, feature_dtypes, feature_order
  

#feature_order = ['person_age', 'person_gender', 'person_education', 'person_income','person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent','loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file']



def dict_to_rows(data: dict, feature_order : list, feature_dtypes : dict ):

  df = pd.DataFrame([data])

  df = df.reindex(columns=feature_order)

  return df.values









