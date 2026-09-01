from pathlib import Path
import json
import re
import pandas as pd

FEATURE_DTYPES_PATH = Path(__file__).parent / "feature_dtypes.json"


def open_and_read_text(path: str):

  with open(path, "r") as f:
    raw_text = f.read()

  return raw_text


def extract_json(llm_output: str):

  match = re.search(r"\{.*\}", llm_output, re.DOTALL)
  if not match:
    raise ValueError("No JSON object found in model output")
  json_str = match.group(0)
  record = json.loads(json_str)

  with open(FEATURE_DTYPES_PATH, "r") as f:
    feature_dtypes = json.load(f)

  feature_order = list(feature_dtypes.keys())

  return record, feature_dtypes, feature_order


def dict_to_rows(data: dict, feature_order: list, feature_dtypes: dict):

  df = pd.DataFrame([data])

  df = df.reindex(columns=feature_order)

  for col in feature_order:
    df[col] = df[col].astype(feature_dtypes[col])

  return df
