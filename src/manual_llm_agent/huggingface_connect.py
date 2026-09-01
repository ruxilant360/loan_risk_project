import huggingface_hub
import numpy as np
import langchain
import pandas as pd
import os
from pathlib import Path


from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Optional, List
from pydantic import BaseModel, Field
import dotenv

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
#from tests import text, dummy_feat_impact

dotenv.load_dotenv(Path(__file__).parent / "HF_TOKEN.env")
hf_token = os.environ.get("HF_TOKEN")

text = "APEX NATIONAL BANK Consumer Credit Underwriting & Personal Financial ProfileApplicant ProfileFieldDetailAge / Gender22 / FemaleEducation LevelMaster’s DegreeHousing StatusRentEmployment Experience0 Years (Recent Graduate / Entry-Level)Annual Gross Income$71,948.00Credit & Risk AssessmentFieldDetailUnderwriter AssessmentCredit Score561Subprime / FairCredit History Length3.0 YearsLimited / Thin FilePrior Loan DefaultsNone on filePositiveLoan Application DetailsFieldDetailRequested Amount$35,000.00Loan Purpose / IntentPersonalOffered / Stated Interest Rate16.02% APRLoan-to-Income Ratio48.65% (0.49)Underwriting Summary & Risk IndicatorsHigh Loan-to-Income Exposure: The requested loan of $35,000 represents nearly 49% of the applicant’s total annual gross income for an unsecured personal loan.Credit Score vs. Education Paradox: While the applicant holds an advanced degree and strong starting income ($71,948.00), a credit score of 561 combined with a 3-year credit history indicates elevated credit risk or high existing credit utilization.Recommendation: Requires manual underwriter review, proof of employment/stipend, or a qualified co-signer prior to loan disbursement due to subprime credit tier and high exposure ratio."

dummy_feat_impact = """ Predicted probability of loan approval: 73.2%
Top contributing factors:
- credit_score = 710: increased approval likelihood (impact score: +0.421)
- debt_to_income = 0.45: decreased approval likelihood (impact score: -0.187)
- income = 52000: increased approval likelihood (impact score: +0.093)
- employment_years = 2: decreased approval likelihood (impact score: -0.061)
- loan_amount = 25000: decreased approval likelihood (impact score: -0.034) """


class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class EducationEnum(str, Enum):
    HIGH_SCHOOL = "High School"
    ASSOCIATE = "Associate"
    BACHELOR = "Bachelor"
    MASTER = "Master"
    DOCTORATE = "Doctorate"


class HomeOwnershipEnum(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanIntentEnum(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"


class LoanDefaultsEnum(str, Enum):
    YES = "Yes"
    NO = "No"


class CreditRiskApplicant(BaseModel):
    person_age: float = Field(
        ...,
        ge=18.0,
        le=100.0,
        description="The age of the loan applicant in years (e.g., 28.0)."
    )
    
    person_gender: GenderEnum = Field(
        ...,
        description="Gender of the applicant. Must be one of: 'male', 'female', 'other'."
    )
    
    person_education: EducationEnum = Field(
        ...,
        description="Highest level of education completed by the applicant (e.g., 'Bachelor', 'High School', 'Master')."
    )
    
    person_income: float = Field(
        ...,
        ge=0.0,
        description="Total annual pre-tax income of the applicant in dollars. If stated monthly or weekly, convert to annual."
    )
    
    person_emp_exp: int = Field(
        ...,
        ge=0,
        description="Total professional employment or work experience in complete years."
    )
    
    person_home_ownership: HomeOwnershipEnum = Field(
        ...,
        description="Current housing situation. Must be one of: 'RENT', 'OWN', 'MORTGAGE', or 'OTHER'."
    )
    
    loan_amnt: float = Field(
        ...,
        gt=0.0,
        description="The requested loan amount in dollars."
    )
    
    loan_intent: LoanIntentEnum = Field(
        ...,
        description="The stated purpose of the loan (e.g., 'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION')."
    )
    
    loan_int_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="The annual interest rate for the loan expressed as a percentage value (e.g., 11.25 for 11.25%). Null if not yet determined."
    )
    
    loan_percent_income: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Loan amount divided by annual income, expressed as a decimal ratio between 0.0 and 1.0 (e.g., 0.25 for 25%)."
    )
    
    cb_person_cred_hist_length: float = Field(
        ...,
        ge=0.0,
        description="Duration of the applicant's credit history on file with credit bureaus, measured in years."
    )
    
    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="Standard credit score rating (e.g., FICO or VantageScore), typically an integer ranging from 300 to 850."
    )
    
    previous_loan_defaults_on_file: LoanDefaultsEnum = Field(
        ...,
        description="Indicator of whether the applicant has any recorded historical loan defaults on file. 'Yes' or 'No'."
    )
    
    loan_status: Optional[int] = Field(
        None,
        description="Historical or target loan approval/default status (0 for non-default/approved, 1 for default/rejected). Null during inference."
    )

def instruct_and_extract_schema(text_from_function : str):

  sysprompt = """You are an expert data extraction engine for loan applicant profiles.

Analyze the provided input text and extract structured information matching the schema below exactly. The output will be validated against a Pydantic model, so field names, types, and enum values must match precisely.

Guidelines:
1. Extract exact facts, numbers, dates, and explicit relationships only. Do not infer or hallucinate unstated values.
2. If a required field cannot be determined from the text, use your best defensible reading of the text — do not invent numbers. For optional fields, use null.
3. Normalize numbers, currency, and units as specified per field below.

Fields:

- person_age (float, 18.0–100.0): Applicant's age in years.
- person_gender (enum: "male" | "female" | "other"): Applicant's stated gender.
- person_education (enum: "High School" | "Bachelor" | "Master" | ...): Highest completed education level. Use the exact enum label, not a paraphrase.
- person_income (float, >= 0.0): Total annual pre-tax income in dollars. If stated as monthly or weekly, convert to annual (monthly × 12, weekly × 52).
- person_emp_exp (int, >= 0): Total employment experience in complete years. Round down partial years.
- person_home_ownership (enum: "RENT" | "OWN" | "MORTGAGE" | "OTHER"): Current housing situation.
- loan_amnt (float, > 0.0): Requested loan amount in dollars.
- loan_intent (enum: "PERSONAL" | "EDUCATION" | "MEDICAL" | "VENTURE" | "HOMEIMPROVEMENT" | "DEBTCONSOLIDATION"): Stated purpose of the loan.
- loan_int_rate (float or null, 0.0–100.0): Annual interest rate as a percentage (e.g. 11.25 for 11.25%). Null if not stated or not yet determined.
- loan_percent_income (float or null, 0.0–1.0): loan_amnt / person_income as a decimal ratio. Compute it yourself only if both values are present in the text; otherwise null.
- cb_person_cred_hist_length (float, >= 0.0): Length of credit history on file, in years.
- credit_score (int, 300–850): Standard FICO/VantageScore-style credit score.
- previous_loan_defaults_on_file (enum: "Yes" | "No"): Whether the applicant has a recorded prior default.
- loan_status (int or null): Historical/target status — 0 for non-default/approved, 1 for default/rejected. Null during inference-only extraction (no ground truth in the text).

Return only the fields defined above. Do not add commentary or extra keys."""

  hg_llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    max_new_tokens= 512,
    temperature = 0.1,
    huggingfacehub_api_token = hf_token
  )

  chat_model = ChatHuggingFace(llm=hg_llm)

  extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", sysprompt),("human", """Extract the structured data from the following text:
    ---
    {extracted_text}
    ---"""
        ) ])

  structured_model =  chat_model.with_structured_output(CreditRiskApplicant, method="json_schema")

  chain = extraction_prompt | structured_model
  result = chain.invoke({"extracted_text": text_from_function})

  return result


def format_for_llm(probability, feature_impacts):

    lines = [f"Predicted probability of loan approval: {probability:.1%}"]

    for feat, impact, val in feature_impacts:
        direction = "increased" if impact > 0 else "decreased"
        lines.append(f"{feat}={val}: {direction} approval likelihood with impact score of {impact:.4f}")

    return '\n'.join(lines)


def instruct_and_interpret_report(form_lines):

    system_prompt = """You are a credit underwriting assistant. You will be given a model's predicted 
probability of loan approval along with the top factors (SHAP values) that 
influenced that prediction, in order of importance.

Your job is to write a short, clear explanation of the decision for a loan 
officer or borrower-facing summary. Follow these rules:

1. State the overall recommendation first (e.g. "This application is likely 
   to be approved" / "flagged for review" / "likely to be denied") based on 
   the probability. Use your judgment on thresholds unless a specific policy 
   threshold is given to you.
2. Explain the 2-3 most influential factors in plain language, without 
   mentioning "SHAP," "impact scores," or any ML jargon. Translate impact 
   direction into borrower-relevant language (e.g. "a strong credit score 
   supported approval" rather than "credit_score had impact +0.421").
3. Do not invent numbers, factors, or values that were not provided to you.
4. Keep the explanation to 3-5 sentences.
5. Do not state or imply this is a final decision — frame it as a model-based 
   recommendation subject to human review.
6. If any factor might reflect a protected characteristic (e.g. age, zip code 
   as a proxy for race) rather than a legitimate credit factor, note that this 
   feature should be reviewed for fair lending compliance rather than stated 
   as a plain justification."""

    human_prompt = """You have this formatted output extract the details and explain :""" + form_lines

    hg_llm = HuggingFaceEndpoint(
        repo_id = "meta-llama/Llama-3.1-8B-Instruct",
        task = "text-generation",
        max_new_tokens= 512,
        temperature = 0.3,
        huggingfacehub_api_token = hf_token
      )

    chat_model = ChatHuggingFace(llm=hg_llm)
    
    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)

    ])
    
    return response.content


if __name__ == "__main__":
    result = instruct_and_extract_schema(text_from_function=text)
    print(result)

    llm_output = instruct_and_interpret_report(form_lines=dummy_feat_impact)
    print(llm_output)