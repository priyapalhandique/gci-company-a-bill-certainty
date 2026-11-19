"""Single source of truth for the GCI final assignment build.

All paths resolve inside ``work/``. The original assignment folder is read-only.
"""
from pathlib import Path

ACCOUNT_NAME = "PriyapalDSAI"

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent
DATA = WORK / "data"
FIGURES = WORK / "figures"
OUTPUT = WORK / "output"
SRC = WORK / "src"

CLIENT_CSV = DATA / "Client.csv"
RECORD_CSV = DATA / "Record.csv"
SOURCE_HASHES = WORK / "source_hashes.json"
NOTEBOOK = WORK / f"{ACCOUNT_NAME}.ipynb"

SEED = 42
N_SPLITS = 5
TARGET = "churn"
ID_COL = "Customer_ID"

# Fields excluded from the deployed model for governance.
SENSITIVE_FIELDS = [
    "ethnic", "marital",
    "kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17",
    "income", "creditcd", "area",
]

# Business-case assumptions (all overridable in the notebook / sensitivity table).
BASE_MARGIN = 0.60          # contribution margin on retained service revenue
BASE_PROGRAM_COST = 30.0    # USD per targeted customer
BASE_CHURN_REDUCTION = 0.15 # relative reduction in churn among treated customers
HORIZON_MONTHS = 12

# Submission guardrails.
MAX_SLIDES = 15
MAX_PDF_BYTES = 10 * 1024 * 1024
