"""Project settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"          # RASFF Window export(s) (.xlsx or .csv)
CLEAN_FILE = ROOT / "data" / "notifications_clean.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Column names in the RASFF Window export (October 2026 format).
COL_REFERENCE = "reference"
COL_TYPE = "type"
COL_DATE = "date"
COL_CATEGORY = "category"        # product category, e.g. "nuts, nut products and seeds"
COL_ORIGIN = "origin"            # country of origin (several countries are comma-separated)
COL_NOTIFYING = "notifying_country"
COL_RISK = "risk_decision"
COL_HAZARDS = "hazards"          # e.g. "Aflatoxin B1 - {mycotoxins},aflatoxin total - {mycotoxins}"

KEEP_TYPES = ["food"]            # analyse food notifications only (drop feed, food contact material, ...)
HAZARD_LEVEL = "hazard_class"    # "hazard_class" (e.g. mycotoxins, 28 classes) or "hazard_name" (e.g. aflatoxin B1)

N_BOOTSTRAP = 1000               # bootstrap replications for uncertainty
SEED = 42
TOP_K = 10                       # how many nodes to report in rankings
