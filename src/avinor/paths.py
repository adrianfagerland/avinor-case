from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CASE_DIR = PROJECT_ROOT / "case"
ARTIFACTS_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = ARTIFACTS_DIR / "models"
FEATURE_CACHE_DIR = ARTIFACTS_DIR / "feature_cache"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_RAW_DIR = EXTERNAL_DATA_DIR / "raw"
EXTERNAL_PROCESSED_DIR = EXTERNAL_DATA_DIR / "processed"
SCHOOL_HOLIDAYS_PATH = EXTERNAL_PROCESSED_DIR / "school_holidays.csv"

for directory in (
    ARTIFACTS_DIR,
    LOG_DIR,
    MODELS_DIR,
    FEATURE_CACHE_DIR,
    EXTERNAL_DATA_DIR,
    EXTERNAL_RAW_DIR,
    EXTERNAL_PROCESSED_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
