from pathlib import Path

PROJECT_PATH = Path(__file__).parent

CONFIG_PATH = PROJECT_PATH / "configs"
MODEL_CONFIGS_PATH = CONFIG_PATH / "model_configs"
TRAINED_PATH = PROJECT_PATH / "models/trained"
Path(MODEL_CONFIGS_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAINED_PATH).mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_PATH / "assets"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(INTERIM_DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
