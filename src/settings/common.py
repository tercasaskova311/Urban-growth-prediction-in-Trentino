# src/settings/common.py
from pathlib import Path
import yaml
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project_root/src/settings -> up two

def load_config(config_filename: str = "config.yaml"):
    """
    Load the single config.yaml from the project root.
    Returns a dict.
    """
    config_path = PROJECT_ROOT / config_filename
    if not config_path.is_file():
        # fallback: maybe user ran script from project root already
        alt = Path.cwd() / config_filename
        if alt.is_file():
            config_path = alt
        else:
            raise FileNotFoundError(f"Could not find {config_filename} at {config_path} or {alt}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # inject secrets from environment if needed (example)
    # cfg.setdefault("secrets", {})["GEE_PROJECT_ID"] = os.environ.get("GEE_PROJECT_ID")
    return cfg
