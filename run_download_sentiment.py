# src/data_acquisition/run_download.py
import yaml
from src.settings.common import load_config
from src.data_acquisition.sentinel2_downloader import Sentinel2Downloader
import os

def main():
    cfg = load_config()   # single source of truth
    print("Loaded config. Region:", cfg["study_area"]["name"])

    # Optionally read secrets from env (safe practice)
    gee_project = os.environ.get("GEE_PROJECT_ID")
    if gee_project:
        cfg.setdefault("secrets", {})["GEE_PROJECT_ID"] = gee_project

    downloader = Sentinel2Downloader(config=cfg)

    info = downloader.get_collection_info()
    print("Collection info:", info)

if __name__ == "__main__":
    main()
