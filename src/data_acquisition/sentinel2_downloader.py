# src/data_acquisition/sentinel2_downloader.py
from pathlib import Path

class Sentinel2Downloader:
    def __init__(self, config: dict = None, bbox=None, start_date=None, end_date=None):
        """
        Simple, config-driven initializer.
        You can override bbox/start/end via explicit args (handy for tests).
        """
        self.config = config or {}
        study_area = self.config.get("study_area", {})
        time_period = self.config.get("time_period", {})

        # Prefer explicit args => fall back to config
        self.bbox = bbox or study_area.get("bbox")
        self.start_date = start_date or time_period.get("start_date")
        self.end_date = end_date or time_period.get("end_date")

        s2 = self.config.get("sentinel2", {})
        self.cloud_cover = s2.get("cloud_cover_threshold", 20)
        self.scale = s2.get("scale", 10)
        self.indices = s2.get("indices", [])

        out = self.config.get("output", {})
        self.output_dir = Path(out.get("data_dir", "data")) / "sentinel2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # stub methods — replace with your real implementations
    def get_collection_info(self):
        # use self.bbox, self.start_date, self.end_date, etc.
        return {"count": 0}

    def get_image_collection(self, cloud_cover=None):
        raise NotImplementedError

    def aggregate_temporal(self, collection, period="month"):
        raise NotImplementedError

    def export_to_geotiff(self, image, filename, scale, bands, folder):
        raise NotImplementedError
