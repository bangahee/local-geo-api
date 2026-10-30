"""The real model backend. No remote inference or paid services."""

import math
import os
from pathlib import Path


DEFAULT_REVISION = "40a931123d2c6451ef7da1d37cf68cded0ec4d35"


class GeoCLIPBackend:
    def __init__(self):
        self.model = None
        self.state = "unloaded"

    def load(self):
        if self.model is not None:
            return
        self.state = "loading"
        try:
            from geoclip import GeoCLIP

            model = GeoCLIP.from_pretrained(
                "VicenteVivanco/geoclip-geolocation",
                revision=os.getenv("MODEL_REVISION") or DEFAULT_REVISION,
                cache_dir=os.getenv("HF_HOME") or None,
                local_files_only=os.getenv("OFFLINE", "0") == "1",
                device=os.getenv("DEVICE", "cpu"),
            )
            self.model = model
            self.state = "ready"
        except Exception:
            self.state = "failed"
            raise

    def predict(self, image_path: Path, top_k: int):
        self.load()
        coordinates, scores = self.model.predict(str(image_path), top_k=top_k)
        predictions = []
        for coordinate, score in zip(coordinates.tolist(), scores.tolist(), strict=True):
            lat, lon = map(float, coordinate)
            score = float(score)
            if not (
                math.isfinite(lat) and -90 <= lat <= 90
                and math.isfinite(lon) and -180 <= lon <= 180
                and math.isfinite(score) and 0 <= score <= 1
            ):
                raise ValueError("Model produced invalid coordinates or scores")
            predictions.append({"coordinates": [lat, lon], "score": score})
        if len(predictions) != top_k:
            raise ValueError("Model returned the wrong number of predictions")
        return predictions
