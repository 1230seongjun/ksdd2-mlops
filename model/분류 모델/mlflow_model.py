import json
from pathlib import Path

import mlflow.models
import mlflow.pyfunc
import numpy as np
import pandas as pd
from PIL import Image

from model import classify_images_with_regions, load_classifier

NORMAL_COLOR = "정상"   # 아무것도 안 칠했을 때 (color_id 0)


class ColorClassifierModel(mlflow.pyfunc.PythonModel):
    """MLflow 모델 레지스트리에 등록되는 모델. 입력은 RGB 이미지 배열 (장수, 높이, 너비, 3) uint8."""

    def load_context(self, context):
        self.model, self.settings = load_classifier(Path(context.artifacts["classifier"]), device="cpu")

    def predict(self, context, model_input, params=None):
        images = [Image.fromarray(np.asarray(array, dtype=np.uint8)) for array in model_input]
        results = classify_images_with_regions(self.model, images, self.settings)
        color_ids = [color for color, _, _ in results]
        return pd.DataFrame({
            "color_id": color_ids,
            "color": [self.settings.classes[color] if color else NORMAL_COLOR for color in color_ids],
            "confidence": pd.Series([confidence for _, confidence, _ in results], dtype=object),   # 안 칠했으면 None (NaN 아님 → DB NULL)
            "regions": [json.dumps(regions, ensure_ascii=False) for _, _, regions in results],   # [{"bbox": [x, y, 너비, 높이], "score"}] JSON 문자열
        })


mlflow.models.set_model(ColorClassifierModel())