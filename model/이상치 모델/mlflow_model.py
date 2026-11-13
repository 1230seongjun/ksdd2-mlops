import mlflow.pyfunc
import mlflow.models
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from model import PatchCoreDetector


class PatchCoreModel(mlflow.pyfunc.PythonModel):
    """MLflow 모델 레지스트리에 등록되는 모델. 입력은 RGB 이미지 배열 (장수, 높이, 너비, 3) uint8."""

    def load_context(self, context):
        self.detector = PatchCoreDetector.load_for_inference(Path(context.artifacts["detector"]), device="cpu")

    def predict(self, context, model_input, params=None):
        images = [Image.fromarray(np.asarray(array, dtype=np.uint8)) for array in model_input]
        scores = self.detector.score_images(images)
        threshold = self.detector.threshold
        return pd.DataFrame({"anomaly_score": scores, "is_defect": scores >= threshold, "threshold": threshold})


mlflow.models.set_model(PatchCoreModel())