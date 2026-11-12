import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from config import Config, InferenceSettings
from data import image_to_array, pad


def build_model(config: Config | InferenceSettings, pretrained: bool) -> torch.nn.Module:
    """U-Net. 픽셀마다 배경 + 라벨 색 중 하나를 고른다. 저장된 가중치를 읽을 때는 pretrained=False (ImageNet 가중치를 내려받지 않음)."""
    return smp.Unet(config.encoder, encoder_weights=config.encoder_weights if pretrained else None, classes=len(config.classes))


def save_classifier(model: torch.nn.Module, config: Config, directory: Path) -> None:
    """가중치와 추론에 필요한 설정만 저장한다. 불러올 때 학습용 설정 파일도, 인터넷도 필요 없다."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "classifier.pt")
    info = {"encoder": config.encoder, "classes": config.classes, "image_height": config.image_height,
            "image_width": config.image_width, "batch_size": config.batch_size}
    (directory / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def load_classifier(directory: Path, device: str = "cpu") -> tuple[torch.nn.Module, InferenceSettings]:
    """저장된 모델 정보만으로 불러온다. 반환: (모델, 추론 설정)"""
    info = json.loads((directory / "model_info.json").read_text(encoding="utf-8"))
    settings = InferenceSettings(**info, device=device)
    model = build_model(settings, pretrained=False)
    model.load_state_dict(torch.load(directory / "classifier.pt", map_location=device))
    return model.to(device).eval(), settings


def predicted_color(prediction_map: np.ndarray) -> int:
    """이미지 1장의 예측 색 = 결함으로 칠한 픽셀 중 가장 많은 색 (없으면 0 = 정상)"""
    painted = prediction_map[prediction_map > 0]
    return int(np.bincount(painted).argmax()) if painted.size else 0


@torch.no_grad()
def classify_images(model: torch.nn.Module, images: list[Image.Image], config: Config | InferenceSettings) -> list[tuple[int, int]]:
    """이미지마다 (예측 색 번호, 결함으로 칠한 픽셀 수). 색 번호 0 = 아무것도 안 칠함"""
    height, width = config.image_size
    results = []
    for start in range(0, len(images), config.batch_size):
        batch = np.stack([pad(image_to_array(image, config), config) for image in images[start:start + config.batch_size]])
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(config.device)
        maps = model(tensor).argmax(1).cpu().numpy()[:, :height, :width]   # 패딩 잘라냄
        results += [(predicted_color(prediction), int((prediction > 0).sum())) for prediction in maps]
    return results