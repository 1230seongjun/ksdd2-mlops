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


MERGE_GAP = 10   # 같은 색 박스 사이 간격이 이 픽셀 이하면 하나로 합침 (WISE 박스 만들 때와 같은 값)


def box_gap(a: tuple, b: tuple) -> int:
    """박스 (x0, y0, x1, y1) 사이 빈 칸 (겹치면 0)"""
    return max(max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]), 0)


def merge_boxes(boxes: list[tuple]) -> list[tuple]:
    """가까운 박스를 더 이상 안 바뀔 때까지 합침"""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if box_gap(boxes[i], boxes[j]) <= MERGE_GAP:
                    a, b = boxes[i], boxes.pop(j)
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    merged = True
                    break
            if merged:
                break
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def predicted_regions(prediction_map: np.ndarray, probs: np.ndarray) -> list[dict]:
    """색마다 칠한 덩어리(8방향 연결)에 박스 1개, 가까운 박스는 합침.
    bbox = COCO [x, y, 너비, 높이] (WISE 학습 박스와 같음), score = 박스 안 그 색 픽셀의 그 색 확률 평균"""
    from scipy import ndimage
    regions = []
    for color in np.unique(prediction_map[prediction_map > 0]):
        painted = prediction_map == color
        labeled, _ = ndimage.label(painted, structure=np.ones((3, 3)))
        raw = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(labeled)]
        for x0, y0, x1, y1 in merge_boxes(raw):
            inside = painted[y0:y1, x0:x1]
            regions.append({"bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                            "score": round(float(probs[color, y0:y1, x0:x1][inside].mean()), 4)})
    return regions


@torch.no_grad()
def classify_images_with_regions(model: torch.nn.Module, images: list[Image.Image], config: Config | InferenceSettings) -> list[tuple[int, float | None, list[dict]]]:
    """이미지마다 (예측 색 번호, confidence, regions).
    confidence = 이미지 색으로 칠한 픽셀들의 그 색 softmax 확률 평균 (보정 안 한 모니터링용 값). 안 칠했으면 None"""
    height, width = config.image_size
    results = []
    for start in range(0, len(images), config.batch_size):
        batch = np.stack([pad(image_to_array(image, config), config) for image in images[start:start + config.batch_size]])
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(config.device)
        probs = model(tensor).softmax(1).cpu().numpy()[:, :, :height, :width]   # 패딩 잘라냄
        for prob in probs:
            prediction = prob.argmax(0)
            color = predicted_color(prediction)
            confidence = round(float(prob[color][prediction == color].mean()), 4) if color else None
            results.append((color, confidence, predicted_regions(prediction, prob)))
    return results
