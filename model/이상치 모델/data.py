from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import Config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
NORMAL_LABEL = 0
DEFECT_LABEL = 1

image_transform = transforms.ToTensor()


def list_image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def preprocess_image(image: Image.Image) -> Image.Image:
    """학습과 추론이 똑같이 거치는 전처리. 지금은 아무것도 하지 않는다 (필터 등을 넣을 자리)."""
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return image_transform(preprocess_image(image.convert("RGB")))


def load_image(path: Path) -> torch.Tensor:
    return image_to_tensor(Image.open(path))


def split_dataset(config: Config) -> dict[str, tuple[list[Path], np.ndarray]]:
    """정상은 학습/검증/테스트로, 결함은 검증/테스트로 나눈다. 시드가 같으면 항상 같은 분할이 나온다."""
    random_generator = np.random.default_rng(config.random_seed)

    def shuffled(paths: list[Path]) -> list[Path]:
        return [paths[index] for index in random_generator.permutation(len(paths))]

    normal_paths = shuffled(list_image_paths(config.normal_dir))
    anomaly_paths = shuffled(list_image_paths(config.anomaly_dir))

    normal_train_end = int(len(normal_paths) * config.normal_train_ratio)
    normal_validation_end = normal_train_end + int(len(normal_paths) * config.normal_validation_ratio)
    anomaly_validation_end = int(len(anomaly_paths) * config.anomaly_validation_ratio)

    def with_labels(normal: list[Path], anomaly: list[Path]) -> tuple[list[Path], np.ndarray]:
        labels = np.array([NORMAL_LABEL] * len(normal) + [DEFECT_LABEL] * len(anomaly))
        return normal + anomaly, labels

    normal_train = normal_paths[:normal_train_end]
    return {
        "train": (normal_train, np.full(len(normal_train), NORMAL_LABEL)),
        "val": with_labels(normal_paths[normal_train_end:normal_validation_end],
                           anomaly_paths[:anomaly_validation_end]),
        "test": with_labels(normal_paths[normal_validation_end:],
                            anomaly_paths[anomaly_validation_end:]),
    }