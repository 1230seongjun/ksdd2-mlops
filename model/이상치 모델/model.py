import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from config import Config, InferenceSettings
from data import image_to_tensor, load_image

POOL_SIZE = 3   # 주변 패치 정보를 섞는 평균 풀링 크기
MODEL_INFO = {  # MLflow 파라미터와 저장하는 모델 정보에 같이 쓴다
    "backbone": "wide_resnet50_2_v1", "feature_layers": "23", "pool_size": POOL_SIZE,
    "sampling": "coreset", "score_mode": "max", "normalize": False,
}


# ---------- 특징 추출 ----------
class FeatureExtractor(nn.Module):
    """사전학습 Wide ResNet50(V1 가중치)의 layer2, layer3 출력을 합쳐 패치별 특징을 만든다."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None   # 저장된 가중치를 불러올 때는 받을 필요가 없다
        backbone = wide_resnet50_2(weights=weights).eval()
        self.early_layers = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        layer2_features = self.layer2(self.early_layers(images))
        layer3_features = self.layer3(layer2_features)

        # 주변 패치 정보를 섞어 위치 변화에 덜 민감하게 만든다
        layer2_features = functional.avg_pool2d(layer2_features, kernel_size=POOL_SIZE, stride=1, padding=POOL_SIZE // 2)
        layer3_features = functional.avg_pool2d(layer3_features, kernel_size=POOL_SIZE, stride=1, padding=POOL_SIZE // 2)
        layer3_features = functional.interpolate(
            layer3_features, size=layer2_features.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat([layer2_features, layer3_features], dim=1)  # (배치, 채널, 높이, 너비)


def extract_tensor_features(extractor: FeatureExtractor, images: list[torch.Tensor], config: Config | InferenceSettings
                            ) -> list[torch.Tensor]:
    """이미지 텐서 묶음(배치) 하나에서 이미지마다 (패치 수, 채널 수) 모양의 특징 텐서를 만든다."""
    features = extractor(torch.stack(images).to(config.device))  # (배치, 채널, 높이, 너비)
    return list(features.flatten(start_dim=2).permute(0, 2, 1).cpu())


def extract_patch_features(extractor: FeatureExtractor, paths: list[Path], config: Config | InferenceSettings
                           ) -> list[torch.Tensor]:
    """이미지 파일마다 (패치 수, 채널 수) 모양의 특징 텐서를 반환한다."""
    features_per_image = []
    for start in range(0, len(paths), config.batch_size):
        batch = [load_image(path) for path in paths[start:start + config.batch_size]]
        features_per_image += extract_tensor_features(extractor, batch, config)
    return features_per_image


# ---------- 학습: 정상 패치 특징 저장 ----------
@torch.no_grad()
def greedy_coreset(features: torch.Tensor, size: int, config: Config) -> torch.Tensor:
    """k-center greedy: 이미 고른 패치들에서 가장 먼 패치를 하나씩 추가한다 (PatchCore 논문 방식).
    거리 계산은 무작위 투영으로 차원을 줄여 빠르게 한다."""
    if size >= len(features):
        return features
    generator = torch.Generator(device="cpu").manual_seed(config.model_seed)
    projection = torch.randn(features.shape[1], config.coreset_projection_dim, generator=generator).to(features.device)
    projected = features @ projection

    selected = [int(torch.randint(len(features), (1,), generator=generator))]
    min_distances = torch.cdist(projected, projected[selected[-1]].unsqueeze(0)).squeeze(1)
    for step in range(1, size):
        next_index = int(torch.argmax(min_distances))
        selected.append(next_index)
        distances = torch.cdist(projected, projected[next_index].unsqueeze(0)).squeeze(1)
        min_distances = torch.minimum(min_distances, distances)
        if step % 5000 == 0:
            print(f"  coreset {step}/{size}")
    return features[torch.tensor(selected, device=features.device)]


def build_memory_bank(extractor: FeatureExtractor, normal_paths: list[Path], config: Config) -> torch.Tensor:
    """이미지마다 일부 패치를 무작위로 골라 후보를 모은 뒤, coreset 으로 max_memory_bank_size 개를 고른다."""
    if not normal_paths:
        raise FileNotFoundError(f"학습용 정상 이미지를 찾지 못했습니다: {config.normal_dir}")
    patches_per_image = -(-config.coreset_pool_size // len(normal_paths))  # 올림 나눗셈
    sampled_features = []
    for start in range(0, len(normal_paths), config.batch_size):
        batch_paths = normal_paths[start:start + config.batch_size]
        for features in extract_patch_features(extractor, batch_paths, config):
            selected_indices = torch.randperm(len(features))[:patches_per_image]
            sampled_features.append(features[selected_indices])
    pool = torch.cat(sampled_features).to(config.device)
    print(f"coreset 후보 패치: {tuple(pool.shape)} → 목표 {config.max_memory_bank_size}개")
    return greedy_coreset(pool, config.max_memory_bank_size, config)


# ---------- 이상 점수 ----------
@torch.no_grad()
def compute_anomaly_score(patch_features: torch.Tensor, memory_bank: torch.Tensor,
                          config: Config | InferenceSettings) -> float:
    """패치별로 가장 가까운 정상 패치까지의 거리를 구하고, 그중 최댓값을 이미지 점수로 사용한다."""
    patch_features = patch_features.to(config.device)
    nearest_distances = torch.cat([
        torch.cdist(patch_features[start:start + config.query_chunk_size], memory_bank).min(dim=1).values
        for start in range(0, len(patch_features), config.query_chunk_size)
    ])
    return nearest_distances.max().item()


# ---------- 모델 ----------
class PatchCoreDetector:
    """정상 이미지로 메모리 뱅크를 만들고(fit) 이미지의 이상 점수를 계산한다(score). 학습과 추론이 같이 쓴다."""

    def __init__(self, config: Config | InferenceSettings, pretrained: bool = True):
        self.config = config
        self.extractor = FeatureExtractor(pretrained).to(config.device)
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float | None = None   # 이 값 이상이면 결함으로 판정 (학습 후 검증 데이터로 정한다)

    def fit(self, normal_paths: list[Path]) -> None:
        self.memory_bank = build_memory_bank(self.extractor, normal_paths, self.config)

    def score(self, paths: list[Path]) -> np.ndarray:
        scores = []
        for start in range(0, len(paths), self.config.batch_size):
            for features in extract_patch_features(self.extractor, paths[start:start + self.config.batch_size], self.config):
                scores.append(compute_anomaly_score(features, self.memory_bank, self.config))
        return np.array(scores)

    def score_images(self, images: list[Image.Image]) -> np.ndarray:
        """이미 읽어 둔 이미지(PIL)의 점수. 추론 서버가 요청으로 받은 이미지에 쓴다."""
        scores = []
        for start in range(0, len(images), self.config.batch_size):
            batch = [image_to_tensor(image) for image in images[start:start + self.config.batch_size]]
            for features in extract_tensor_features(self.extractor, batch, self.config):
                scores.append(compute_anomaly_score(features, self.memory_bank, self.config))
        return np.array(scores)

    def save(self, directory: Path) -> None:
        """메모리 뱅크, 임계값, 백본 가중치(실제로 쓰는 층만, 약 100MB)를 저장한다. 불러올 때 인터넷이 필요 없다."""
        if self.memory_bank is None or self.threshold is None:
            raise ValueError("메모리 뱅크와 임계값이 있어야 저장할 수 있습니다 (fit 과 임계값 설정을 먼저 하세요)")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.memory_bank.cpu(), directory / "memory_bank.pt")
        torch.save(self.extractor.state_dict(), directory / "extractor.pt")
        info = {**MODEL_INFO, "threshold": float(self.threshold),
                "batch_size": self.config.batch_size, "query_chunk_size": self.config.query_chunk_size}
        (directory / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, config: Config | InferenceSettings) -> "PatchCoreDetector":
        detector = cls(config, pretrained=False)
        detector.extractor.load_state_dict(torch.load(directory / "extractor.pt", map_location=config.device))
        detector.memory_bank = torch.load(directory / "memory_bank.pt", map_location=config.device)
        detector.threshold = json.loads((directory / "model_info.json").read_text(encoding="utf-8"))["threshold"]
        return detector

    @classmethod
    def load_for_inference(cls, directory: Path, device: str = "cpu") -> "PatchCoreDetector":
        """저장된 모델 정보만으로 불러온다 (학습용 설정 파일이 필요 없다)."""
        info = json.loads((directory / "model_info.json").read_text(encoding="utf-8"))
        settings = InferenceSettings(batch_size=info["batch_size"], query_chunk_size=info["query_chunk_size"], device=device)
        return cls.load(directory, settings)