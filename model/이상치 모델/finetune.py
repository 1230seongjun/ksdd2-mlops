"""정상 이미지에 가짜 결함을 그려 넣고, 패치가 가짜 결함인지 맞히도록 백본 일부(layer3, layer2)를 학습한다.
에폭마다 그 백본으로 메모리 뱅크를 다시 만들어 val AP 를 재고, 가장 좋았던 에폭의 가중치를 남긴다(0 = 파인튜닝 전)."""
import copy
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score

from config import Config
from data import load_image
from model import PatchCoreDetector

MAX_POSITIVE_WEIGHT = 50.0   # 결함 패치가 아주 적을 때 손실 가중치가 너무 커지지 않게 막는 상한


# ---------- 가짜 결함 ----------
def draw_line_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """가늘고 살짝 휜 선(긁힘) 마스크. 값은 0~1."""
    mask = np.zeros((height, width), dtype=np.float32)
    length, angle = rng.uniform(15, 120), rng.uniform(0, np.pi)
    thickness, bend = rng.uniform(1.0, 3.0), rng.uniform(-0.01, 0.01)
    center_y, center_x = rng.uniform(0, height), rng.uniform(0, width)
    offsets = np.linspace(-length / 2, length / 2, int(length * 2))
    xs = center_x + offsets * np.cos(angle) - bend * offsets ** 2 * np.sin(angle)
    ys = center_y + offsets * np.sin(angle) + bend * offsets ** 2 * np.cos(angle)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    mask[ys[inside].round().astype(int).clip(0, height - 1), xs[inside].round().astype(int).clip(0, width - 1)] = 1.0
    mask = gaussian_filter(mask, sigma=thickness / 2)
    return mask / max(mask.max(), 1e-6)


def draw_blob_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """얼룩/점 같은 타원 마스크. 값은 0~1."""
    radius_a, radius_b, angle = rng.uniform(2, 14), rng.uniform(2, 14), rng.uniform(0, np.pi)
    center_y, center_x = rng.uniform(0, height), rng.uniform(0, width)
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    shifted_y, shifted_x = grid_y - center_y, grid_x - center_x
    rotated_x = shifted_x * np.cos(angle) + shifted_y * np.sin(angle)
    rotated_y = -shifted_x * np.sin(angle) + shifted_y * np.cos(angle)
    mask = ((rotated_x / radius_a) ** 2 + (rotated_y / radius_b) ** 2 <= 1.0).astype(np.float32)
    mask = gaussian_filter(mask, sigma=rng.uniform(0.7, 1.5))
    return mask / max(mask.max(), 1e-6)


def random_defect_color(rng: np.random.Generator) -> np.ndarray:
    kind = rng.choice(["bright", "dark", "random"])
    if kind == "bright":
        return np.clip(rng.uniform(0.8, 1.0) + rng.uniform(-0.05, 0.05, 3), 0, 1)
    if kind == "dark":
        return np.clip(rng.uniform(0.0, 0.2) + rng.uniform(-0.05, 0.05, 3), 0, 1)
    return rng.uniform(0, 1, 3)


def add_fake_defects(image: torch.Tensor, rng: np.random.Generator, config: Config) -> tuple[torch.Tensor, np.ndarray]:
    """(가짜 결함을 그린 이미지, 결함 위치 마스크)를 반환한다."""
    _, height, width = image.shape
    canvas = image.numpy().copy()
    combined_mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(rng.integers(1, config.max_fake_defects + 1))):
        soft_mask = draw_line_mask(height, width, rng) if rng.random() < 0.5 else draw_blob_mask(height, width, rng)
        alpha = soft_mask * rng.uniform(0.5, 1.0)
        color = random_defect_color(rng).astype(np.float32)[:, None, None]
        canvas = canvas * (1 - alpha) + color * alpha
        combined_mask = np.maximum(combined_mask, soft_mask)
    return torch.from_numpy(canvas.astype(np.float32)), (combined_mask > 0.3).astype(np.float32)


def make_training_batch(paths: list[Path], rng: np.random.Generator, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    images, masks = [], []
    for path in paths:
        image = load_image(path, config)
        if rng.random() < config.fake_defect_probability:
            image, mask = add_fake_defects(image, rng, config)
        else:
            mask = np.zeros(image.shape[-2:], dtype=np.float32)
        images.append(image)
        masks.append(torch.from_numpy(mask))
    return torch.stack(images).to(config.device), torch.stack(masks).unsqueeze(1).to(config.device)


# ---------- 학습할 층 ----------
def unfreeze_convolutions(modules: list[nn.Module]) -> list[nn.Parameter]:
    """Conv 가중치만 학습 대상으로 푼다. BatchNorm 은 건드리지 않는다(모델은 계속 eval 모드)."""
    parameters = []
    for module in modules:
        for sub_module in module.modules():
            if isinstance(sub_module, nn.Conv2d):
                for parameter in sub_module.parameters():
                    parameter.requires_grad_(True)
                    parameters.append(parameter)
    return parameters


def trainable_groups(extractor: nn.Module, config: Config) -> list[dict]:
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    layer3_blocks = list(extractor.layer3.children())
    groups = [{"params": unfreeze_convolutions(layer3_blocks[-3:] if config.fine_tune_stage == 1 else layer3_blocks),
               "lr": config.layer3_learning_rate}]
    if config.fine_tune_stage >= 3:
        groups.append({"params": unfreeze_convolutions(list(extractor.layer2.children())), "lr": config.layer2_learning_rate})
    return groups


# ---------- 학습 ----------
PARAM_NAMES = ("fine_tune_stage", "fine_tune_epochs", "fine_tune_batch_size", "layer3_learning_rate", "layer2_learning_rate",
               "head_learning_rate", "anchor_weight", "fake_defect_probability", "max_fake_defects")


def fine_tune_params(config: Config) -> dict:
    return {name: getattr(config, name) for name in PARAM_NAMES}


def validation_average_precision(detector: PatchCoreDetector, train_paths: list[Path], val_paths: list[Path],
                                 val_labels: np.ndarray, rng_state: torch.Tensor) -> float:
    torch.set_rng_state(rng_state)   # 에폭마다 같은 패치 샘플링 → 점수 차이는 백본 차이만
    detector.fit(train_paths)
    return float(average_precision_score(val_labels, detector.score(val_paths)))


def fine_tune(detector: PatchCoreDetector, train_paths: list[Path], val_paths: list[Path], val_labels: np.ndarray,
              config: Config) -> None:
    rng_state = torch.get_rng_state()   # 원래 train.py 가 fit 할 때의 난수 상태 → epoch 0 = 최종 모델과 같은 메모리 뱅크
    extractor = detector.extractor
    groups = trainable_groups(extractor, config)
    trainable = [parameter for group in groups for parameter in group["params"]]
    print(f"파인튜닝 stage {config.fine_tune_stage}: 학습 파라미터 {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    with torch.no_grad():
        channels = extractor(load_image(train_paths[0], config).unsqueeze(0).to(config.device)).shape[1]
    head = nn.Sequential(nn.Conv2d(channels, 256, 1), nn.ReLU(), nn.Conv2d(256, 1, 1)).to(config.device)
    optimizer = torch.optim.AdamW(groups + [{"params": head.parameters(), "lr": config.head_learning_rate}], weight_decay=0.0)
    frozen = copy.deepcopy(extractor).requires_grad_(False) if config.anchor_weight > 0 else None

    rng = np.random.default_rng(config.model_seed)
    best_ap = validation_average_precision(detector, train_paths, val_paths, val_labels, rng_state)
    best_epoch, best_weights = 0, copy.deepcopy(extractor.state_dict())
    print(f"epoch 0 (파인튜닝 전) val_AP={best_ap:.4f}")
    mlflow.log_metric("finetune_val_average_precision", best_ap, step=0)

    for epoch in range(1, config.fine_tune_epochs + 1):
        losses = []
        order = rng.permutation(len(train_paths))
        for start in range(0, len(order), config.fine_tune_batch_size):
            images, pixel_masks = make_training_batch(
                [train_paths[index] for index in order[start:start + config.fine_tune_batch_size]], rng, config)
            features = extractor(images)
            cell_masks = functional.adaptive_max_pool2d(pixel_masks, features.shape[-2:])   # 특징 격자 크기로 줄인 결함 위치
            positives = cell_masks.sum()
            positive_weight = ((cell_masks.numel() - positives) / positives).clamp(max=MAX_POSITIVE_WEIGHT) \
                if positives > 0 else torch.tensor(1.0, device=config.device)
            loss = functional.binary_cross_entropy_with_logits(head(features), cell_masks, pos_weight=positive_weight)

            if frozen is not None:
                with torch.no_grad():
                    original = frozen(images)
                keep = 1.0 - functional.max_pool2d(cell_masks, kernel_size=3, stride=1, padding=1)   # 결함 근처가 아닌 곳
                change = ((features - original) ** 2).mean(dim=1, keepdim=True) / original.pow(2).mean().clamp(min=1e-8)
                loss = loss + config.anchor_weight * (change * keep).sum() / keep.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 1.0)
            optimizer.step()
            losses.append(loss.item())

        average_precision = validation_average_precision(detector, train_paths, val_paths, val_labels, rng_state)
        print(f"epoch {epoch} loss={np.mean(losses):.4f} val_AP={average_precision:.4f}")
        mlflow.log_metrics({"finetune_loss": float(np.mean(losses)), "finetune_val_average_precision": average_precision},
                           step=epoch)
        if average_precision > best_ap:
            best_ap, best_epoch, best_weights = average_precision, epoch, copy.deepcopy(extractor.state_dict())

    extractor.load_state_dict(best_weights)
    extractor.requires_grad_(False)
    torch.set_rng_state(rng_state)   # 이어지는 train.py 의 fit 도 같은 샘플링
    mlflow.log_param("finetune_best_epoch", best_epoch)
    print(f"선택: epoch {best_epoch} (val_AP={best_ap:.4f})" + ("  -> 파인튜닝 전 가중치 그대로" if best_epoch == 0 else ""))
