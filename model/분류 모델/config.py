import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

import os

PAD_MULTIPLE = 32   # U-Net 은 입력 크기가 32의 배수여야 한다
def padded(height: int, width: int) -> tuple[int, int]:
    return math.ceil(height / PAD_MULTIPLE) * PAD_MULTIPLE, math.ceil(width / PAD_MULTIPLE) * PAD_MULTIPLE


@dataclass(frozen=True)
class Config:
    """config.yaml 의 값. 키 이름이 틀리거나 빠지면 불러올 때 바로 오류가 난다."""
    data_source: str
    data_dir: Path
    label_dir_names: tuple[str, ...]
    normal_dir_name: str | None
    mask_dir_name: str
    part_mask_names: dict[str, str]
    object_storage_endpoint: str
    object_storage_region: str
    object_storage_bucket: str
    object_storage_prefix: str
    cache_dir: Path
    test_ratio: float
    n_folds: int
    random_seed: int
    image_height: int
    image_width: int
    encoder: str
    encoder_weights: str | None
    model_seed: int
    epochs: int
    batch_size: int
    lr: float
    lr_schedule: str | None
    lr_min: float
    dice_weight: float
    patience: int | None
    device: str
    mlflow_tracking_uri: str
    mlflow_experiment: str
    save_model: bool
    registered_model_name: str
    good_accuracy: float
    ok_accuracy: float
    good_dice: float
    ok_dice: float
    weak_recall: float
    run_name: str

    @property
    def dataset_dir(self) -> Path:
        """학습에 실제로 읽는 이미지 폴더. 오브젝트 스토리지를 쓰면 내려받은 캐시 폴더."""
        return self.cache_dir if self.data_source == "object_storage" else self.data_dir

    @property
    def mask_dir(self) -> Path:
        return self.dataset_dir / self.mask_dir_name

    @property
    def classes(self) -> list[str]:
        """클래스 번호 순서: 0 = 배경, 1~ = 라벨 폴더 순서"""
        return ["배경", *self.label_dir_names]

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def padded_size(self) -> tuple[int, int]:
        return padded(self.image_height, self.image_width)
        
@dataclass(frozen=True)
class InferenceSettings:
    """색 분류만 할 때(추론 서버) 필요한 설정. 학습용 설정(Config)과 같은 이름의 값만 가진다."""
    encoder: str
    classes: list[str]
    image_height: int
    image_width: int
    batch_size: int
    device: str

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def padded_size(self) -> tuple[int, int]:
        return padded(self.image_height, self.image_width)


def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return {key.lower(): value for key, value in yaml.safe_load(file).items()}   # yaml 은 대문자, 코드는 소문자


def load_config(path: str | Path = "config.yaml") -> Config:
    from dotenv import load_dotenv   # 추론 서버가 config.py 를 불러올 때는 필요 없어서 여기서 불러온다
    load_dotenv()   # .env 가 있으면 환경변수로 읽는다 (없어도 오류 없음)
    path = Path(path).resolve()
    values = {}
    common_path = path.parent.parent / "common.yaml"   # 두 모델 폴더의 상위 폴더에 있는, 두 모델이 같이 쓰는 설정
    if common_path.exists():
        values.update({key: value for key, value in read_yaml(common_path).items() if key in Config.__dataclass_fields__})
    values.update(read_yaml(path))   # 모델 폴더의 config.yaml 에 같은 키가 있으면 그 값이 우선
    missing = [key.upper() for key in Config.__dataclass_fields__ if key not in values]
    if missing:
        raise KeyError(f"설정에 없는 키: {', '.join(missing)} (모델 폴더의 config.yaml 이나 {common_path} 에 있어야 합니다)")
    if values["data_source"] not in ("local", "object_storage"):
        raise ValueError(f"data_source 는 local 또는 object_storage 여야 합니다: {values['data_source']}")
    if values["data_source"] == "object_storage" and not values["object_storage_endpoint"]:
        raise ValueError("data_source 가 object_storage 이면 object_storage_endpoint 를 채워야 합니다")
    for key in ("data_dir", "cache_dir"):
        values[key] = (path.parent / values[key]).resolve()
    values["label_dir_names"] = tuple(values["label_dir_names"])
    values["mlflow_tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI", values["mlflow_tracking_uri"])
    if values["device"] == "auto":
        values["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return Config(**values)