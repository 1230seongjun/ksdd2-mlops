import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class Config:
    """config.yaml 의 값. 키 이름이 틀리거나 빠지면 불러올 때 바로 오류가 난다."""
    data_source: str
    data_dir: Path
    normal_dir_name: str
    anomaly_dir_name: str
    object_storage_endpoint: str
    object_storage_region: str
    object_storage_bucket: str
    object_storage_prefix: str
    cache_dir: Path
    normal_train_ratio: float
    normal_validation_ratio: float
    anomaly_validation_ratio: float
    random_seed: int
    max_memory_bank_size: int
    coreset_pool_size: int
    coreset_projection_dim: int
    model_seed: int
    batch_size: int
    query_chunk_size: int
    device: str
    target_recall: float
    evaluate_test: bool
    mlflow_tracking_uri: str
    mlflow_experiment: str
    run_name: str
    save_model: bool
    registered_model_name: str

    @property
    def dataset_dir(self) -> Path:
        """학습에 실제로 읽는 이미지 폴더. 오브젝트 스토리지를 쓰면 내려받은 캐시 폴더."""
        return self.cache_dir if self.data_source == "object_storage" else self.data_dir

    @property
    def normal_dir(self) -> Path:
        return self.dataset_dir / self.normal_dir_name

    @property
    def anomaly_dir(self) -> Path:
        return self.dataset_dir / self.anomaly_dir_name

@dataclass(frozen=True)
class InferenceSettings:
    """점수만 계산할 때(추론 서버) 필요한 설정. 학습용 설정(Config)과 같은 이름의 값만 가진다."""
    batch_size: int
    query_chunk_size: int
    device: str


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
    values["data_dir"] = Path(values["data_dir"])
    values["cache_dir"] = Path(values["cache_dir"])
    if values["device"] == "auto":
        values["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    values["mlflow_tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI", values["mlflow_tracking_uri"])
    return Config(**values)