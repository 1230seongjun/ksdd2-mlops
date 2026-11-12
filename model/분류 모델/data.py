import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import Config, InferenceSettings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet 기준 정규화
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"폴더가 없습니다: {folder}")
    return sorted(path for path in folder.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


# ---------- 이미지 / 마스크 읽기 ----------
def preprocess_image(image: Image.Image) -> Image.Image:
    """학습과 추론이 똑같이 거치는 전처리. 지금은 아무것도 하지 않는다 (필터 등을 넣을 자리)."""
    return image


def image_to_array(image: Image.Image, config: Config | InferenceSettings) -> np.ndarray:
    """RGB → ImageNet 정규화한 (높이, 너비, 3) 배열. 학습 때와 크기가 다르면 멈춘다."""
    array = np.asarray(preprocess_image(image.convert("RGB")), np.float32)
    if array.shape[:2] != config.image_size:
        raise ValueError(f"이미지 크기 {array.shape[:2]}(세로, 가로)가 학습 크기 {config.image_size}와 다릅니다")
    return (array / 255 - MEAN) / STD


def pad(array: np.ndarray, config: Config | InferenceSettings) -> np.ndarray:
    """오른쪽·아래에 0 을 채워 32의 배수 크기로 만든다."""
    height, width = array.shape[:2]
    padded_height, padded_width = config.padded_size
    extra = [(0, padded_height - height), (0, padded_width - width)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, extra)


def read_binary_mask(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127   # RGBA 마스크도 흑백 1채널로


def load_mask(row: pd.Series, config: Config) -> np.ndarray:
    """픽셀값 = 클래스 번호 (0 배경, 1~ 색). 두 색 섞인 이미지는 색별 마스크를 합침. 정상은 전부 0"""
    if row["label"] == config.normal_dir_name:
        return np.zeros(config.image_size, np.int64)
    labels = [row["label"]] + [label for label in str(row["also"]).split("+") if label in config.label_dir_names]
    if len(labels) == 1:
        return read_binary_mask(row["mask"]).astype(np.int64) * config.classes.index(row["label"])
    mask = np.zeros(config.image_size, np.int64)
    for label in labels:
        part = config.mask_dir / f"{row['id']}_GT({config.part_mask_names[label]}).png"
        mask[read_binary_mask(part)] = config.classes.index(label)
    return mask


# ---------- 분할 ----------
def load_defect_items(config: Config) -> pd.DataFrame:
    """결함 이미지 목록. 두 폴더에 같이 있는 이미지는 1장으로 세고, 처음 나온 폴더를 label, 나머지를 also 에 적는다."""
    from sklearn.model_selection import StratifiedKFold, train_test_split   # 추론 서버에는 필요 없어서 여기서 불러온다
    rows = [{"id": path.stem, "label": label, "image": str(path), "mask": str(config.mask_dir / f"{path.stem}_GT.png")}
            for label in config.label_dir_names for path in list_image_paths(config.dataset_dir / label)]
    if not rows:
        raise FileNotFoundError(f"결함 이미지를 찾지 못했습니다: {config.dataset_dir}")
    df = pd.DataFrame(rows)
    missing = [mask for mask in df["mask"] if not Path(mask).exists()]
    if missing:
        raise FileNotFoundError(f"마스크 {len(missing)}장이 없습니다. 예: {missing[0]}")
    also = df.groupby("id")["label"].apply(lambda labels: "+".join(labels.iloc[1:]))
    df = df.drop_duplicates("id").reset_index(drop=True)
    df["also"] = df["id"].map(also)
    return df


def assign_defect_splits(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """결함: 라벨 비율을 유지(층화)하며 test 를 떼고, 나머지를 fold 로 나눈다."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    rest, test = train_test_split(df.index, test_size=config.test_ratio, stratify=df["label"], random_state=config.random_seed)
    df["split"] = ""
    df.loc[test, "split"] = "test"
    folds = StratifiedKFold(config.n_folds, shuffle=True, random_state=config.random_seed)
    for fold, (_, validation) in enumerate(folds.split(rest, df.loc[rest, "label"])):
        df.loc[rest[validation], "split"] = f"fold{fold}"
    return df


def load_normal_split(config: Config) -> pd.DataFrame:
    """정상 이미지: 마스크 없음. 결함과 따로 같은 비율로 나눠서 결함 분할은 정상 유무와 상관없이 같게 유지된다."""
    from sklearn.model_selection import KFold, train_test_split
    paths = list_image_paths(config.dataset_dir / config.normal_dir_name)
    df = pd.DataFrame({"id": [path.stem for path in paths], "label": config.normal_dir_name,
                       "image": [str(path) for path in paths], "mask": "", "also": "", "split": ""})
    rest, test = train_test_split(df.index, test_size=config.test_ratio, random_state=config.random_seed)
    df.loc[test, "split"] = "test"
    for fold, (_, validation) in enumerate(KFold(config.n_folds, shuffle=True, random_state=config.random_seed).split(rest)):
        df.loc[rest[validation], "split"] = f"fold{fold}"
    return df


def split_dataset(config: Config) -> pd.DataFrame:
    """이미지 한 장당 한 줄: id, label, also, image, mask, split(test 또는 fold0~). 파일 목록과 시드가 같으면 항상 같은 분할이다."""
    df = assign_defect_splits(load_defect_items(config), config)
    if config.normal_dir_name:
        df = pd.concat([df, load_normal_split(config)], ignore_index=True)
    return df


def split_hash(df: pd.DataFrame) -> str:
    """분할 지문. 값이 같으면 같은 이미지가 같은 fold 에 들어간 것이다."""
    names = sorted(f"{row.split}/{row.label}/{row.id}" for row in df.itertuples())
    return hashlib.md5("\n".join(names).encode()).hexdigest()[:8]


# ---------- 학습용 데이터셋 ----------
class SegDataset(Dataset):
    """한 줄 = 이미지 1장 + 색 번호 마스크 1장. 학습 때만 뒤집기 (색은 안 건드림)"""

    def __init__(self, df: pd.DataFrame, config: Config, train: bool):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = image_to_array(Image.open(row["image"]), self.config)
        mask = load_mask(row, self.config)
        if self.train:
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]   # 좌우 뒤집기
            if np.random.rand() < 0.5:
                image, mask = image[::-1], mask[::-1]         # 상하 뒤집기
        image, mask = pad(image, self.config), pad(mask, self.config)
        return torch.from_numpy(image.transpose(2, 0, 1).copy()), torch.from_numpy(mask.copy())   # 마스크: (H, W) 클래스 번호