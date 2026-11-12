import argparse
import hashlib
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
import torchvision
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec

from config import Config, load_config
from data import DEFECT_LABEL, NORMAL_LABEL, split_dataset
from evaluation import evaluate, plot_pr_curve, save_errors, save_scores, select_threshold
from model import MODEL_INFO, PatchCoreDetector
from storage import download_dataset

import sys

CODE_DIR = Path(__file__).resolve().parent   # serving.py 등 모델과 함께 올릴 코드가 있는 폴더


# ---------- MLflow ----------
def setup_mlflow(config: Config) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)


def run_params(config: Config, dataset_split: dict) -> dict:
    """run 을 비교할 때 기준이 되는 설정과 분할 정보. split_hash 가 같아야 같은 이미지로 비교한 것이다."""
    names = sorted(f"{split}/{path.name}" for split in ("train", "val") for path in dataset_split[split][0])
    val_labels = dataset_split["val"][1]
    return {
        **MODEL_INFO,
        "coreset_pool_size": config.coreset_pool_size, "coreset_projection_dim": config.coreset_projection_dim,
        "max_memory_bank_size": config.max_memory_bank_size,
        "threshold_method": "val_recall", "target_recall": config.target_recall, "batch_size": config.batch_size,
        "split_seed": config.random_seed, "model_seed": config.model_seed, "device": config.device,
        "split_hash": hashlib.md5("\n".join(names).encode()).hexdigest()[:8],
        "n_train_normal": len(dataset_split["train"][0]),
        "n_val_normal": int((val_labels == NORMAL_LABEL).sum()),
        "n_val_defect": int((val_labels == DEFECT_LABEL).sum()),
    }


def log_split_result(split_name: str, paths: list[Path], labels: np.ndarray, scores: np.ndarray,
                     threshold: float, metrics: dict) -> None:
    """지표를 기록하고, PR 곡선·이미지별 점수·틀린 이미지 목록을 MLflow 의 <split_name>/ 폴더에 올린다."""
    mlflow.log_metrics(metrics)
    with tempfile.TemporaryDirectory() as folder:
        plot_pr_curve(labels, scores, metrics, split_name, Path(folder) / "pr_curve.png")
        save_scores(paths, labels, scores, Path(folder) / "scores.csv")
        save_errors(paths, labels, scores, threshold, Path(folder) / "errors.csv")
        mlflow.log_artifacts(folder, artifact_path=split_name)


def register_model(detector: PatchCoreDetector, config: Config) -> None:
    """모델(메모리 뱅크, 임계값, 백본 가중치, 코드)을 MLflow 모델 레지스트리에 새 버전으로 등록한다."""
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("uint8"), (-1, -1, -1, 3))]),   # RGB 이미지 (장수, 높이, 너비, 3)
        outputs=Schema([ColSpec("double", "anomaly_score"), ColSpec("boolean", "is_defect"), ColSpec("double", "threshold")]),
    )
    requirements = [f"torch=={torch.__version__.split('+')[0]}", f"torchvision=={torchvision.__version__.split('+')[0]}",
                    "pillow", "numpy", "pandas"]
    with tempfile.TemporaryDirectory() as folder:
        detector.save(Path(folder))
        mlflow.pyfunc.log_model(
            name="model", python_model=str(CODE_DIR / "mlflow_model.py"), artifacts={"detector": folder},
            code_paths=[str(CODE_DIR / name) for name in ("config.py", "data.py", "model.py")],
            signature=signature, pip_requirements=requirements,
            registered_model_name=config.registered_model_name,
        )


# ---------- 실행 ----------
def main(config: Config) -> None:
    torch.manual_seed(config.model_seed)

    if config.data_source == "object_storage":
        download_dataset(config)
    dataset_split = split_dataset(config)
    for split_name, (paths, labels) in dataset_split.items():
        print(f"{split_name}: 이미지 {len(paths)}장 (결함 {int(labels.sum())}장)")

    setup_mlflow(config)
    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(run_params(config, dataset_split))

        detector = PatchCoreDetector(config)
        train_paths, _ = dataset_split["train"]
        detector.fit(train_paths)
        print("memory bank 크기:", tuple(detector.memory_bank.shape))
        mlflow.log_param("memory_bank_size", len(detector.memory_bank))

        val_paths, val_labels = dataset_split["val"]
        val_scores = detector.score(val_paths)
        detector.threshold = float(select_threshold(val_labels, val_scores, config.target_recall))
        print(f"임계값 (검증 데이터, recall>={config.target_recall}): {detector.threshold:.4f}")

        val_metrics = evaluate(val_labels, val_scores, detector.threshold, "val")
        log_split_result("val", val_paths, val_labels, val_scores, detector.threshold, val_metrics)
        mlflow.log_metric("threshold", detector.threshold)

        if config.save_model:
            register_model(detector, config)

        if config.evaluate_test:   # 최종 모델 확정 후 한 번만
            test_paths, test_labels = dataset_split["test"]
            test_scores = detector.score(test_paths)
            test_metrics = evaluate(test_labels, test_scores, detector.threshold, "test")
            log_split_result("test", test_paths, test_labels, test_scores, detector.threshold, test_metrics)
    print(f"MLflow 기록 완료: {config.mlflow_experiment} / {config.run_name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # cp949 콘솔은 MLflow 가 찍는 이모지를 못 써서 학습 끝에 에러가 난다
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(load_config(parser.parse_args().config))