import argparse
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torchvision
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec
from torch.utils.data import DataLoader

from config import Config, load_config
from data import SegDataset, split_dataset, split_hash
from evaluation import confusion_matrix, evaluate, verdict
from model import build_model, save_classifier
from storage import download_dataset

CODE_DIR = Path(__file__).resolve().parent   # mlflow_model.py 등 모델과 함께 올릴 코드가 있는 폴더


def run_params(config: Config, fold: int, df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """run 을 비교할 때 기준이 되는 설정과 분할 정보. split_hash 가 같아야 같은 이미지로 비교한 것이다."""
    return {
        "fold": fold, "split_hash": split_hash(df), "split_seed": config.random_seed, "model_seed": config.model_seed,
        "test_ratio": config.test_ratio, "n_folds": config.n_folds, "n_train": len(train_df), "n_val": len(val_df),
        "encoder": config.encoder, "encoder_weights": config.encoder_weights, "epochs": config.epochs,
        "batch_size": config.batch_size, "lr": config.lr, "lr_schedule": config.lr_schedule, "lr_min": config.lr_min,
        "dice_weight": config.dice_weight, "patience": config.patience, "device": config.device,
    }


def register_model(model: torch.nn.Module, config: Config) -> None:
    """가중치, 설정, 코드를 MLflow 모델 레지스트리에 새 버전으로 등록한다."""
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("uint8"), (-1, -1, -1, 3))]),   # RGB 이미지 (장수, 높이, 너비, 3)
        outputs=Schema([ColSpec("long", "color_id"), ColSpec("string", "color"), ColSpec("long", "painted_pixels")]),
    )
    requirements = [f"torch=={torch.__version__.split('+')[0]}", f"torchvision=={torchvision.__version__.split('+')[0]}",
                    f"segmentation-models-pytorch=={smp.__version__}", "pillow", "numpy", "pandas", "pyyaml"]
    with tempfile.TemporaryDirectory() as folder:
        save_classifier(model, config, Path(folder))
        mlflow.pyfunc.log_model(
            name="model", python_model=str(CODE_DIR / "mlflow_model.py"), artifacts={"classifier": folder},
            code_paths=[str(CODE_DIR / name) for name in ("config.py", "data.py", "model.py")],
            signature=signature, pip_requirements=requirements,
            registered_model_name=config.registered_model_name,
        )


def main(config: Config, fold: int) -> None:
    """fold 번째 fold 로 검증하고 나머지 fold 로 학습한다. test 는 여기서 쓰지 않는다."""
    torch.manual_seed(config.model_seed)
    np.random.seed(config.model_seed)

    if config.data_source == "object_storage":
        download_dataset(config)
    df = split_dataset(config)
    print(pd.crosstab(df["label"], df["split"], margins=True))
    print("두 폴더에 같이 있는 이미지:", df.loc[df["also"] != "", ["id", "label", "also", "split"]].to_dict("records"))
    print("split_hash:", split_hash(df))

    folds = [f"fold{k}" for k in range(config.n_folds)]
    train_df = df[df["split"].isin([name for name in folds if name != f"fold{fold}"])]
    val_df = df[df["split"] == f"fold{fold}"]
    train_loader = DataLoader(SegDataset(train_df, config, train=True), config.batch_size, shuffle=True)
    val_loader = DataLoader(SegDataset(val_df, config, train=False), config.batch_size)

    model = build_model(config, pretrained=True).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr_min)
                 if config.lr_schedule == "cosine" else None)
    dice_loss, cross_entropy = smp.losses.DiceLoss("multiclass"), torch.nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name=f"{config.run_name}_fold{fold}"):
        mlflow.log_params(run_params(config, fold, df, train_df, val_df))

        best = None   # (점수, epoch, 지표, 정답색, 예측색, 가중치)
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses = []
            for images, masks in train_loader:
                images, masks = images.to(config.device), masks.to(config.device)
                logits = model(images)
                loss = config.dice_weight * dice_loss(logits, masks) + (1 - config.dice_weight) * cross_entropy(logits, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            learning_rate = optimizer.param_groups[0]["lr"]
            if scheduler:
                scheduler.step()

            metrics, true_colors, predicted_colors = evaluate(model, val_loader, config)
            print(f"ep{epoch:3d} lr={learning_rate:.1e} loss={np.mean(losses):.3f} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
            mlflow.log_metrics({"train_loss": float(np.mean(losses)), "lr": learning_rate,
                                **{f"val_{k}": v for k, v in metrics.items()}}, step=epoch)
            score = (metrics["all_acc"], metrics["dice"])     # 전체 정확도(결함 색 + 정상 안 칠함) 우선, 같으면 Dice
            if best is None or score > best[0]:
                best = (score, epoch, metrics, true_colors, predicted_colors,
                        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
            elif config.patience and epoch - best[1] >= config.patience:
                print(f"조기 종료: {config.patience}에폭 동안 개선 없음")
                break

        _, best_epoch, metrics, true_colors, predicted_colors, best_weights = best
        matrix = confusion_matrix(true_colors, predicted_colors, config)
        text = verdict(metrics, matrix, config)
        print(f"\n[fold{fold}] best ep{best_epoch} (마지막 ep{epoch})\n{matrix.to_string()}\n\n{text}")

        mlflow.log_metrics({"best_epoch": best_epoch, "last_epoch": epoch, **{f"best_{k}": v for k, v in metrics.items()}})
        mlflow.log_text(text, "verdict.txt")
        mlflow.log_text(matrix.to_csv(), f"fold{fold}_cm.csv")
        mlflow.log_text(df.to_csv(index=False), "split.csv")

        if config.save_model:
            model.load_state_dict(best_weights)
            register_model(model, config)
    print(f"MLflow 기록 완료: {config.mlflow_experiment} / {config.run_name}_fold{fold}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # cp949 콘솔은 MLflow 가 찍는 이모지를 못 써서 학습 끝에 에러가 난다
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    arguments = parser.parse_args()
    main(load_config(arguments.config), arguments.fold)