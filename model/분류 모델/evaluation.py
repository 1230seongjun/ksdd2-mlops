import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from model import predicted_color

MISS = "안칠함(정상)"            # 결함 픽셀을 하나도 안 칠한 경우 = 정상이라고 예측
NORMAL_ROW = "정상"             # 혼동행렬의 정상 행 이름


def dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    return 2 * (prediction & truth).sum() / (prediction.sum() + truth.sum() + 1e-6)


def compute_metrics(true_colors, predicted_colors, dices) -> dict:
    truth, prediction = np.array(true_colors), np.array(predicted_colors)
    is_defect = truth > 0                                                                      # 결함 이미지만 (정상은 따로)
    return {
        "all_acc": float((truth == prediction).mean()),                                        # 핵심: 결함 색 + 정상 안 칠함
        "color_acc": float((truth == prediction)[is_defect].mean()),                           # 결함 사진만
        "normal_acc": float((prediction[~is_defect] == 0).mean()) if (~is_defect).any() else float("nan"),   # 정상 사진을 안 칠한 비율
        "dice": float(np.mean(np.array(dices)[is_defect])),                                    # 결함 위치 (색 무시)
    }


def confusion_matrix(true_colors, predicted_colors, config: Config) -> pd.DataFrame:
    """행 = 정답 색(+정상), 열 = 예측 색(+안칠함). 대각선이 맞힌 수"""
    rows = list(config.label_dir_names) + ([NORMAL_ROW] if config.normal_dir_name else [])
    matrix = pd.DataFrame(0, index=pd.Index(rows, name="정답＼예측"), columns=list(config.label_dir_names) + [MISS])
    for true, predicted in zip(true_colors, predicted_colors):
        matrix.loc[config.classes[true] if true else NORMAL_ROW, config.classes[predicted] if predicted else MISS] += 1
    return matrix


def verdict(metrics: dict, matrix: pd.DataFrame, config: Config) -> str:
    grade = lambda value, good, ok: "좋음" if value >= good else "보통" if value >= ok else "미흡"
    defect_rows = matrix.loc[list(config.label_dir_names)]
    total = defect_rows.values.sum()
    recalls = {label: matrix.loc[label, label] / matrix.loc[label].sum() for label in config.label_dir_names if matrix.loc[label].sum()}
    weak = [f"{label} {recall:.0%}" for label, recall in recalls.items() if recall < config.weak_recall]
    normal_lines = []
    if NORMAL_ROW in matrix.index and matrix.loc[NORMAL_ROW].sum():
        count = matrix.loc[NORMAL_ROW].sum()
        normal_lines = [
            f"전체 정확도(결함 색+정상) {metrics['all_acc']:.1%} → {grade(metrics['all_acc'], config.good_accuracy, config.ok_accuracy)}",
            f"정상 맞힘(아무것도 안 칠함) {metrics['normal_acc']:.1%} → {grade(metrics['normal_acc'], config.good_accuracy, config.ok_accuracy)}"
            f"  (정상 {count}장 중 결함으로 칠한 것 {count - matrix.loc[NORMAL_ROW, MISS]}장)",
        ]
    return "\n".join(normal_lines + [
        f"색 정확도(결함 사진만) {metrics['color_acc']:.1%} → {grade(metrics['color_acc'], config.good_accuracy, config.ok_accuracy)}"
        f"  (가장 많은 색만 찍으면 {defect_rows.sum(1).max() / total:.1%} / 기준 좋음≥{config.good_accuracy:.0%}, 보통≥{config.ok_accuracy:.0%})",
        f"위치 Dice {metrics['dice']:.2f} → {grade(metrics['dice'], config.good_dice, config.ok_dice)}"
        f"  (기준 좋음≥{config.good_dice}, 보통≥{config.ok_dice})",
        "색별 맞힌 비율: " + ", ".join(f"{label} {recall:.0%}" for label, recall in recalls.items()),
        f"약한 색(<{config.weak_recall:.0%}): " + (", ".join(weak) or "없음"),
        f"결함 자체를 못 찾은 이미지: {defect_rows[MISS].sum()}장 / {total}장",
        "※ 이 검증셋으로 best 에폭을 골라 점수가 약간 높게 나옴. 최종 판단은 5 fold 평균(summary.py) + 마지막 test",
    ])


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, config: Config):
    """반환: (지표 dict, 정답 색 목록, 예측 색 목록)"""
    model.eval()
    height, width = config.image_size
    predicted_colors, dices = [], []
    for images, masks in loader:
        maps = model(images.to(config.device)).argmax(1).cpu().numpy()[:, :height, :width]   # 픽셀마다 가장 높은 클래스, 패딩 잘라냄
        truths = masks.numpy()[:, :height, :width]
        for prediction, truth in zip(maps, truths):
            predicted_colors.append(predicted_color(prediction))
            dices.append(dice(prediction > 0, truth > 0))
    df = loader.dataset.df
    true_colors = [config.classes.index(label) if label in config.label_dir_names else 0 for label in df["label"]]   # 정상 = 0 (아무것도 안 칠해야 정답)
    # 두 색 섞인 이미지는 둘 중 하나를 맞히면 정답 처리
    true_colors = [predicted if config.classes[predicted] in str(also).split("+") else true
                   for true, predicted, also in zip(true_colors, predicted_colors, df["also"])]
    metrics = compute_metrics(true_colors, predicted_colors, dices)
    return metrics, true_colors, predicted_colors


def summarize_folds(config: Config) -> None:
    """MLflow 에 기록된 fold 별 결과 모아보기: fold별 지표 + 평균·흔들림(표준편차) + 혼동행렬 합 + 판정"""
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment)
    if experiment is None:
        raise SystemExit(f"MLflow 에 '{config.mlflow_experiment}' experiment 가 없습니다")
    runs = mlflow.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"])
    runs = runs[runs["tags.mlflow.runName"].str.startswith(f"{config.run_name}_fold")]
    runs = runs.drop_duplicates("params.fold")   # 같은 fold 를 여러 번 돌렸으면 가장 최근 것만
    if runs.empty:
        raise SystemExit(f"'{config.run_name}_fold' 로 시작하는 완료된 run 이 없습니다")
    runs = runs.assign(fold=runs["params.fold"].astype(int)).sort_values("fold").set_index("fold")
    df = pd.DataFrame({name: runs[f"metrics.best_{name}"] for name in ("all_acc", "color_acc", "normal_acc", "dice")})
    df.insert(0, "best_epoch", runs["metrics.best_epoch"].astype(int))
    df["split_hash"] = runs["params.split_hash"]
    print(df.round(3).to_string(), "\n")
    if df["split_hash"].nunique() > 1:
        print("※ fold 마다 split_hash 가 다릅니다 - 서로 다른 분할로 학습한 결과가 섞여 있습니다\n")
    scores = df.drop(columns=["best_epoch", "split_hash"])
    print(scores.agg(["mean", "std"]).round(3).to_string(), "\n")
    matrix = sum(pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"fold{fold}_cm.csv"), index_col=0)
                 for fold, run_id in runs["run_id"].items())
    print(f"[{len(df)}개 fold 합친 혼동행렬]\n{matrix.to_string()}\n")
    print(verdict(scores.mean().to_dict(), matrix, config))


