import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from data import DEFECT_LABEL, NORMAL_LABEL

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def select_threshold(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """재현율이 target_recall 이상인 임계값 중 가장 큰 값을 고른다 (오경보를 최소화)."""
    _, recalls, thresholds = precision_recall_curve(labels, scores)
    candidate_indices = np.where(recalls[:-1] >= target_recall)[0]  # recalls[:-1]이 thresholds와 대응
    return thresholds[candidate_indices.max()]


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float, split_name: str) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    false_positives = int(((predictions == 1) & (labels == NORMAL_LABEL)).sum())
    false_negatives = int(((predictions == 0) & (labels == DEFECT_LABEL)).sum())
    metrics = {
        f"{split_name}_average_precision": float(average_precision_score(labels, scores)),
        f"{split_name}_auroc": float(roc_auc_score(labels, scores)),
        f"{split_name}_recall": float(recall_score(labels, predictions)),
        f"{split_name}_precision": float(precision_score(labels, predictions, zero_division=0)),
        f"{split_name}_false_positives": false_positives,
        f"{split_name}_false_negatives": false_negatives,
        f"{split_name}_false_positive_rate": false_positives / max(int((labels == NORMAL_LABEL).sum()), 1),
    }
    print(f"[{split_name}] AP={metrics[f'{split_name}_average_precision']:.3f} "
          f"AUROC={metrics[f'{split_name}_auroc']:.3f}")
    print(f"[{split_name}] 선택한 임계값 기준: recall={metrics[f'{split_name}_recall']:.3f} "
          f"precision={metrics[f'{split_name}_precision']:.3f} (오경보 {false_positives}장, 놓친 결함 {false_negatives}장)")
    return metrics


def plot_pr_curve(labels: np.ndarray, scores: np.ndarray, metrics: dict, split_name: str, plot_path: Path) -> None:
    curve_precisions, curve_recalls, _ = precision_recall_curve(labels, scores)
    plt.figure()
    plt.plot(curve_recalls, curve_precisions)
    plt.scatter([metrics[f"{split_name}_recall"]], [metrics[f"{split_name}_precision"]], c="red",
                label="선택한 임계값 기준 지점")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_name} PR curve  AP={metrics[f'{split_name}_average_precision']:.3f}")
    plt.legend()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def save_scores(paths: list[Path], labels: np.ndarray, scores: np.ndarray, csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "label", "score"])
        for path, label, score in zip(paths, labels, scores):
            writer.writerow([path.name, int(label), f"{score:.6f}"])


def save_errors(paths: list[Path], labels: np.ndarray, scores: np.ndarray, threshold: float, csv_path: Path) -> None:
    """임계값 기준으로 틀린 이미지 목록. 놓친 결함은 점수가 낮은 순, 오경보는 점수가 높은 순(= 가장 심하게 틀린 순)."""
    errors = []
    for path, label, score in zip(paths, labels, scores):
        if label == DEFECT_LABEL and score < threshold:
            errors.append(("missed_defect", path.name, int(label), score))
        elif label == NORMAL_LABEL and score >= threshold:
            errors.append(("false_alarm", path.name, int(label), score))
    errors.sort(key=lambda row: (row[0], row[3] if row[0] == "missed_defect" else -row[3]))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["error_type", "file", "label", "score", "threshold", "score_over_threshold"])
        for error_type, name, label, score in errors:
            writer.writerow([error_type, name, label, f"{score:.6f}", f"{threshold:.6f}", f"{score / threshold:.3f}"])