"""불량제로 실험 비교 대시보드 (Streamlit)

MLflow 에 기록된 모든 익스페리먼트의 run 을 읽어서 핵심 지표(AP·재현율·정밀도)를 비교한다.
  - 시드는 42 로 고정: seed 가 42 가 아닌 run(43, 44 등)은 모두 뺀다. seed 기록이 없는 초반 run 은 42 로 본다.
  - 익스페리먼트마다 지표 이름이 달라도(test_average_precision / ap / auprc ...) 같은 칸으로 모아서 보여 준다.
  - 재현율 98% 지점: 이미지별 점수 파일(*_scores.csv)로 test 에서 재현율이 98% 이상이 되는 가장 높은 임계값을 찾고,
    그 지점의 재현율·정밀도·오경보·놓친 결함을 계산한다 (baseline.py 의 'recall>=98% 지점의 precision' 과 같은 값).
    학습 때 val 로 정한 임계값의 결과는 '기록' 칸에 따로 보여 준다.

실행 (노트북, SSH 터널로 MLflow 가 127.0.0.1:5000 에 열려 있을 때):
  $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"   # PowerShell
  streamlit run app.py
컨테이너에서는 MLFLOW_TRACKING_URI=http://mlflow:5000 으로 실행
"""
import math
import os
import re
import tempfile

import altair as alt
import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
FIXED_SEED = 42
FIXED_RECALL = 0.98
mlflow.set_tracking_uri(TRACKING_URI)
st.set_page_config(page_title="불량제로 실험 비교", page_icon="🔍", layout="wide")

# 화면 색: 지표마다 같은 색을 모든 그래프에서 쓴다
INK, MUTED, LINE, PANEL = "#1d2731", "#66727e", "#dfe4e8", "#f6f8f9"
COLOR = {"AP": "#2b5d78", "재현율": "#3c8a68", "정밀도": "#c27a2c"}
NORMAL_COLOR, DEFECT_COLOR = "#5b8db8", "#d0603a"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .page-title {{ font-size: 1.75rem; font-weight: 700; color: {INK}; margin: 0; }}
  .page-sub {{ color: {MUTED}; font-size: .92rem; margin: .2rem 0 1.2rem; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1.4rem; }}
  .chip {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 999px; padding: .15rem .7rem;
           font-size: .82rem; color: {MUTED}; }}
  .chip b {{ color: {INK}; font-weight: 600; }}
  .section {{ font-size: 1.05rem; font-weight: 650; color: {INK}; margin: 1.6rem 0 .2rem; }}
  .hint {{ color: {MUTED}; font-size: .85rem; margin: 0 0 .7rem; }}
  .kpi {{ border: 1px solid {LINE}; border-radius: 10px; padding: .9rem 1rem; background: white; height: 100%; }}
  .kpi .label {{ font-size: .8rem; color: {MUTED}; letter-spacing: .02em; }}
  .kpi .value {{ font-size: 1.9rem; font-weight: 700; line-height: 1.2; font-variant-numeric: tabular-nums; }}
  .kpi .who {{ font-size: .8rem; color: {MUTED}; margin-top: .25rem; word-break: break-all; }}
  .kpi .bar {{ height: 4px; border-radius: 2px; margin-bottom: .6rem; }}
  div[data-testid="stTabs"] button p {{ font-size: .95rem; }}
  section[data-testid="stSidebar"] .stMarkdown p {{ font-size: .85rem; }}
  section[data-testid="stSidebar"] {{ min-width: 300px; }}
  section[data-testid="stSidebar"] span[data-baseweb="tag"] {{ max-width: 100%; }}
</style>
""", unsafe_allow_html=True)

# 화면에 쓰는 지표: (보여 줄 이름, 정확히 같은 이름 후보, 이름 패턴, 제외 패턴)
# 익스페리먼트마다 지표 이름이 달라서, 위에서부터 먼저 찾은 값을 쓴다. 패턴으로 찾을 때는 test_ 가 붙은 것을 먼저 쓴다.
METRIC_RULES = {
    "ap": ("AP",
           ["test_average_precision", "average_precision", "test_ap", "ap", "test_auprc", "auprc", "pr_auc"],
           r"average_precision|auprc|pr_auc|^(test_|val_)?ap$", r"train"),
    "recall": ("재현율",
               ["test_recall", "recall", "image_recall"],
               r"recall", r"small|medium|large|train|target|at_|@"),
    "precision": ("정밀도",
                  ["test_precision", "precision", "image_precision"],
                  r"precision", r"average|at_|target|train|@"),
    "precision_at_target": ("목표 재현율 지점 정밀도",
                            ["test_precision_at_target_recall"],
                            r"precision_at|precision@", r"train"),
    "fp": ("오경보",
           ["test_false_positives", "false_positives", "fp"],
           r"false_pos|^(test_)?fp$", r"train"),
    "fn": ("놓친 결함",
           ["test_false_negatives", "false_negatives", "fn"],
           r"false_neg|^(test_)?fn$", r"train"),
    "recall_small": ("재현율(작은 결함)", ["test_recall_small"], None, None),
    "recall_medium": ("재현율(중간 결함)", ["test_recall_medium"], None, None),
    "recall_large": ("재현율(큰 결함)", ["test_recall_large"], None, None),
}
LABEL = {key: rule[0] for key, rule in METRIC_RULES.items()}
LABEL.update({"r98_recall": "재현율", "r98_precision": "정밀도", "r98_fp": "오경보", "r98_fn": "놓친 결함"})
KEY = ["ap", "recall", "precision"]
COUNT = ["fp", "fn"]
R98 = int(round(FIXED_RECALL * 100))


# ---------- 데이터 읽기·정리 ----------
@st.cache_data(ttl=60, show_spinner="MLflow 에서 run 을 읽는 중 ...")
def load_raw_runs() -> pd.DataFrame:
    client = MlflowClient()
    experiments = {e.experiment_id: e.name for e in client.search_experiments(max_results=1000)}
    if not experiments:
        return pd.DataFrame()
    runs = mlflow.search_runs(experiment_ids=list(experiments), max_results=50000, output_format="pandas")
    if runs.empty:
        return runs
    runs["experiment"] = runs["experiment_id"].map(experiments)
    return runs


def column(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df), index=df.index)


def pick_metric(raw: pd.DataFrame, exact, pattern, exclude):
    """후보 지표 칸 중 값이 있는 첫 번째를 run 마다 고른다. (값, 실제로 쓴 지표 이름) 을 돌려준다."""
    names = [n for n in exact if f"metrics.{n}" in raw.columns]
    if pattern:
        found = [c[len("metrics."):] for c in raw.columns if c.startswith("metrics.")]
        found = [n for n in found if n not in names and re.search(pattern, n, re.I)
                 and not (exclude and re.search(exclude, n, re.I))]
        found.sort(key=lambda n: (0 if "test" in n.lower() else 2 if "val" in n.lower() else 1, n))
        names += found
    value = pd.Series(np.nan, index=raw.index, dtype=float)
    source = pd.Series("", index=raw.index, dtype=object)
    for name in names:
        candidate = pd.to_numeric(raw[f"metrics.{name}"], errors="coerce")
        fill = value.isna() & candidate.notna()
        value[fill] = candidate[fill]
        source[fill] = name
    return value, source


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """MLflow run 표를 비교하기 쉬운 모양으로 바꾸고, 시드 42 가 아닌 run 은 뺀다."""
    df = pd.DataFrame(index=raw.index)
    df["run_id"] = raw["run_id"]
    df["experiment"] = raw["experiment"]
    df["run_name"] = column(raw, "tags.mlflow.runName").fillna(raw["run_id"].str[:8])
    df["status"] = column(raw, "status")
    df["start_time"] = pd.to_datetime(column(raw, "start_time"), utc=True).dt.tz_convert("Asia/Seoul")
    df["kind"] = column(raw, "params.kind").fillna("")
    df["backbone"] = column(raw, "params.backbone")
    df["split_hash"] = column(raw, "params.split_hash")
    df["target_recall"] = pd.to_numeric(column(raw, "params.target_recall"), errors="coerce")

    # 시드: params.seed → 없으면 run 이름의 seed숫자 → 그래도 없으면 42 로 본다
    seed = pd.to_numeric(column(raw, "params.seed"), errors="coerce")
    from_name = pd.to_numeric(df["run_name"].str.extract(r"seed[_-]?(\d+)", flags=re.I)[0], errors="coerce")
    df["seed"] = seed.fillna(from_name).fillna(FIXED_SEED).astype(int)

    for key, (_, exact, pattern, exclude) in METRIC_RULES.items():
        value, source = pick_metric(raw, exact, pattern, exclude)
        df[key], df[f"{key}_source"] = value.astype(float), source
    threshold = pd.to_numeric(column(raw, "metrics.threshold"), errors="coerce")
    df["threshold"] = threshold.fillna(pd.to_numeric(column(raw, "params.threshold"), errors="coerce"))

    keep = (df["seed"] == FIXED_SEED) & (df["kind"] != "seed_summary") & (df["status"] != "RUNNING")
    return df[keep].copy()


@st.cache_data(ttl=600, show_spinner=False)
def load_scores(run_id: str) -> pd.DataFrame:
    """run 의 *_scores.csv artifact (split, file, label, score, mask_area) 를 읽는다. 없으면 빈 표."""
    client = MlflowClient()
    paths = [a.path for a in client.list_artifacts(run_id) if a.path.endswith("_scores.csv")]
    if not paths:
        return pd.DataFrame()
    with tempfile.TemporaryDirectory() as folder:
        local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=paths[0], dst_path=folder)
        scores = pd.read_csv(local, encoding="utf-8-sig")
    scores["label"] = scores["label"].astype(int)
    return scores


def threshold_for_recall(scores: pd.DataFrame, target: float) -> float:
    """재현율이 target 이상인 임계값들 중 정밀도가 가장 높은 임계값 (같으면 더 높은 임계값).
    baseline.py 의 precision_at_target_recall 과 같은 지점을 고른다."""
    labels = scores["label"].to_numpy()
    values = scores["score"].to_numpy(dtype=float)
    n_defect = int((labels == 1).sum())
    if n_defect == 0:
        return np.nan
    order = np.argsort(-values, kind="mergesort")
    values, labels = values[order], labels[order]
    tp, fp = np.cumsum(labels == 1), np.cumsum(labels == 0)
    last = np.r_[values[1:] != values[:-1], True]          # 같은 점수 묶음의 마지막 위치 = 그 점수를 임계값으로 쓴 결과
    recall, precision = tp[last] / n_defect, tp[last] / (tp[last] + fp[last])
    candidates = np.where(recall >= target - 1e-12)[0]
    best = candidates[np.argmax(precision[candidates])]     # argmax 는 첫 번째(=가장 높은 임계값)를 고름
    return float(values[last][best])


def evaluate(scores: pd.DataFrame, threshold: float) -> dict:
    predicted = scores["score"] >= threshold
    truth = scores["label"] == 1
    tp, fp, fn = int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum())
    return {"r98_recall": tp / (tp + fn) if tp + fn else np.nan,
            "r98_precision": tp / (tp + fp) if tp + fp else np.nan,
            "r98_fp": fp, "r98_fn": fn, "r98_threshold": threshold}


def recall98(run: pd.Series) -> dict:
    """test 에서 재현율이 98% 이상이 되는 지점의 재현율·정밀도. 점수 파일 → 기록된 지점 정밀도 순서로 찾는다."""
    try:
        scores = load_scores(run["run_id"])
    except Exception:
        scores = pd.DataFrame()
    if not scores.empty and {"split", "score", "label"} <= set(scores.columns):
        test = scores[scores["split"] == "test"]
        if (test["label"] == 1).any():
            result = evaluate(test, threshold_for_recall(test, FIXED_RECALL))
            result["r98_source"] = "점수 파일로 계산"
            return result
    if abs((run["target_recall"] or 0) - FIXED_RECALL) < 1e-6 and pd.notna(run["precision_at_target"]):
        return {"r98_recall": np.nan, "r98_precision": run["precision_at_target"], "r98_fp": np.nan,
                "r98_fn": np.nan, "r98_threshold": np.nan, "r98_source": "기록된 지점 정밀도 (정밀도만)"}
    return {"r98_recall": np.nan, "r98_precision": np.nan, "r98_fp": np.nan, "r98_fn": np.nan,
            "r98_threshold": np.nan, "r98_source": "계산 불가 (점수 파일 없음)"}


def best_per_experiment(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """익스페리먼트마다 대표 run 하나 (AP 최고 또는 가장 최근)."""
    if rule == "AP 최고":
        ordered = df.sort_values(["ap", "recall", "start_time"], ascending=[False, False, False], na_position="last")
    else:
        ordered = df.sort_values("start_time", ascending=False)
    return ordered.drop_duplicates("experiment", keep="first").sort_values("ap", ascending=False, na_position="last")


def judge(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """임계값으로 판정하고 결과 유형(정상 통과/오경보/결함 검출/놓침)을 붙인다."""
    judged = scores.copy()
    judged["판정"] = np.where(judged["score"] >= threshold, "불량", "정상")
    judged["임계값 대비"] = judged["score"] / threshold
    judged["결과"] = np.select(
        [(judged["label"] == 1) & (judged["판정"] == "불량"), (judged["label"] == 1) & (judged["판정"] == "정상"),
         (judged["label"] == 0) & (judged["판정"] == "불량")],
        ["결함 검출", "놓침", "오경보"], default="정상 통과")
    return judged


def fmt(value, digits=3):
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def section(title, hint=None):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<p class="hint">{hint}</p>', unsafe_allow_html=True)


def kpi(box, label, value, who, color):
    box.markdown(f'<div class="kpi"><div class="bar" style="background:{color}"></div>'
                 f'<div class="label">{label}</div><div class="value" style="color:{color}">{value}</div>'
                 f'<div class="who">{who}</div></div>', unsafe_allow_html=True)


def ratio(label, help_text=None, color=None):
    """0~1 값을 막대로 보여 주는 칸. color 는 지표 색 (AP 파랑 · 재현율 초록 · 정밀도 주황)."""
    return st.column_config.ProgressColumn(label, format="%.3f", min_value=0.0, max_value=1.0, help=help_text,
                                           color=color or COLOR["AP"])


def number(label, fmt_text="%.3f", help_text=None):
    return st.column_config.NumberColumn(label, format=fmt_text, help=help_text)


BASE_CONFIG = {
    "experiment": st.column_config.TextColumn("익스페리먼트", width="medium"),
    "run_name": st.column_config.TextColumn("run", width="medium"),
    "start_time": st.column_config.DatetimeColumn("시작", format="MM-DD HH:mm"),
    "threshold": number("임계값", "%.4f"),
    "status": st.column_config.TextColumn("상태"),
    "ap": ratio("AP", color=COLOR["AP"]),
    "recall": ratio("재현율", color=COLOR["재현율"]),
    "precision": ratio("정밀도", color=COLOR["정밀도"]),
    "precision_at_target": number("목표 재현율 지점 정밀도"),
    "fp": number("오경보", "%d"),
    "fn": number("놓친 결함", "%d"),
}


def grouped_runs(df: pd.DataFrame, columns: list, config: dict, key: str) -> None:
    """익스페리먼트별로 접어 두고, 펼치면 그 익스페리먼트의 run 이 최신순으로 나온다."""
    if df.empty:
        st.info("보여 줄 run 이 없습니다.")
        return
    expand_all = st.toggle("모두 펼치기", value=False, key=f"{key}_expand_all")
    order = df.groupby("experiment")["start_time"].max().sort_values(ascending=False).index
    for experiment in order:
        group = df[df["experiment"] == experiment].sort_values("start_time", ascending=False)
        label = (f"**{experiment}**  ·  run {len(group)}개  ·  최고 AP {fmt(group['ap'].max())}  ·  "
                 f"최근 {group['start_time'].max():%m-%d %H:%M}")
        with st.expander(label, expanded=expand_all):
            st.dataframe(group[columns], column_config=config, hide_index=True, use_container_width=True,
                         height=min(460, 36 * (len(group) + 1) + 4))


# ---------- 화면 ----------
try:
    raw_runs = load_raw_runs()
except Exception as error:  # MLflow 에 연결되지 않을 때 원인을 화면에 보여 줌
    st.error(f"MLflow 에 연결하지 못했습니다: {error}\n\nSSH 터널이 열려 있는지, MLFLOW_TRACKING_URI 가 맞는지 확인하세요.")
    st.stop()
if raw_runs.empty:
    st.info("MLflow 에 기록된 run 이 없습니다.")
    st.stop()

runs = prepare(raw_runs)
if runs.empty:
    st.info(f"시드 {FIXED_SEED} run 이 없습니다.")
    st.stop()

with st.sidebar:
    st.markdown("### 보기 설정")
    all_experiments = runs.groupby("experiment")["start_time"].max().sort_values(ascending=False).index.tolist()
    chosen = st.multiselect("익스페리먼트", all_experiments, default=all_experiments)
    only_metrics = st.toggle("지표가 있는 run 만", value=True, help="AP·재현율·정밀도 중 하나라도 기록된 run 만 봅니다.")
    st.divider()
    if st.button("MLflow 에서 다시 읽기", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"MLflow {TRACKING_URI}\n\n1분마다 자동으로 새로 읽습니다.")

selected = runs[runs["experiment"].isin(chosen)]
if only_metrics:
    selected = selected[selected[KEY].notna().any(axis=1)]

st.markdown('<p class="page-title">불량제로 실험 비교</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">KolektorSDD2 표면 결함 검출 · PatchCore 실험을 같은 기준으로 비교합니다.</p>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="chips"><span class="chip">시드 <b>{FIXED_SEED}</b> 고정</span>'
    f'<span class="chip">비교 기준 재현율 <b>≥ {R98}%</b></span>'
    f'<span class="chip">익스페리먼트 <b>{selected["experiment"].nunique()}</b>개</span>'
    f'<span class="chip">run <b>{len(selected)}</b>개</span></div>', unsafe_allow_html=True)

if selected.empty:
    st.info("조건에 맞는 run 이 없습니다. 왼쪽에서 조건을 바꿔 보세요.")
    st.stop()

# 화면 전환: st.tabs 는 위젯을 바꿀 때마다 첫 탭으로 돌아가는 경우가 있어서, 선택값을 세션에 저장하는 버튼으로 바꿈
VIEWS = ["핵심 지표", "run 비교", "run 상세", "자주 틀리는 이미지", "추론 결과"]
st.session_state.setdefault("view", VIEWS[0])
st.session_state.setdefault("last_view", VIEWS[0])


def keep_view():
    """이미 선택된 버튼을 다시 눌러 선택이 풀리면 직전 화면을 유지한다."""
    if st.session_state["view"] is None:
        st.session_state["view"] = st.session_state["last_view"]
    st.session_state["last_view"] = st.session_state["view"]


view = st.segmented_control("화면", VIEWS, key="view", on_change=keep_view, label_visibility="collapsed") or VIEWS[0]
st.markdown(f'<div style="border-bottom:1px solid {LINE}; margin:.2rem 0 .6rem"></div>', unsafe_allow_html=True)

# ----- 핵심 지표 -----
if view == "핵심 지표":
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    rule = left.segmented_control("익스페리먼트 대표 run", ["AP 최고", "가장 최근"], default="AP 최고")
    rule = rule or "AP 최고"
    best = best_per_experiment(selected, rule).copy()
    with st.spinner("재현율 98% 기준 지표를 계산하는 중 ..."):
        extra = pd.DataFrame([recall98(row) for _, row in best.iterrows()], index=best.index)
    best = best.join(extra)
    for key in ["r98_recall", "r98_precision", "r98_fp", "r98_fn"]:
        best[key] = pd.to_numeric(best[key], errors="coerce").astype(float)

    cards = st.columns(4)
    candidates = [("AP 최고", "ap", COLOR["AP"]),
                  (f"재현율 ≥ {R98}% 에서 정밀도 최고", "r98_precision", COLOR["정밀도"])]
    for box, (label, key, color) in zip(cards[:2], candidates):
        have = best.dropna(subset=[key])
        if have.empty:
            kpi(box, label, "–", "값이 있는 익스페리먼트가 없습니다", color)
        else:
            top = have.sort_values(key, ascending=False).iloc[0]
            kpi(box, label, fmt(top[key]), f"{top['experiment']}<br>{top['run_name']}", color)
    have = best.dropna(subset=["r98_fp"])
    if have.empty:
        kpi(cards[2], f"재현율 ≥ {R98}% 에서 오경보 최소", "–", "값이 있는 익스페리먼트가 없습니다", INK)
    else:
        top = have.sort_values("r98_fp").iloc[0]
        kpi(cards[2], f"재현율 ≥ {R98}% 에서 오경보 최소", f"{int(top['r98_fp'])}장",
            f"{top['experiment']}<br>{top['run_name']}", INK)
    kpi(cards[3], "비교한 익스페리먼트", f"{len(best)}개",
        "대표 run: " + ("익스페리먼트마다 AP 가 가장 높은 run" if rule == "AP 최고" else "익스페리먼트마다 가장 최근 run"), MUTED)

    section("익스페리먼트별 핵심 지표",
            f"'기록' 칸은 학습 때 val 로 정한 임계값을 test 에 적용한 결과라서 test 재현율이 {R98}% 아래로 내려갈 수 있습니다. "
            f"'재현율 ≥{R98}%' 칸은 test 에서 재현율이 {R98}% 이상이 되는 가장 높은 임계값을 잡았을 때의 값으로, 모든 익스페리먼트를 같은 기준으로 비교합니다.")
    table = best[["experiment", "ap", "recall", "precision", "r98_recall", "r98_precision", "r98_fp", "r98_fn",
                  "r98_source", "run_name"]]
    config = dict(BASE_CONFIG)
    config.update({
        "recall": ratio("재현율 (기록)", "학습 때 val 로 정한 임계값을 test 에 적용한 결과", COLOR["재현율"]),
        "precision": ratio("정밀도 (기록)", "학습 때 val 로 정한 임계값을 test 에 적용한 결과", COLOR["정밀도"]),
        "r98_recall": ratio(f"재현율 ≥{R98}% · 재현율", color=COLOR["재현율"]),
        "r98_precision": ratio(f"재현율 ≥{R98}% · 정밀도", "불량이라고 판정한 것 중 진짜 결함의 비율. 높을수록 오경보가 적습니다.",
                               COLOR["정밀도"]),
        "r98_fp": number(f"재현율 ≥{R98}% · 오경보", "%d"),
        "r98_fn": number(f"재현율 ≥{R98}% · 놓친 결함", "%d"),
        "r98_source": st.column_config.TextColumn("계산 방법"),
    })
    st.dataframe(table, column_config=config, hide_index=True, use_container_width=True)

    section("한눈에 보기", f"AP 와 재현율 ≥ {R98}% 기준 재현율·정밀도입니다. 막대 끝 숫자가 값입니다.")
    chart_source = best.rename(columns={"ap": "AP", "r98_recall": "재현율", "r98_precision": "정밀도"})
    long = chart_source.melt(id_vars=["experiment", "run_name"], value_vars=list(COLOR),
                             var_name="지표", value_name="value").dropna(subset=["value"])
    order = best["experiment"].tolist()
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("experiment:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelLimit=260, labelFontSize=12, labelColor=INK)),
        yOffset=alt.YOffset("지표:N", sort=list(COLOR)),
        x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 1.08]),
                axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], format=".1f", grid=True, gridColor=LINE)),
        color=alt.Color("지표:N", sort=list(COLOR),
                        scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
        tooltip=["experiment", "run_name", "지표", alt.Tooltip("value:Q", format=".3f")])
    labels = bars.mark_text(align="left", dx=4, fontSize=11, color=MUTED).encode(
        text=alt.Text("value:Q", format=".3f"))
    st.altair_chart((bars + labels).properties(height=max(160, 78 * len(order)))
                    .configure_view(strokeWidth=0).configure_axis(domainColor=LINE, tickColor=LINE, labelColor=MUTED),
                    use_container_width=True)

# ----- run 비교 -----
if view == "run 비교":
    section("run 비교", f"시드 {FIXED_SEED} run 전체입니다. 익스페리먼트를 누르면 그 안의 run 이 최신순으로 펼쳐집니다. "
                        "기준 run 을 고르면 각 run 이 기준보다 얼마나 달라졌는지 붙습니다. 값은 학습 때 기록된 지표입니다.")
    ordered = selected
    latest_first = selected.sort_values("start_time", ascending=False)   # 고르는 목록은 최신순
    names = {row.run_id: f"{row.start_time:%m-%d %H:%M} · {row.experiment} · {row.run_name}"
             for row in latest_first.itertuples()}
    baseline_id = st.selectbox("기준 run", ["(없음)"] + list(names),
                               format_func=lambda r: r if r == "(없음)" else names[r], key="compare_base")
    columns = ["experiment", "run_name", "ap", "recall", "precision", "precision_at_target", "fp", "fn"]
    shown = ordered[columns + ["threshold", "status", "start_time"]].copy()
    layout = columns[1:]
    config = dict(BASE_CONFIG)
    if baseline_id != "(없음)":
        base = selected[selected["run_id"] == baseline_id].iloc[0]
        layout = ["run_name"]
        for key in columns[2:]:
            shown[f"{key}_delta"] = ordered[key] - base[key]
            config[f"{key}_delta"] = number("차이", "%+d" if key in COUNT else "%+.3f")
            layout += [key, f"{key}_delta"]
        st.caption(f"기준: {base['experiment']} · {base['run_name']} — AP {fmt(base['ap'])} · 재현율 {fmt(base['recall'])} "
                   f"· 정밀도 {fmt(base['precision'])}. 오경보·놓친 결함은 차이가 음수일수록 좋습니다.")
    grouped_runs(shown, ["start_time"] + layout + ["threshold", "status"], config, "compare")
    with st.expander("익스페리먼트별로 어떤 지표 이름을 읽었는지"):
        sources = (selected.groupby("experiment")[[f"{k}_source" for k in KEY]]
                   .agg(lambda s: ", ".join(sorted({v for v in s if v})) or "없음"))
        sources.columns = [LABEL[k] for k in KEY]
        st.dataframe(sources, use_container_width=True)
        st.caption("'없음'이면 그 익스페리먼트에는 해당 지표가 기록되지 않았거나 이름이 달라 찾지 못한 것입니다.")

# ----- run 상세 -----
if view == "run 상세":
    options = selected.sort_values("start_time", ascending=False)   # 최신순
    labels = {row.run_id: f"{row.start_time:%m-%d %H:%M} · {row.experiment} · {row.run_name}" for row in options.itertuples()}
    run_id = st.selectbox("run", list(labels), format_func=labels.get, key="detail_run")
    run = selected[selected["run_id"] == run_id].iloc[0]
    detail = recall98(run)
    boxes = st.columns(4)
    kpi(boxes[0], "AP", fmt(run["ap"]), run["experiment"], COLOR["AP"])
    kpi(boxes[1], f"재현율 ≥{R98}% · 재현율", fmt(detail["r98_recall"]), detail["r98_source"], COLOR["재현율"])
    kpi(boxes[2], f"재현율 ≥{R98}% · 정밀도", fmt(detail["r98_precision"]), detail["r98_source"], COLOR["정밀도"])
    kpi(boxes[3], "오경보 / 놓친 결함",
        "–" if pd.isna(detail["r98_fp"]) else f"{int(detail['r98_fp'])} / {int(detail['r98_fn'])}",
        f"재현율 ≥{R98}% 기준", INK)
    st.caption(f"학습 때 기록: 재현율 {fmt(run['recall'])} · 정밀도 {fmt(run['precision'])} · "
               f"크기별 재현율(작은/중간/큰) {fmt(run['recall_small'])} / {fmt(run['recall_medium'])} / {fmt(run['recall_large'])}")
    try:
        client = MlflowClient()
        artifacts = [a for a in client.list_artifacts(run_id) if not a.is_dir]
    except Exception as error:
        artifacts = []
        st.warning(f"artifact 목록을 읽지 못했습니다: {error}")
    images = [a.path for a in artifacts if a.path.endswith(".png")]
    texts = [a.path for a in artifacts if a.path.endswith("_hard_defects.txt")]
    if images:
        section("그림", "히트맵이 먼저 나옵니다.")
        with tempfile.TemporaryDirectory() as folder:
            for path in sorted(images, key=lambda p: ("heatmap" not in p, p)):
                local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=folder)
                st.image(local, caption=path, use_container_width=True)
            for path in texts:
                local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=folder)
                with open(local, encoding="utf-8") as f:
                    with st.expander(path):
                        st.text(f.read())
    else:
        st.info("이 run 에는 그림 artifact 가 없습니다.")

# ----- 추론 결과 -----
if view == "추론 결과":
    section("추론 결과", "run 이 이미지마다 내린 판정을 봅니다. 두 번째 run 을 고르면 같은 이미지의 판정이 어떻게 바뀌었는지 비교합니다.")
    infer_options = selected.sort_values("start_time", ascending=False)   # 최신순
    infer_labels = {row.run_id: f"{row.start_time:%m-%d %H:%M} · {row.experiment} · {row.run_name}"
                    for row in infer_options.itertuples()}
    c1, c2, c3 = st.columns([2, 2, 1.3])
    run_a = c1.selectbox("run", list(infer_labels), format_func=infer_labels.get, key="infer_a")
    compare_ids = ["(비교 안 함)"] + [r for r in infer_labels if r != run_a]
    run_b = c2.selectbox("비교할 run", compare_ids,
                         format_func=lambda r: r if r == "(비교 안 함)" else infer_labels[r], key="infer_b")
    threshold_mode = c3.radio("임계값", [f"재현율 ≥{R98}% 지점", "학습 때 값"], key="infer_mode",
                              help=f"재현율 ≥{R98}% 지점: 보고 있는 데이터에서 재현율이 {R98}% 이상이 되는 가장 높은 임계값")
    split = st.segmented_control("데이터", ["test", "val"], default="test", key="infer_split") or "test"

    def threshold_of(info, scores):
        if threshold_mode == "학습 때 값":
            return info["threshold"]
        return threshold_for_recall(scores[scores["split"] == split], FIXED_RECALL)

    info_a = selected[selected["run_id"] == run_a].iloc[0]
    try:
        scores_a = load_scores(run_a)
    except Exception as error:
        scores_a = pd.DataFrame()
        st.warning(f"점수 파일을 읽지 못했습니다: {error}")
    threshold_a = np.nan if scores_a.empty else threshold_of(info_a, scores_a)
    if scores_a.empty or pd.isna(threshold_a):
        st.info("이 run 에는 이미지별 점수 파일(*_scores.csv) 또는 임계값 기록이 없습니다.")
    else:
        judged_a = judge(scores_a[scores_a["split"] == split], threshold_a)
        counts = judged_a["결과"].value_counts()
        boxes = st.columns(4)
        for box, (name, color) in zip(boxes, [("결함 검출", COLOR["재현율"]), ("놓침", DEFECT_COLOR),
                                              ("오경보", COLOR["정밀도"]), ("정상 통과", MUTED)]):
            kpi(box, name, f"{int(counts.get(name, 0))}장", "", color)
        defects = judged_a[judged_a["label"] == 1]
        flagged = judged_a[judged_a["판정"] == "불량"]
        st.caption(f"임계값 {threshold_a:.4f} · 재현율 {(defects['판정'] == '불량').mean():.3f} · "
                   f"정밀도 {(flagged['label'] == 1).mean() if len(flagged) else 0:.3f} ({split} 기준)")

        section("이상 점수 분포", "점선이 임계값입니다. 정상(파랑)이 점선 오른쪽이면 오경보, 결함(주황)이 왼쪽이면 놓친 결함입니다.")
        histogram_data = judged_a.assign(구분=np.where(judged_a["label"] == 1, "결함", "정상"))
        histogram = alt.Chart(histogram_data).mark_bar(opacity=0.6).encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=60), title="이상 점수"),
            y=alt.Y("count():Q", stack=None, title="이미지 수"),
            color=alt.Color("구분:N", scale=alt.Scale(domain=["정상", "결함"], range=[NORMAL_COLOR, DEFECT_COLOR]),
                            legend=alt.Legend(orient="top", title=None)))
        threshold_rule = alt.Chart(pd.DataFrame({"x": [threshold_a]})).mark_rule(
            color=INK, strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((histogram + threshold_rule).properties(height=280).configure_view(strokeWidth=0)
                        .configure_axis(gridColor=LINE, domainColor=LINE, labelColor=MUTED, titleColor=MUTED),
                        use_container_width=True)

        wrong_columns = ["file", "결과", "score", "임계값 대비", "mask_area"]
        wrong_config = {"score": number("점수", "%.4f"), "임계값 대비": number("임계값 대비", "%.2f×"),
                        "mask_area": number("결함 면적(px)", "%d"), "file": "파일", "결과": "결과"}
        col_fp, col_fn = st.columns(2)
        with col_fp:
            section("오경보", "점수가 높은 순서입니다.")
            st.dataframe(judged_a[judged_a["결과"] == "오경보"].sort_values("score", ascending=False)[wrong_columns],
                         column_config=wrong_config, hide_index=True, use_container_width=True)
        with col_fn:
            section("놓친 결함과 아슬아슬한 결함", "임계값의 1.15배 미만인 결함까지 보여 줍니다.")
            st.dataframe(defects[defects["임계값 대비"] < 1.15].sort_values("score")[wrong_columns],
                         column_config=wrong_config, hide_index=True, use_container_width=True)

        if run_b != "(비교 안 함)":
            info_b = selected[selected["run_id"] == run_b].iloc[0]
            scores_b = load_scores(run_b)
            threshold_b = np.nan if scores_b.empty else threshold_of(info_b, scores_b)
            if scores_b.empty or pd.isna(threshold_b):
                st.info("비교할 run 에 점수 파일 또는 임계값 기록이 없습니다.")
            elif (pd.notna(info_a["split_hash"]) and pd.notna(info_b["split_hash"])
                  and info_a["split_hash"] != info_b["split_hash"]):
                st.warning("두 run 의 분할(split_hash)이 달라 같은 이미지끼리 비교할 수 없습니다.")
            else:
                judged_b = judge(scores_b[scores_b["split"] == split], threshold_b)
                merged = judged_a.merge(judged_b, on=["file", "label"], suffixes=(" A", " B"))
                merged["변화"] = np.select(
                    [(merged["결과 A"].isin(["놓침", "오경보"])) & (merged["결과 B"].isin(["결함 검출", "정상 통과"])),
                     (merged["결과 A"].isin(["결함 검출", "정상 통과"])) & (merged["결과 B"].isin(["놓침", "오경보"]))],
                    ["B에서 고쳐짐", "B에서 새로 틀림"], default="같음")
                section(f"A: {info_a['run_name']}  vs  B: {info_b['run_name']}")
                change_counts = merged["변화"].value_counts()
                boxes = st.columns(3)
                kpi(boxes[0], "B에서 고쳐짐", f"{int(change_counts.get('B에서 고쳐짐', 0))}장", "", COLOR["재현율"])
                kpi(boxes[1], "B에서 새로 틀림", f"{int(change_counts.get('B에서 새로 틀림', 0))}장", "", DEFECT_COLOR)
                kpi(boxes[2], "판정 같음", f"{int(change_counts.get('같음', 0))}장", "", MUTED)
                merged["구분"] = np.where(merged["label"] == 1, "결함", "정상")
                scatter = alt.Chart(merged).mark_circle(size=45, opacity=0.7).encode(
                    x=alt.X("임계값 대비 A:Q", title="A 점수 / A 임계값", scale=alt.Scale(zero=False)),
                    y=alt.Y("임계값 대비 B:Q", title="B 점수 / B 임계값", scale=alt.Scale(zero=False)),
                    color=alt.Color("구분:N", scale=alt.Scale(domain=["정상", "결함"], range=[NORMAL_COLOR, DEFECT_COLOR]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["file", "구분", "결과 A", "결과 B", alt.Tooltip("임계값 대비 A:Q", format=".2f"),
                             alt.Tooltip("임계값 대비 B:Q", format=".2f")])
                guide_v = alt.Chart(pd.DataFrame({"v": [1.0]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="v:Q")
                guide_h = alt.Chart(pd.DataFrame({"v": [1.0]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="v:Q")
                st.altair_chart((scatter + guide_v + guide_h).properties(height=420).configure_view(strokeWidth=0)
                                .configure_axis(gridColor=LINE, domainColor=LINE, labelColor=MUTED, titleColor=MUTED),
                                use_container_width=True)
                st.caption("점선(1.0)이 각 run 의 임계값입니다. 결함(주황)은 위·오른쪽, 정상(파랑)은 아래·왼쪽에 있어야 맞게 판정한 것입니다.")
                changed = merged[merged["변화"] != "같음"].sort_values(["변화", "file"])
                st.dataframe(changed[["file", "구분", "변화", "결과 A", "결과 B", "임계값 대비 A", "임계값 대비 B"]],
                             column_config={"임계값 대비 A": number("A 임계값 대비", "%.2f×"),
                                            "임계값 대비 B": number("B 임계값 대비", "%.2f×"), "file": "파일"},
                             hide_index=True, use_container_width=True)

# ----- 자주 틀리는 이미지 -----
def size_group(area):
    if pd.isna(area) or area <= 0:
        return "–"
    return "작은 결함" if area < 500 else "중간 결함" if area < 3000 else "큰 결함"


if view == "자주 틀리는 이미지":
    section("자주 틀리는 이미지",
            "여러 run 의 이미지별 점수 파일을 모아서, 어떤 이미지를 몇 번 틀렸는지 셉니다. "
            "같은 분할(split_hash)로 돌린 run 끼리만 모읍니다. 시드 42 고정이라 test 이미지 구성이 같습니다.")
    h1, h2, h3 = st.columns([1.3, 1.3, 1])
    hard_scope = h1.segmented_control("모을 run", ["익스페리먼트 대표 run", "모든 run"], default="익스페리먼트 대표 run",
                                      key="hard_scope",
                                      help="대표 run: 익스페리먼트마다 AP 가 가장 높은 run 1개. run 을 많이 돌린 익스페리먼트가 결과를 끌고 가지 않게 합니다.")
    hard_scope = hard_scope or "익스페리먼트 대표 run"
    hard_mode = h2.segmented_control("임계값", [f"재현율 ≥{R98}% 지점", "학습 때 값"], default=f"재현율 ≥{R98}% 지점",
                                     key="hard_mode") or f"재현율 ≥{R98}% 지점"
    hard_split = h3.segmented_control("데이터", ["test", "val"], default="test", key="hard_split") or "test"

    pool = best_per_experiment(selected, "AP 최고") if hard_scope == "익스페리먼트 대표 run" else selected
    pool = pool[pool["split_hash"].notna()]
    collected = []
    with st.spinner("run 마다 점수 파일을 받는 중 ..."):
        for row in pool.itertuples():
            try:
                scores = load_scores(row.run_id)
            except Exception:
                continue
            if scores.empty:
                continue
            part = scores[scores["split"] == hard_split]
            threshold = (row.threshold if hard_mode == "학습 때 값" else threshold_for_recall(part, FIXED_RECALL))
            if part.empty or pd.isna(threshold):
                continue
            judged = judge(part, threshold)
            judged["run"] = f"{row.experiment} · {row.run_name}"
            judged["split_hash"] = row.split_hash
            collected.append(judged)

    if not collected:
        st.info("점수 파일(*_scores.csv)이 있는 run 이 없습니다. 분할 고정 이후 baseline.py 로 돌린 run 만 모을 수 있습니다.")
    else:
        everything = pd.concat(collected, ignore_index=True)
        main_hash = everything.groupby("split_hash")["run"].nunique().idxmax()
        dropped = everything.loc[everything["split_hash"] != main_hash, "run"].nunique()
        everything = everything[everything["split_hash"] == main_hash]
        n_runs = everything["run"].nunique()
        everything["틀림"] = everything["결과"].isin(["놓침", "오경보"])

        stats = (everything.groupby(["file", "label"])
                 .agg(틀린_run=("틀림", "sum"), 전체_run=("run", "nunique"),
                      평균_임계값대비=("임계값 대비", "mean"), 최소_임계값대비=("임계값 대비", "min"),
                      최대_임계값대비=("임계값 대비", "max"),
                      mask_area=("mask_area", "first") if "mask_area" in everything else ("label", "size"))
                 .reset_index())
        stats["틀린 비율"] = stats["틀린_run"] / stats["전체_run"]
        wrong_runs = (everything[everything["틀림"]].groupby("file")["run"]
                      .agg(lambda r: ", ".join(sorted(r))).rename("틀린 run"))
        stats = stats.merge(wrong_runs, on="file", how="left").fillna({"틀린 run": ""})
        stats["크기"] = [size_group(a) if l == 1 else "–" for a, l in zip(stats["mask_area"], stats["label"])]

        defects = stats[stats["label"] == 1]
        normals = stats[stats["label"] == 0]
        always_missed = int((defects["틀린_run"] == n_runs).sum())
        always_alarm = int((normals["틀린_run"] == n_runs).sum())
        once_missed = int((defects["틀린_run"] > 0).sum())
        once_alarm = int((normals["틀린_run"] > 0).sum())

        st.caption(f"run {n_runs}개 · {hard_split} 결함 {len(defects)}장 · 정상 {len(normals)}장"
                   + (f" · 분할이 다른 run {dropped}개는 뺐습니다" if dropped else ""))
        boxes = st.columns(4)
        kpi(boxes[0], "모든 run 에서 놓친 결함", f"{always_missed}장", f"한 번이라도 놓친 결함 {once_missed}장", DEFECT_COLOR)
        kpi(boxes[1], "모든 run 에서 오경보", f"{always_alarm}장", f"한 번이라도 오경보 {once_alarm}장", COLOR["정밀도"])
        missed_any = defects[defects["틀린_run"] > 0]
        by_size = missed_any["크기"].value_counts()
        top_size = by_size.index[0] if len(by_size) else "–"
        kpi(boxes[2], "놓친 결함이 가장 많은 크기", top_size,
            (" · ".join(f"{k.split(' ')[0]} {v}장" for k, v in by_size.items()) or "놓친 결함 없음")
            + "<br>작은 &lt;500px · 중간 500~3000px · 큰 ≥3000px", INK)
        kpi(boxes[3], "모은 run", f"{n_runs}개", hard_scope, MUTED)

        hard_config = {
            "file": st.column_config.TextColumn("파일"),
            "틀린_run": number("틀린 run 수", "%d"),
            "전체_run": number("전체 run 수", "%d"),
            "틀린 비율": ratio("틀린 비율", color=DEFECT_COLOR),
            "평균_임계값대비": number("평균 점수/임계값", "%.2f×",
                                 "1 보다 작으면 정상으로 판정, 크면 불량으로 판정. 결함은 클수록, 정상은 작을수록 좋습니다."),
            "최소_임계값대비": number("최소", "%.2f×"),
            "최대_임계값대비": number("최대", "%.2f×"),
            "mask_area": number("결함 면적(px)", "%d"),
            "크기": st.column_config.TextColumn("크기"),
            "틀린 run": st.column_config.TextColumn("틀린 run", width="large"),
        }
        show_all = st.toggle("한 번도 안 틀린 이미지도 보기", value=False, key="hard_all")

        col_d, col_n = st.columns(2)
        with col_d:
            section("놓친 결함", "많이 놓친 순서, 같으면 점수가 낮은 순서입니다.")
            table_d = defects if show_all else defects[defects["틀린_run"] > 0]
            table_d = table_d.sort_values(["틀린_run", "평균_임계값대비"], ascending=[False, True])
            st.dataframe(table_d[["file", "틀린_run", "전체_run", "틀린 비율", "평균_임계값대비", "최소_임계값대비",
                                  "mask_area", "크기", "틀린 run"]],
                         column_config=hard_config, hide_index=True, use_container_width=True,
                         height=min(420, 36 * (len(table_d) + 1) + 4))
        with col_n:
            section("오경보", "오경보를 많이 낸 순서, 같으면 점수가 높은 순서입니다.")
            table_n = normals if show_all else normals[normals["틀린_run"] > 0]
            table_n = table_n.sort_values(["틀린_run", "평균_임계값대비"], ascending=[False, False])
            st.dataframe(table_n[["file", "틀린_run", "전체_run", "틀린 비율", "평균_임계값대비", "최대_임계값대비", "틀린 run"]],
                         column_config=hard_config, hide_index=True, use_container_width=True,
                         height=min(420, 36 * (len(table_n) + 1) + 4))

        section("이미지 × run 판정표",
                "많이 틀린 이미지 상위 30장을 run 별로 펼쳤습니다. 칸 색은 점수/임계값이고, ✕ 가 틀린 판정입니다. "
                "결함은 1 보다 커야, 정상은 1 보다 작아야 맞게 판정한 것입니다.")
        top_files = pd.concat([table_d.head(15), table_n.head(15)])["file"].tolist()
        grid = everything[everything["file"].isin(top_files)].copy()
        if grid.empty:
            st.info("틀린 이미지가 없습니다.")
        else:
            grid["구분"] = np.where(grid["label"] == 1, "결함", "정상")
            grid["이미지"] = grid["구분"] + " · " + grid["file"]
            grid["표시"] = np.where(grid["틀림"], "✕", "")
            grid["비율"] = grid["임계값 대비"].clip(0.5, 1.5)
            image_order = [("결함 · " if f in set(table_d["file"]) else "정상 · ") + f for f in top_files]
            base = alt.Chart(grid).encode(
                x=alt.X("run:N", title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=230, labelOverlap=False,
                                                            labelFontSize=10.5, labelColor=MUTED)),
                y=alt.Y("이미지:N", sort=image_order, title=None, axis=alt.Axis(labelColor=INK, labelLimit=200)))
            cells = base.mark_rect(stroke="white", strokeWidth=1).encode(
                color=alt.Color("비율:Q", title="점수/임계값",
                                scale=alt.Scale(domain=[0.5, 1.0, 1.5], range=[NORMAL_COLOR, "#f4f1ea", DEFECT_COLOR])),
                tooltip=["file", "구분", "run", "결과", alt.Tooltip("임계값 대비:Q", format=".2f")])
            marks = base.mark_text(fontSize=12, fontWeight="bold", color=INK).encode(text="표시:N")
            st.altair_chart((cells + marks).properties(height=alt.Step(24))
                            .configure_view(strokeWidth=0), use_container_width=True)
            st.caption("파랑 = 점수가 임계값보다 낮음(정상 판정), 주황 = 임계값보다 높음(불량 판정). "
                       "결함 행에 파랑이 있거나 정상 행에 주황이 있으면 틀린 것입니다.")
