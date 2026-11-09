"""불량제로 실험 비교 대시보드 (Streamlit)

MLflow 에 기록된 모든 익스페리먼트의 run 을 읽어서 팀 성능지표(val 기준)를 비교한다.
  - 성능지표 8개 (MLflow 이름 그대로 읽음, 다른 이름은 섞어 읽지 않음):
    val_average_precision, val_auroc, val_recall, val_precision,
    val_false_positives(오검출), val_false_negatives(놓친 결함), val_false_positive_rate(오검출률), threshold
  - 시드는 42 로 고정: seed(또는 model_seed) 가 42 가 아닌 run 은 뺀다. 기록이 없는 run 은 42 로 본다.
  - 이미지별 점수 파일은 두 형식을 모두 읽는다: <split>/scores.csv (팀 baseline) 와 *_scores.csv (예전 코드).

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
COLOR = {"Average Precision": "#2b5d78", "AUROC": "#5d6fa8", "재현율": "#3c8a68", "정밀도": "#c27a2c",
         "오검출률": "#b4553b"}
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

# 팀 성능지표 (11/09 확정): (보여 줄 이름, MLflow 지표 이름, 이름 패턴, 제외 패턴)
# 모두 val 기준이라 이름을 정확히 그대로 읽는다. 이름이 다른 지표(test_*, ap, auprc 등)는 섞어 읽지 않는다.
METRIC_RULES = {
    "ap": ("Average Precision", ["val_average_precision"], None, None),
    "auroc": ("AUROC", ["val_auroc"], None, None),
    "recall": ("재현율", ["val_recall"], None, None),
    "precision": ("정밀도", ["val_precision"], None, None),
    "fp": ("오검출", ["val_false_positives"], None, None),
    "fn": ("놓친 결함", ["val_false_negatives"], None, None),
    "fpr": ("오검출률", ["val_false_positive_rate"], None, None),
}
LABEL = {key: rule[0] for key, rule in METRIC_RULES.items()}
KEY = ["ap", "recall", "precision"]
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

    # 시드: params.seed → params.model_seed(팀 baseline) → run 이름의 seed숫자 → 그래도 없으면 42 로 본다
    seed = pd.to_numeric(column(raw, "params.seed"), errors="coerce")
    seed = seed.fillna(pd.to_numeric(column(raw, "params.model_seed"), errors="coerce"))
    from_name = pd.to_numeric(df["run_name"].str.extract(r"seed[_-]?(\d+)", flags=re.I)[0], errors="coerce")
    df["seed"] = seed.fillna(from_name).fillna(FIXED_SEED).astype(int)

    for key, (_, exact, pattern, exclude) in METRIC_RULES.items():
        value, source = pick_metric(raw, exact, pattern, exclude)
        df[key], df[f"{key}_source"] = value.astype(float), source
    threshold = pd.to_numeric(column(raw, "metrics.threshold"), errors="coerce")
    df["threshold"] = threshold.fillna(pd.to_numeric(column(raw, "params.threshold"), errors="coerce"))

    # 과거 실험(test 기준으로 기록해 val 지표가 없는 run)은 보여주지 않는다
    has_val_metrics = df["ap"].notna()
    keep = (df["seed"] == FIXED_SEED) & (df["kind"] != "seed_summary") & (df["status"] != "RUNNING") & has_val_metrics
    return df[keep].copy()


@st.cache_data(ttl=600, show_spinner=False)
def load_scores(run_id: str) -> pd.DataFrame:
    """run 의 이미지별 점수를 (split, file, label, score, mask_area) 표로 읽는다. 없으면 빈 표.
    - 팀 baseline: val/scores.csv, test/scores.csv (file, label, score) → split 칸을 붙여 합침
    - 예전 코드: 맨 위의 *_scores.csv (split, file, label, score, mask_area)"""
    client = MlflowClient()
    top = client.list_artifacts(run_id)
    parts = []
    with tempfile.TemporaryDirectory() as folder:
        for split in ("val", "test"):
            if any(a.is_dir and a.path == split for a in top):
                if any(a.path == f"{split}/scores.csv" for a in client.list_artifacts(run_id, split)):
                    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{split}/scores.csv",
                                                                dst_path=folder)
                    parts.append(pd.read_csv(local, encoding="utf-8-sig").assign(split=split))
        if not parts:
            old = [a.path for a in top if a.path.endswith("_scores.csv")]
            if old:
                local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=old[0], dst_path=folder)
                parts.append(pd.read_csv(local, encoding="utf-8-sig"))
    if not parts:
        return pd.DataFrame()
    scores = pd.concat(parts, ignore_index=True)
    if "mask_area" not in scores.columns:
        scores["mask_area"] = np.nan        # 팀 baseline 은 결함 면적을 기록하지 않음
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


def best_per_experiment(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """익스페리먼트마다 대표 run 하나 (AP 최고 또는 가장 최근)."""
    if rule == "Average Precision 최고":
        ordered = df.sort_values(["ap", "recall", "start_time"], ascending=[False, False, False], na_position="last")
    else:
        ordered = df.sort_values("start_time", ascending=False)
    return ordered.drop_duplicates("experiment", keep="first").sort_values("ap", ascending=False, na_position="last")


def judge(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """임계값으로 판정하고 결과 유형(정상 통과/오검출/결함 검출/놓침)을 붙인다."""
    judged = scores.copy()
    judged["판정"] = np.where(judged["score"] >= threshold, "불량", "정상")
    judged["임계값 대비"] = judged["score"] / threshold
    judged["결과"] = np.select(
        [(judged["label"] == 1) & (judged["판정"] == "불량"), (judged["label"] == 1) & (judged["판정"] == "정상"),
         (judged["label"] == 0) & (judged["판정"] == "불량")],
        ["결함 검출", "놓침", "오검출"], default="정상 통과")
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
    """0~1 값을 막대로 보여 주는 칸. color 는 지표 색 (AP 파랑 · AUROC 남색 · 재현율 초록 · 정밀도 주황 · 오검출률 빨강)."""
    return st.column_config.ProgressColumn(label, format="%.3f", min_value=0.0, max_value=1.0, help=help_text,
                                           color=color or COLOR["Average Precision"])


def number(label, fmt_text="%.3f", help_text=None):
    return st.column_config.NumberColumn(label, format=fmt_text, help=help_text)


def show_table(data: pd.DataFrame, column_config: dict | None = None, **kwargs) -> None:
    """st.dataframe 과 같지만, 값이 없는 숫자 칸을 'None' 대신 공란으로 보여 준다.
    Streamlit 은 빈 숫자 칸에 'None' 을 표시하고 이를 바꾸는 옵션이 없어서, 숫자 칸(NumberColumn)을
    정해 둔 형식의 글자로 바꾸고 빈 값은 빈 글자로 둔다. 막대 칸(ProgressColumn)은 빈 값이 원래 공란이라 그대로 둔다."""
    data = data.copy()
    config = dict(column_config or {})
    for name, setting in list(config.items()):
        if name not in data.columns or not isinstance(setting, dict):
            continue
        type_config = setting.get("type_config") or {}
        if type_config.get("type") != "number":
            continue
        printf = type_config.get("format") or "%.3f"
        python_format = re.sub(r"%(\+?)(\.\d+)?[df]", lambda m: "{:" + m[1] + (m[2] or ".0") + "f}", printf)
        data[name] = [("" if pd.isna(value) else python_format.format(value)) for value in data[name]]
        config[name] = st.column_config.TextColumn(setting.get("label"), help=setting.get("help"))
    st.dataframe(data, column_config=config, **kwargs)


BASE_CONFIG = {
    "experiment": st.column_config.TextColumn("익스페리먼트", width="medium"),
    "run_name": st.column_config.TextColumn("run", width="medium"),
    "start_time": st.column_config.DatetimeColumn("시작", format="MM-DD HH:mm"),
    "threshold": number("임계값", "%.4f", f"val 에서 재현율 {int(round(FIXED_RECALL * 100))}% 이상이 되도록 정한 판정 기준"),
    "status": st.column_config.TextColumn("상태"),
    "ap": ratio("Average Precision", "val. 임계값과 상관없이 결함을 정상보다 높은 점수로 줄 세운 정도 (PR 곡선 아래 넓이)",
                COLOR["Average Precision"]),
    "auroc": ratio("AUROC", "val. 임계값과 상관없이 결함과 정상을 가르는 정도 (ROC 곡선 아래 넓이)", COLOR["AUROC"]),
    "recall": ratio("재현율", "val. 정한 임계값으로 판정했을 때 결함 중 잡아낸 비율", COLOR["재현율"]),
    "precision": ratio("정밀도", "val. 불량이라고 판정한 것 중 진짜 결함의 비율", COLOR["정밀도"]),
    "fp": number("오검출", "%d", "val. 정상인데 불량으로 판정한 장수"),
    "fn": number("놓친 결함", "%d", "val. 결함인데 정상으로 판정한 장수"),
    "fpr": ratio("오검출률", "val. 오검출 ÷ 정상 장수. 낮을수록 좋습니다.", COLOR["오검출률"]),
}


KEY_COLUMNS = ["ap", "auroc", "recall", "precision", "fp", "fn", "fpr", "threshold"]   # 팀 성능지표 8개
CHART_METRICS = ["Average Precision", "정밀도"]   # 한눈에 보기 막대그래프에 그릴 지표


def key_metric_config() -> dict:
    """핵심 지표 표와 run 비교 표가 같이 쓰는 칸 설정."""
    return dict(BASE_CONFIG)


def grouped_runs(df: pd.DataFrame, columns: list, config: dict, key: str) -> None:
    """익스페리먼트별로 접어 두고, 펼치면 그 익스페리먼트의 run 이 최신순으로 나온다."""
    if df.empty:
        st.info("보여 줄 run 이 없습니다.")
        return
    expand_all = st.toggle("모두 펼치기", value=False, key=f"{key}_expand_all")
    order = df.groupby("experiment")["start_time"].max().sort_values(ascending=False).index
    for experiment in order:
        group = df[df["experiment"] == experiment].sort_values("start_time", ascending=False)
        best_ap = group["ap"].max()
        label = (f"**{experiment}**  ·  run {len(group)}개  ·  "
                 + ("" if pd.isna(best_ap) else f"최고 Average Precision {fmt(best_ap)}  ·  ")
                 + f"최근 {group['start_time'].max():%m-%d %H:%M}")
        with st.expander(label, expanded=expand_all):
            show_table(group[columns], column_config=config, hide_index=True, use_container_width=True,
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
    st.caption("팀 성능지표(val)가 기록되지 않은 예전 run 도 모두 보여 주고, 없는 지표 칸은 비워 둡니다.")
    st.divider()
    if st.button("MLflow 에서 다시 읽기", use_container_width=True):
        load_raw_runs.clear()
        st.rerun()
    st.caption(f"MLflow {TRACKING_URI}\n\n1분마다 자동으로 새로 읽습니다.")

selected = runs[runs["experiment"].isin(chosen)]

st.markdown('<p class="page-title">불량제로 실험 비교</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">KolektorSDD2 표면 결함 검출 · PatchCore 실험을 같은 기준으로 비교합니다.</p>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="chips"><span class="chip">시드 <b>{FIXED_SEED}</b> 고정</span>'
    f'<span class="chip">평가 데이터 <b>val</b></span>'
    f'<span class="chip">임계값 <b>val 재현율 ≥ {R98}%</b></span>'
    f'<span class="chip">익스페리먼트 <b>{selected["experiment"].nunique()}</b>개</span>'
    f'<span class="chip">run <b>{len(selected)}</b>개</span></div>', unsafe_allow_html=True)

if selected.empty:
    st.info("조건에 맞는 run 이 없습니다. 왼쪽에서 조건을 바꿔 보세요.")
    st.stop()

# 화면 전환: st.tabs 는 위젯을 바꿀 때마다 첫 탭으로 돌아가는 경우가 있어서, 선택값을 세션에 저장하는 버튼으로 바꿈
VIEWS = ["핵심 지표", "run 비교", "지표 관계", "run 상세", "자주 틀리는 이미지", "추론 결과"]
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
    rule = left.segmented_control("익스페리먼트 대표 run", ["Average Precision 최고", "가장 최근"], default="Average Precision 최고")
    rule = rule or "Average Precision 최고"
    best = best_per_experiment(selected, rule).copy()

    cards = st.columns(4)
    # 정밀도·오검출 카드는 임계값에 따라 달라지므로 재현율·놓친 결함을 함께 표시
    candidates = [("Average Precision 최고", "ap", COLOR["Average Precision"]),
                  ("정밀도 최고", "precision", COLOR["정밀도"])]
    for box, (label, key, color) in zip(cards[:2], candidates):
        have = best.dropna(subset=[key])
        if have.empty:
            kpi(box, label, "–", "값이 있는 익스페리먼트가 없습니다", color)
        else:
            top = have.sort_values(key, ascending=False).iloc[0]
            note = f" · 재현율 {fmt(top['recall'])}" if key == "precision" else ""
            kpi(box, label, fmt(top[key]), f"{top['experiment']}{note}<br>{top['run_name']}", color)
    have = best.dropna(subset=["fp"])
    if have.empty:
        kpi(cards[2], "오검출 최소", "–", "값이 있는 익스페리먼트가 없습니다", INK)
    else:
        top = have.sort_values("fp").iloc[0]
        missed = "" if pd.isna(top["fn"]) else f" · 놓친 결함 {int(top['fn'])}장"
        rate = "" if pd.isna(top["fpr"]) else f" ({top['fpr']:.1%})"
        kpi(cards[2], "오검출 최소", f"{int(top['fp'])}장{rate}",
            f"{top['experiment']}{missed}<br>{top['run_name']}", INK)
    kpi(cards[3], "비교한 익스페리먼트", f"{len(best)}개",
        "대표 run: " + ("익스페리먼트마다 Average Precision 이 가장 높은 run" if rule == "Average Precision 최고" else "익스페리먼트마다 가장 최근 run"), MUTED)

    section("익스페리먼트별 핵심 지표",
            f"모두 val 기준입니다. Average Precision·AUROC 는 임계값과 상관없는 값이고, 재현율·정밀도·오검출·놓친 결함·오검출률은 "
            f"val 에서 재현율 {R98}% 이상이 되도록 정한 임계값으로 판정한 결과입니다.")
    table = best[["experiment"] + KEY_COLUMNS + ["run_name"]]
    config = key_metric_config()
    show_table(table, column_config=config, hide_index=True, use_container_width=True)

    section("한눈에 보기", "val Average Precision 과 val 정밀도입니다. 막대 끝 숫자가 값입니다.")
    chart_source = best.rename(columns={"ap": "Average Precision", "precision": "정밀도"})
    long = chart_source.melt(id_vars=["experiment", "run_name"], value_vars=CHART_METRICS,
                             var_name="지표", value_name="value").dropna(subset=["value"])
    order = best["experiment"].tolist()
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("experiment:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelLimit=260, labelFontSize=12, labelColor=INK)),
        yOffset=alt.YOffset("지표:N", sort=CHART_METRICS),
        x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 1.08]),
                axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], format=".1f", grid=True, gridColor=LINE)),
        color=alt.Color("지표:N", sort=CHART_METRICS,
                        scale=alt.Scale(domain=CHART_METRICS, range=[COLOR[m] for m in CHART_METRICS]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
        tooltip=["experiment", "run_name", "지표", alt.Tooltip("value:Q", format=".3f")])
    labels = bars.mark_text(align="left", dx=4, fontSize=11, color=MUTED).encode(
        text=alt.Text("value:Q", format=".3f"))
    st.altair_chart((bars + labels).properties(height=max(160, 78 * len(order)))
                    .configure_view(strokeWidth=0).configure_axis(domainColor=LINE, tickColor=LINE, labelColor=MUTED),
                    use_container_width=True)

# ----- run 비교 -----
if view == "run 비교":
    section("run 비교", f"시드 {FIXED_SEED} run 전체입니다. 칸 구성은 핵심 지표와 같습니다. "
                        "익스페리먼트를 누르면 그 안의 run 이 최신순으로 펼쳐지고, 기준 run 을 고르면 기준과의 차이가 붙습니다.")
    ordered = selected
    latest_first = ordered.sort_values("start_time", ascending=False)   # 고르는 목록은 최신순
    names = {row.run_id: f"{row.start_time:%m-%d %H:%M} · {row.experiment} · {row.run_name}"
             for row in latest_first.itertuples()}
    baseline_id = st.selectbox("기준 run", ["(없음)"] + list(names),
                               format_func=lambda r: r if r == "(없음)" else names[r], key="compare_base")
    columns = ["experiment", "run_name"] + KEY_COLUMNS
    shown = ordered[columns + ["status", "start_time"]].copy()
    layout = columns[1:]
    config = key_metric_config()
    if baseline_id != "(없음)":
        base = ordered[ordered["run_id"] == baseline_id].iloc[0]
        layout = ["run_name"]
        for key in columns[2:]:
            shown[f"{key}_delta"] = ordered[key] - base[key]
            config[f"{key}_delta"] = number("차이", "%+d" if key in ["fp", "fn"] else
                                            "%+.4f" if key == "threshold" else "%+.3f")
            layout += [key, f"{key}_delta"]
        st.caption(f"기준: {base['experiment']} · {base['run_name']} — Average Precision {fmt(base['ap'])} · "
                   f"AUROC {fmt(base['auroc'])} · 재현율 {fmt(base['recall'])} · 정밀도 {fmt(base['precision'])}. "
                   "오검출·놓친 결함·오검출률은 차이가 음수일수록 좋습니다.")
    grouped_runs(shown, ["start_time"] + layout + ["status"], config, "compare")
    with st.expander("익스페리먼트별로 팀 성능지표가 기록됐는지"):
        metric_keys = [k for k in KEY_COLUMNS if k in METRIC_RULES]
        sources = (selected.groupby("experiment")[[f"{k}_source" for k in metric_keys]]
                   .agg(lambda s: ", ".join(sorted({v for v in s if v})) or "없음"))
        sources.columns = [LABEL[k] for k in metric_keys]
        show_table(sources, use_container_width=True)
        st.caption("'없음'이면 그 익스페리먼트의 run 에 해당 val 지표가 기록되지 않은 것입니다.")

# ----- 지표 관계 -----
METRIC_NAME = {**LABEL, "threshold": "임계값"}
BETTER = {"ap": "높을수록 좋음", "auroc": "높을수록 좋음", "recall": "높을수록 좋음", "precision": "높을수록 좋음",
          "fp": "낮을수록 좋음", "fn": "낮을수록 좋음", "fpr": "낮을수록 좋음", "threshold": "좋고 나쁨 없음"}

if view == "지표 관계":
    data = selected[["experiment", "run_name"] + KEY_COLUMNS].copy()
    for key in KEY_COLUMNS:
        data[key] = pd.to_numeric(data[key], errors="coerce")

    section("성능지표 상관관계",
            "run 들 사이에서 두 지표가 같이 움직이는 정도입니다. +1 에 가까우면 한쪽이 오를 때 다른 쪽도 오르고(파랑), "
            "−1 에 가까우면 반대로 움직입니다(주황). 지표가 3개 이상 run 에 기록된 경우만 계산하고, 칸에 마우스를 올리면 "
            "계산에 쓴 run 수가 나옵니다.")
    method = st.segmented_control("계산 방법", ["스피어만 (순위)", "피어슨 (직선)"], default="스피어만 (순위)",
                                  key="corr_method",
                                  help="스피어만: 순위가 같이 움직이는지 봅니다. run 수가 적고 값이 한쪽에 몰릴 때 더 안정적입니다. "
                                       "피어슨: 값이 직선 관계인지 봅니다.") or "스피어만 (순위)"
    usable = [key for key in KEY_COLUMNS if data[key].notna().sum() >= 3 and data[key].nunique() > 1]
    constant = [METRIC_NAME[key] for key in KEY_COLUMNS
                if data[key].notna().sum() >= 3 and data[key].nunique() <= 1]
    if constant:   # 모든 run 이 같은 값이면 '같이 움직이는 정도'를 계산할 수 없음 (예: 임계값을 재현율 98% 에 맞춰 재현율이 모두 같을 때)
        st.caption(f"모든 run 에서 값이 같아 상관관계를 계산할 수 없는 지표: {', '.join(constant)}")
    if len(usable) < 2:
        st.info("상관관계를 계산하려면 같은 지표가 기록된 run 이 3개 이상 필요합니다.")
    else:
        corr = data[usable].corr(method="spearman" if method.startswith("스피어만") else "pearson", min_periods=3)
        present = data[usable].notna().astype(int)
        pair_counts = present.T @ present                        # 두 지표가 모두 기록된 run 수
        heat_data = pd.DataFrame([
            {"x": METRIC_NAME[a], "y": METRIC_NAME[b], "상관계수": corr.loc[a, b], "run 수": int(pair_counts.loc[a, b])}
            for a in usable for b in usable])
        heat_data["표시"] = heat_data["상관계수"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
        order = [METRIC_NAME[k] for k in usable]
        base = alt.Chart(heat_data).encode(
            x=alt.X("x:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelAlign="left", labelColor=INK,
                                                                orient="top", labelLimit=160)),
            y=alt.Y("y:N", sort=order, title=None, axis=alt.Axis(labelColor=INK)))
        cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("상관계수:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[DEFECT_COLOR, "#f4f1ea", "#2b5d78"]),
                            legend=alt.Legend(title="상관계수", orient="right")),
            tooltip=["x", "y", alt.Tooltip("상관계수:Q", format="+.3f"), "run 수"])
        labels = base.mark_text(fontSize=13, fontWeight="bold").encode(
            text="표시:N",
            color=alt.condition("abs(datum['상관계수']) > 0.6", alt.value("white"), alt.value(INK)))
        st.altair_chart((cells + labels).properties(height=alt.Step(52), width=alt.Step(110))
                        .configure_view(strokeWidth=0), use_container_width=False)
        st.caption("오검출·오검출률·정밀도처럼 같은 판정 결과에서 계산되는 지표는 원래 강하게 묶여 있습니다. "
                   "run 수가 적으면(10개 미만) 계수가 크게 흔들리니 참고용으로 보세요.")

    section("두 지표 비교", "가로축과 세로축 지표를 고르면 run 마다 점 하나로 찍습니다. 색은 익스페리먼트입니다. "
                        "두 지표 중 하나라도 기록되지 않은 run 은 빠집니다.")
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    x_key = c1.selectbox("가로축", KEY_COLUMNS, index=KEY_COLUMNS.index("ap"), key="pair_x",
                         format_func=lambda k: f"{METRIC_NAME[k]} ({BETTER[k]})")
    y_key = c2.selectbox("세로축", KEY_COLUMNS, index=KEY_COLUMNS.index("precision"), key="pair_y",
                         format_func=lambda k: f"{METRIC_NAME[k]} ({BETTER[k]})")
    show_names = c3.toggle("run 이름 표시", value=False, key="pair_names")
    points = data.dropna(subset=[x_key, y_key]).copy()
    if x_key == y_key:
        st.info("서로 다른 지표 두 개를 골라 주세요.")
    elif points.empty:
        st.info("두 지표가 모두 기록된 run 이 없습니다.")
    else:
        points["가로"], points["세로"] = points[x_key], points[y_key]
        axis_format = {k: (".0f" if k in ("fp", "fn") else ".4f" if k == "threshold" else ".3f") for k in KEY_COLUMNS}
        tooltip = ["experiment", "run_name"] + [alt.Tooltip(f"{k}:Q", title=METRIC_NAME[k], format=axis_format[k])
                                                for k in KEY_COLUMNS]
        scatter = alt.Chart(points).mark_circle(size=110, opacity=0.85, stroke="white", strokeWidth=1).encode(
            x=alt.X("가로:Q", title=f"{METRIC_NAME[x_key]} ({BETTER[x_key]})", scale=alt.Scale(zero=False, nice=True),
                    axis=alt.Axis(format=axis_format[x_key])),
            y=alt.Y("세로:Q", title=f"{METRIC_NAME[y_key]} ({BETTER[y_key]})", scale=alt.Scale(zero=False, nice=True),
                    axis=alt.Axis(format=axis_format[y_key])),
            color=alt.Color("experiment:N", title="익스페리먼트", scale=alt.Scale(scheme="tableau10"),
                            legend=alt.Legend(orient="right", labelLimit=220)),
            tooltip=tooltip)
        layers = scatter
        if show_names:
            layers = layers + scatter.mark_text(align="left", dx=8, fontSize=10.5, color=MUTED).encode(
                text="run_name:N", color=alt.value(MUTED))
        st.altair_chart(layers.properties(height=460).configure_view(strokeWidth=0)
                        .configure_axis(gridColor=LINE, domainColor=LINE, labelColor=MUTED, titleColor=INK,
                                        titleFontSize=12.5),
                        use_container_width=True)
        dropped = len(data) - len(points)
        pair_corr = (points[[x_key, y_key]].corr(method="spearman").iloc[0, 1] if len(points) >= 3 else np.nan)
        st.caption(f"run {len(points)}개" + (f" · 지표가 없어 빠진 run {dropped}개" if dropped else "")
                   + ("" if pd.isna(pair_corr) else f" · 스피어만 상관계수 {pair_corr:+.2f}")
                   + ". 점에 마우스를 올리면 그 run 의 지표 8개가 모두 나옵니다.")

# ----- run 상세 -----
if view == "run 상세":
    options = selected.sort_values("start_time", ascending=False)   # 최신순
    labels = {row.run_id: f"{row.start_time:%m-%d %H:%M} · {row.experiment} · {row.run_name}" for row in options.itertuples()}
    run_id = st.selectbox("run", list(labels), format_func=labels.get, key="detail_run")
    run = selected[selected["run_id"] == run_id].iloc[0]
    def blank(value, text):
        """값이 없으면 공란, 있으면 text (카드 아래 설명 줄용)."""
        return "" if pd.isna(value) else text

    def card_value(value, digits=3):
        return "&nbsp;" if pd.isna(value) else f"{value:.{digits}f}"   # 값이 없으면 공란 (카드 높이는 유지)

    boxes = st.columns(4)
    kpi(boxes[0], "Average Precision", card_value(run["ap"]), blank(run["auroc"], f"AUROC {fmt(run['auroc'])}"),
        COLOR["Average Precision"])
    kpi(boxes[1], "재현율", card_value(run["recall"]), blank(run["fn"], f"놓친 결함 {fmt(run['fn'], 0)}장"),
        COLOR["재현율"])
    precision_note = " · ".join(part for part in [blank(run["fp"], f"오검출 {fmt(run['fp'], 0)}장"),
                                                  blank(run["fpr"], f"오검출률 {run['fpr']:.1%}" if pd.notna(run["fpr"]) else "")]
                                if part)
    kpi(boxes[2], "정밀도", card_value(run["precision"]), precision_note, COLOR["정밀도"])
    kpi(boxes[3], "임계값", card_value(run["threshold"], 4), blank(run["threshold"], f"val 재현율 ≥{R98}% 로 정한 값"), INK)
    st.caption(f"{run['experiment']} · {run['run_name']} · 모든 값은 val 기준입니다. 기록되지 않은 지표는 공란입니다.")
    try:
        client = MlflowClient()
        artifacts = []
        for item in client.list_artifacts(run_id):        # 맨 위 + 한 단계 아래 폴더(val/, test/)까지
            artifacts += client.list_artifacts(run_id, item.path) if item.is_dir else [item]
        artifacts = [a for a in artifacts if not a.is_dir]
    except Exception as error:
        artifacts = []
        st.warning(f"artifact 목록을 읽지 못했습니다: {error}")
    images = [a.path for a in artifacts if a.path.endswith(".png")]
    texts = [a.path for a in artifacts if a.path.endswith("_hard_defects.txt")]
    error_lists = [a.path for a in artifacts if a.path.endswith("errors.csv")]
    if images or error_lists:
        with tempfile.TemporaryDirectory() as folder:
            for path in error_lists:        # 팀 baseline 의 틀린 이미지 목록 (<split>/errors.csv)
                local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=folder)
                errors = pd.read_csv(local, encoding="utf-8-sig")
                errors["error_type"] = errors["error_type"].map({"missed_defect": "놓친 결함", "false_alarm": "오검출"})
                section(f"틀린 이미지 ({path.split('/')[0]})",
                        "놓친 결함은 점수가 낮은 순, 오검출은 점수가 높은 순(= 가장 심하게 틀린 순)입니다.")
                show_table(errors, hide_index=True, use_container_width=True,
                             column_config={"error_type": "구분", "file": "파일", "label": "라벨",
                                            "score": number("점수", "%.4f"), "threshold": number("임계값", "%.4f"),
                                            "score_over_threshold": number("점수/임계값", "%.3f×")},
                             height=min(420, 36 * (len(errors) + 1) + 4))
            if images:
                section("그림", "히트맵이 먼저 나옵니다.")
            for path in sorted(images, key=lambda p: ("heatmap" not in p, p)):
                local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=folder)
                st.image(local, caption=path, use_container_width=True)
            for path in texts:
                local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=folder)
                with open(local, encoding="utf-8") as f:
                    with st.expander(path):
                        st.text(f.read())
    else:
        st.info("이 run 에는 그림이나 틀린 이미지 목록 artifact 가 없습니다.")

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
    threshold_mode = c3.radio("임계값", ["학습 때 값", f"재현율 ≥{R98}% 지점"], key="infer_mode",
                              help=f"학습 때 값: run 에 기록된 threshold (val 재현율 ≥{R98}%). "
                                   f"재현율 ≥{R98}% 지점: 보고 있는 데이터에서 재현율이 {R98}% 이상이 되는 가장 높은 임계값")
    split = st.segmented_control("데이터", ["val", "test"], default="val", key="infer_split") or "val"

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
    has_split = not scores_a.empty and (scores_a["split"] == split).any()
    threshold_a = threshold_of(info_a, scores_a) if has_split else np.nan
    if scores_a.empty:
        st.info("이 run 에는 이미지별 점수 파일(val/scores.csv 또는 *_scores.csv)이 없습니다.")
    elif not has_split:
        st.info(f"이 run 에는 {split} 점수가 없습니다. 팀 baseline 은 test 를 최종 확정 때만 기록합니다.")
    elif pd.isna(threshold_a):
        st.info("이 run 에는 임계값 기록이 없습니다.")
    else:
        judged_a = judge(scores_a[scores_a["split"] == split], threshold_a)
        counts = judged_a["결과"].value_counts()
        boxes = st.columns(4)
        for box, (name, color) in zip(boxes, [("결함 검출", COLOR["재현율"]), ("놓침", DEFECT_COLOR),
                                              ("오검출", COLOR["정밀도"]), ("정상 통과", MUTED)]):
            kpi(box, name, f"{int(counts.get(name, 0))}장", "", color)
        defects = judged_a[judged_a["label"] == 1]
        flagged = judged_a[judged_a["판정"] == "불량"]
        st.caption(f"임계값 {threshold_a:.4f} · 재현율 {(defects['판정'] == '불량').mean():.3f} · "
                   f"정밀도 {(flagged['label'] == 1).mean() if len(flagged) else 0:.3f} ({split} 기준)")

        section("이상 점수 분포", "점선이 임계값입니다. 정상(파랑)이 점선 오른쪽이면 오검출, 결함(주황)이 왼쪽이면 놓친 결함입니다.")
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
            section("오검출", "점수가 높은 순서입니다.")
            show_table(judged_a[judged_a["결과"] == "오검출"].sort_values("score", ascending=False)[wrong_columns],
                         column_config=wrong_config, hide_index=True, use_container_width=True)
        with col_fn:
            section("놓친 결함과 아슬아슬한 결함", "임계값의 1.15배 미만인 결함까지 보여 줍니다.")
            show_table(defects[defects["임계값 대비"] < 1.15].sort_values("score")[wrong_columns],
                         column_config=wrong_config, hide_index=True, use_container_width=True)

        if run_b != "(비교 안 함)":
            info_b = selected[selected["run_id"] == run_b].iloc[0]
            scores_b = load_scores(run_b)
            has_split_b = not scores_b.empty and (scores_b["split"] == split).any()
            threshold_b = threshold_of(info_b, scores_b) if has_split_b else np.nan
            if not has_split_b or pd.isna(threshold_b):
                st.info(f"비교할 run 에 {split} 점수 파일 또는 임계값 기록이 없습니다.")
            elif (pd.notna(info_a["split_hash"]) and pd.notna(info_b["split_hash"])
                  and info_a["split_hash"] != info_b["split_hash"]):
                st.warning("두 run 의 분할(split_hash)이 달라 같은 이미지끼리 비교할 수 없습니다.")
            else:
                judged_b = judge(scores_b[scores_b["split"] == split], threshold_b)
                merged = judged_a.merge(judged_b, on=["file", "label"], suffixes=(" A", " B"))
                merged["변화"] = np.select(
                    [(merged["결과 A"].isin(["놓침", "오검출"])) & (merged["결과 B"].isin(["결함 검출", "정상 통과"])),
                     (merged["결과 A"].isin(["결함 검출", "정상 통과"])) & (merged["결과 B"].isin(["놓침", "오검출"]))],
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
                show_table(changed[["file", "구분", "변화", "결과 A", "결과 B", "임계값 대비 A", "임계값 대비 B"]],
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
            "같은 분할(split_hash)로 돌린 run 끼리만 모읍니다.")
    h1, h2, h3 = st.columns([1.3, 1.3, 1])
    hard_scope = h1.segmented_control("모을 run", ["익스페리먼트 대표 run", "모든 run"], default="익스페리먼트 대표 run",
                                      key="hard_scope",
                                      help="대표 run: 익스페리먼트마다 AP 가 가장 높은 run 1개. run 을 많이 돌린 익스페리먼트가 결과를 끌고 가지 않게 합니다.")
    hard_scope = hard_scope or "익스페리먼트 대표 run"
    hard_mode = h2.segmented_control("임계값", ["학습 때 값", f"재현율 ≥{R98}% 지점"], default="학습 때 값",
                                     key="hard_mode") or "학습 때 값"
    hard_split = h3.segmented_control("데이터", ["val", "test"], default="val", key="hard_split") or "val"

    pool = best_per_experiment(selected, "Average Precision 최고") if hard_scope == "익스페리먼트 대표 run" else selected
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
        st.info(f"{hard_split} 점수 파일이 있는 run 이 없습니다. 팀 baseline 은 val/scores.csv 를 남기고, "
                "test 점수는 최종 확정 때만 남깁니다.")
    else:
        everything = pd.concat(collected, ignore_index=True)
        main_hash = everything.groupby("split_hash")["run"].nunique().idxmax()
        dropped = everything.loc[everything["split_hash"] != main_hash, "run"].nunique()
        everything = everything[everything["split_hash"] == main_hash]
        n_runs = everything["run"].nunique()
        everything["틀림"] = everything["결과"].isin(["놓침", "오검출"])

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
        kpi(boxes[1], "모든 run 에서 오검출", f"{always_alarm}장", f"한 번이라도 오검출 {once_alarm}장", COLOR["정밀도"])
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
            show_table(table_d[["file", "틀린_run", "전체_run", "틀린 비율", "평균_임계값대비", "최소_임계값대비",
                                  "mask_area", "크기", "틀린 run"]],
                         column_config=hard_config, hide_index=True, use_container_width=True,
                         height=min(420, 36 * (len(table_d) + 1) + 4))
        with col_n:
            section("오검출", "오검출을 많이 낸 순서, 같으면 점수가 높은 순서입니다.")
            table_n = normals if show_all else normals[normals["틀린_run"] > 0]
            table_n = table_n.sort_values(["틀린_run", "평균_임계값대비"], ascending=[False, False])
            show_table(table_n[["file", "틀린_run", "전체_run", "틀린 비율", "평균_임계값대비", "최대_임계값대비", "틀린 run"]],
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
