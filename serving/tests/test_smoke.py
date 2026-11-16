"""PR마다 실행하는 가벼운 스모크 테스트입니다.

모델 로딩이나 MLflow 접속처럼 무거운 검사는 하지 않고,
병합하면 서버에서 바로 문제가 되는 실수만 빠르게 잡습니다.
표준 라이브러리만 사용하므로 ci.yaml에서 pytest만 설치해도 실행됩니다.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
MAX_FILE_MB = 5
# KolektorSDD2 원본 파일 이름 형식 (예: 10001.png, 10001_GT.png) - 라이선스상 저장소에 올리면 안 됨
KSDD2_NAME = re.compile(r"(^|/)\d{5}(_GT)?\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
SECRET_NAME = re.compile(r"(^|/)(\.env(\..+)?|.*\.pem|.*\.key|id_rsa.*|id_ed25519.*)$")
ALLOWED_ENV_FILES = {".env.example", ".env.sample", ".env.template"}


def _walk(pattern):
    for path in ROOT.rglob(pattern):
        if not SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            yield path


def _tracked_files():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("git 저장소가 아니라 추적 파일 검사를 건너뜁니다")
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def test_python_files_compile():
    """모든 파이썬 파일에 문법 오류가 없는지 확인합니다."""
    errors = []
    for path in _walk("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not errors, "문법 오류가 있는 파이썬 파일:\n" + "\n".join(errors)


def test_yaml_files_have_no_tabs():
    """워크플로·compose YAML에 탭 문자가 없는지 확인합니다 (YAML은 탭 들여쓰기를 허용하지 않음)."""
    bad = []
    for pattern in ("*.yaml", "*.yml"):
        for path in _walk(pattern):
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                bad.append(f"{path.relative_to(ROOT)}: 빈 파일")
            elif "\t" in text:
                line = next(i for i, l in enumerate(text.splitlines(), 1) if "\t" in l)
                bad.append(f"{path.relative_to(ROOT)}:{line}: 탭 문자")
    assert not bad, "YAML 문제:\n" + "\n".join(bad)


def test_shell_scripts_syntax():
    """배포 스크립트 등 셸 스크립트에 문법 오류가 없는지 확인합니다 (bash -n)."""
    if shutil.which("bash") is None:
        pytest.skip("bash가 없어 셸 스크립트 검사를 건너뜁니다")
    errors = []
    for path in _walk("*.sh"):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append(f"{path.relative_to(ROOT)}: {result.stderr.strip()}")
    assert not errors, "문법 오류가 있는 셸 스크립트:\n" + "\n".join(errors)


def test_no_secrets_tracked():
    """.env, 키 파일 같은 비밀값 파일이 Git에 올라가지 않았는지 확인합니다."""
    found = [
        f for f in _tracked_files()
        if SECRET_NAME.search(f) and Path(f).name not in ALLOWED_ENV_FILES
    ]
    assert not found, "Git에 올라간 비밀값 파일 (git rm --cached 후 키 교체 필요):\n" + "\n".join(found)


def test_no_dataset_or_large_files_tracked():
    """KolektorSDD2 원본 이미지나 너무 큰 파일(모델·데이터)이 올라가지 않았는지 확인합니다."""
    problems = []
    for f in _tracked_files():
        path = ROOT / f
        if KSDD2_NAME.search(f):
            problems.append(f"{f}: KolektorSDD2 원본 이미지로 보임 (공개 금지)")
        elif path.is_file() and path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            size_mb = path.stat().st_size / 1024 / 1024
            problems.append(f"{f}: {size_mb:.1f}MB (기준 {MAX_FILE_MB}MB 초과)")
    assert not problems, "올리면 안 되는 파일:\n" + "\n".join(problems)
