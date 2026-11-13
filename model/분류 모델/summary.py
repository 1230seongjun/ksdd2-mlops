# 5 fold 결과 모아보기. 실행: python summary.py   /  노트북: !python summary.py
import argparse

from config import load_config
from evaluation import summarize_folds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    summarize_folds(load_config(parser.parse_args().config))