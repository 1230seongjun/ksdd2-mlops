import os
from pathlib import Path

from config import Config
from data import IMAGE_SUFFIXES


def make_client(config: Config):
    access_key = os.environ.get("OBJECT_STORAGE_ACCESS_KEY")
    secret_key = os.environ.get("OBJECT_STORAGE_SECRET_KEY")
    if not access_key or not secret_key:
        raise EnvironmentError("환경변수 OBJECT_STORAGE_ACCESS_KEY, OBJECT_STORAGE_SECRET_KEY 가 필요합니다 (.env 에 넣으세요)")
    import boto3   # 로컬 데이터만 쓸 때는 boto3 없이도 학습이 돌도록 여기서 불러온다
    return boto3.client(
        "s3", endpoint_url=config.object_storage_endpoint, region_name=config.object_storage_region or None,
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
    )


def download_dataset(config: Config) -> None:
    """버킷의 normal/anomaly 폴더 이미지를 cache_dir 로 내려받는다. 이미 있고 크기가 같은 파일은 건너뛴다."""
    client = make_client(config)
    paginator = client.get_paginator("list_objects_v2")
    for folder_name in (config.normal_dir_name, config.anomaly_dir_name):
        prefix = "/".join(part for part in (config.object_storage_prefix.strip("/"), folder_name) if part) + "/"
        downloaded = skipped = 0
        for page in paginator.paginate(Bucket=config.object_storage_bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if Path(key).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                target = config.dataset_dir / folder_name / Path(key).name
                if target.exists() and target.stat().st_size == item["Size"]:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(config.object_storage_bucket, key, str(target))
                downloaded += 1
        if downloaded + skipped == 0:
            raise FileNotFoundError(f"버킷 {config.object_storage_bucket} 의 {prefix} 에서 이미지를 찾지 못했습니다")
        print(f"[{folder_name}] 새로 받음 {downloaded}장, 이미 있어서 건너뜀 {skipped}장")