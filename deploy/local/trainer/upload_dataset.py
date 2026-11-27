"""정리된 학습 데이터를 Object Storage 의 dataset 버킷에 올립니다 (운영 때 팀이 NHN Object Storage 에 올리던 것과 같은 구조).

  <폴더>/normal/*.png   → s3://dataset/normal/
  <폴더>/anomaly/*.png  → s3://dataset/anomaly/

버킷이 없으면 만들고, 이름과 크기가 같은 파일은 건너뜁니다. 학습 컨테이너 안에서 실행하면
주소와 키(OBJECT_STORAGE_ENDPOINT, OBJECT_STORAGE_ACCESS_KEY, OBJECT_STORAGE_SECRET_KEY)를 그대로 씁니다.

  podman cp ./KolektorSDD2 ksdd2-trainer:/tmp/upload
  podman exec ksdd2-trainer python /app/upload_dataset.py /tmp/upload
"""
import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path, help="normal/ 과 anomaly/ 가 들어 있는 폴더")
parser.add_argument("--bucket", default="dataset")
parser.add_argument("--delete-extra", action="store_true", help="폴더에 없는 버킷 파일은 지움 (버킷을 폴더와 똑같이 맞춤)")
args = parser.parse_args()

client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("OBJECT_STORAGE_ENDPOINT", "http://127.0.0.1:9000"),
    region_name="KR1",
    aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
    aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],
)

try:
    client.head_bucket(Bucket=args.bucket)
except ClientError:
    client.create_bucket(Bucket=args.bucket, CreateBucketConfiguration={"LocationConstraint": "KR1"})
    print(f"버킷 {args.bucket} 생성")

for folder_name in ("normal", "anomaly"):
    folder = args.folder / folder_name
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []
    if not files:
        sys.exit(f"{folder} 에 이미지가 없습니다 (normal/ 과 anomaly/ 폴더가 필요합니다)")
    existing = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=f"{folder_name}/"):
        existing.update({item["Key"]: item["Size"] for item in page.get("Contents", [])})
    uploaded = skipped = 0
    for path in files:
        key = f"{folder_name}/{path.name}"
        if existing.pop(key, None) == path.stat().st_size:
            skipped += 1
            continue
        client.upload_file(str(path), args.bucket, key)
        uploaded += 1
    if args.delete_extra:
        for key in existing:
            client.delete_object(Bucket=args.bucket, Key=key)
    extra = f", 폴더에 없는 파일 {'삭제' if args.delete_extra else '남김'} {len(existing)}개" if existing else ""
    print(f"[{folder_name}] 올림 {uploaded}장, 이미 있어서 건너뜀 {skipped}장{extra}")
