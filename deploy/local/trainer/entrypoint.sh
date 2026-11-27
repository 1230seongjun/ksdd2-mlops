#!/bin/sh
# 학습 서버 시작 전에 학습 데이터를 어디서 읽을지 정합니다 (원본 config.yaml · common.yaml 은 저장소에 그대로 두고 컨테이너 안에서만 바꿈).
#   DATASET_SOURCE=object_storage : OBJECT_STORAGE_ENDPOINT 의 dataset 버킷에서 내려받아 /data 에 캐시한 뒤 학습
#                                   (로컬 실행에서는 같은 Pod 의 MinIO, http://127.0.0.1:9000)
#   DATASET_SOURCE=local          : /data 볼륨에 직접 넣은 데이터를 그대로 읽음 (기본값)
set -eu
SOURCE="${DATASET_SOURCE:-local}"
case "$SOURCE" in
  local|object_storage) ;;
  *) echo "DATASET_SOURCE 는 local 또는 object_storage 여야 합니다: $SOURCE" >&2; exit 1 ;;
esac
if [ "$SOURCE" = object_storage ] && { [ -z "${OBJECT_STORAGE_ACCESS_KEY:-}" ] || [ -z "${OBJECT_STORAGE_SECRET_KEY:-}" ]; }; then
  echo "DATASET_SOURCE=object_storage 인데 OBJECT_STORAGE_ACCESS_KEY / OBJECT_STORAGE_SECRET_KEY 가 없습니다" >&2
  exit 1
fi
for CONFIG in "/app/model/이상치 모델/config.yaml" "/app/model/분류 모델/config.yaml"; do
  sed -i "s#^DATA_SOURCE:.*#DATA_SOURCE: $SOURCE#" "$CONFIG"
done
if [ -n "${OBJECT_STORAGE_ENDPOINT:-}" ]; then
  sed -i "s#^OBJECT_STORAGE_ENDPOINT:.*#OBJECT_STORAGE_ENDPOINT: '$OBJECT_STORAGE_ENDPOINT'#" /app/model/common.yaml
fi
echo "학습 데이터: $SOURCE ${OBJECT_STORAGE_ENDPOINT:-}"
exec "$@"
