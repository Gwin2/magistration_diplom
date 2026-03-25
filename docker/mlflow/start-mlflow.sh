#!/usr/bin/env sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT is required}"

PORT="${MLFLOW_PORT:-5000}"

python - <<'PY'
import os
import time
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

artifact_root = os.environ["MLFLOW_ARTIFACT_ROOT"]
endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
access_key = os.environ.get("AWS_ACCESS_KEY_ID")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
bucket_name = urlparse(artifact_root).netloc or artifact_root.replace("s3://", "", 1).split("/", 1)[0]

if endpoint_url and bucket_name:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    for attempt in range(30):
        try:
            client.head_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} already exists.")
            break
        except EndpointConnectionError:
            time.sleep(2)
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchBucket", "NotFound"}:
                client.create_bucket(Bucket=bucket_name)
                print(f"Bucket {bucket_name} created.")
                break
            if error_code in {"301", "403"}:
                print(f"Bucket {bucket_name} is reachable.")
                break
            raise
    else:
        raise RuntimeError(f"MinIO bucket {bucket_name} is not reachable at {endpoint_url}.")
PY

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port "${PORT}"
