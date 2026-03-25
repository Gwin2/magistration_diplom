FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/home/model-server/src

RUN set -eux; \
    for i in 1 2 3 4; do \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        -o Acquire::Retries=5 \
        openjdk-21-jre-headless \
        curl \
        ca-certificates && \
      break; \
      if [ "$i" -eq 4 ]; then exit 1; fi; \
      rm -rf /var/lib/apt/lists/*; \
      sleep 8; \
    done; \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1+cpu \
      torchvision==0.20.1+cpu && \
    pip install \
      torchserve==0.12.0 \
      torch-model-archiver==0.12.0 \
      "transformers>=4.46" \
      "timm>=1.0" \
      "pyyaml>=6.0.1" \
      "Pillow>=10.3"

RUN useradd --create-home --shell /bin/bash model-server && \
    mkdir -p /home/model-server/model-store /home/model-server/config && \
    chown -R model-server:model-server /home/model-server

COPY src /home/model-server/src
COPY docker/torchserve/config.properties /home/model-server/config/config.properties

USER model-server
WORKDIR /home/model-server

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--foreground", "--ts-config", "/home/model-server/config/config.properties", "--model-store", "/home/model-server/model-store"]
