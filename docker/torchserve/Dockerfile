FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/home/model-server/src

# Perform package installation in a single step. Apt's built-in retry options handle transient network failures.
RUN set -eux; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch and TorchServe along with supporting Python libraries.
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

# Create a dedicated user and prepare directories for TorchServe.
RUN useradd --create-home --shell /bin/bash model-server && \
    mkdir -p /home/model-server/model-store /home/model-server/config && \
    chown -R model-server:model-server /home/model-server

# Copy application source and TorchServe config.
COPY src /home/model-server/src
COPY docker/torchserve/config.properties /home/model-server/config/config.properties

# Switch to non-root user and set working directory.
USER model-server
WORKDIR /home/model-server

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--foreground", "--ts-config", "/home/model-server/config/config.properties", "--model-store", "/home/model-server/model-store"]
