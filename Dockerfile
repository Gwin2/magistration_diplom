FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN set -eux; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    for i in 1 2 3; do \
      apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o Acquire::ForceIPv4=true update && \
      apt-get install -y --no-install-recommends --fix-missing \
      ffmpeg \
      libgl1 && \
      break; \
      echo "apt attempt $i failed, retrying..."; \
      rm -rf /var/lib/apt/lists/*; \
      sleep 5; \
    done; \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY docs /app/docs

RUN pip install --upgrade pip && \
    for i in 1 2 3; do \
      if pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu; then \
        break; \
      fi; \
      if [ "$i" -eq 3 ]; then \
        exit 1; \
      fi; \
      echo "pip torch attempt $i failed, retrying..."; \
      sleep 5; \
    done && \
    pip install --no-cache-dir -e .

CMD ["uav-vit", "--help"]
