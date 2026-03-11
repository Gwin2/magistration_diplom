FROM pytorch/torchserve:0.11.1-cpu

USER root

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "transformers>=4.46" \
    "timm>=1.0" \
    "pyyaml>=6.0.1" \
    "Pillow>=10.3"

COPY pyproject.toml README.md /home/model-server/
COPY src /home/model-server/src
RUN pip install --no-cache-dir -e /home/model-server

RUN mkdir -p /home/model-server/model-store /home/model-server/config
COPY docker/torchserve/config.properties /home/model-server/config/config.properties

USER model-server

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--foreground", "--ts-config", "/home/model-server/config/config.properties", "--model-store", "/home/model-server/model-store"]
