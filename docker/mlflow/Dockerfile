FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mlflow==2.20.3 \
    boto3==1.37.7 \
    psycopg2-binary==2.9.10

COPY docker/mlflow/start-mlflow.sh /usr/local/bin/start-mlflow.sh
RUN chmod +x /usr/local/bin/start-mlflow.sh

EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/start-mlflow.sh"]
