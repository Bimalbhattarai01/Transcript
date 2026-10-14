# Podcast SRT: web UI + queue worker. CPU only (sherpa-onnx speaker labels); no torch, no HuggingFace token.
FROM node:20-slim AS node

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/data NO_BROWSER=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=node /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
COPY requirements-vps.txt .
RUN pip install -r requirements-vps.txt
COPY *.py ./
COPY static ./static
COPY docker/entrypoint.sh /entrypoint.sh
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data && chmod +x /entrypoint.sh
USER app
VOLUME /data
EXPOSE 8765
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "ui.py"]
