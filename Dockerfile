FROM python:3.11-slim

ARG REQUIREMENTS_FILE=requirements-docker.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

COPY requirements-docker.txt /workspace/requirements-docker.txt
COPY requirements-docker-gpu.txt /workspace/requirements-docker-gpu.txt
RUN pip install --upgrade pip && pip install -r /workspace/${REQUIREMENTS_FILE}

COPY halfweg /workspace/halfweg
COPY scripts /workspace/scripts
COPY README.md /workspace/README.md

CMD ["python", "scripts/train_halfweg.py", "--help"]
