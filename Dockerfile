# CUDA 12.8 matches the pinned PyTorch build used for the recorded run.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements-mlsec.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

WORKDIR /workspace
COPY src/ /workspace/src/
COPY config/ /workspace/config/

CMD ["python", "-m", "src.run_submission"]
