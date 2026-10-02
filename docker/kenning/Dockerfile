# Kenning, the local System One model: a non-generative cross-encoder served with
# TypeSafe Jev's wire format (POST /v1/systemone). Also runs training:
#   docker compose run --rm --no-deps kenning python -m systemone_builder.kenning.train ...
# Same base as the trainer image, so the CUDA/PyTorch layers are shared.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf_cache CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN pip install --no-cache-dir "transformers>=4.48,<6" safetensors \
      "fastapi>=0.110" "uvicorn>=0.27" "pydantic>=2.6" "pydantic-settings>=2.2" "httpx>=0.27"
# Only the pure-Python parts of the package are used here (contract, s1model).
COPY backend /opt/kenning/backend
RUN pip install --no-cache-dir --no-deps /opt/kenning/backend
WORKDIR /workspace
EXPOSE 8000
CMD ["python", "-m", "systemone_builder.kenning.serve"]
