# Container A - Unsloth QLoRA trainer (GPU 0), launched one-shot by the
# lifecycle orchestrator only after the vLLM student is stopped.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf_cache
RUN pip install --no-cache-dir "unsloth" "unsloth_zoo" "trl" "datasets" "peft" "transformers" "bitsandbytes" "accelerate"
COPY backend/systemone/training/trainer_entry.py /opt/s1/trainer_entry.py
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/s1/trainer_entry.py"]
