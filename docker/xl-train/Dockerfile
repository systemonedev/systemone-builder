# Kenning-XL trainer: the kenning image plus QLoRA/LoRA dependencies.
ARG KENNING_IMAGE=ghcr.io/systemonedev/systemone-kenning:main
FROM ${KENNING_IMAGE}
RUN pip install --no-cache-dir "peft>=0.14" "bitsandbytes>=0.46" "datasets>=3.0"
