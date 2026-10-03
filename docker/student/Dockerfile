# Container B / C - vLLM student runner (GPU 0) and triage server (GPU 1).
# Pin the base image by setting S1_VLLM_VERSION in .env to a tag you have
# validated (e.g. v0.x.y); compose passes it as VLLM_IMAGE. The launcher
# already tolerates CLI drift, but a pinned tag keeps builds reproducible.
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}
COPY docker/student/serve_student.py /opt/s1/serve_student.py
ENTRYPOINT ["python3", "/opt/s1/serve_student.py"]
