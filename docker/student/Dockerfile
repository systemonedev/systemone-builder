# Container B - vLLM student runner (GPU 0)
FROM vllm/vllm-openai:latest
COPY docker/student/serve_student.py /opt/s1/serve_student.py
ENTRYPOINT ["python3", "/opt/s1/serve_student.py"]
