"""Central configuration for systemone-builder.

Every value can be overridden with an environment variable prefixed ``S1_``
(e.g. ``S1_REDIS_URL``) or through a ``.env`` file. The defaults describe the
reference topology from the specification:

* Linux host: GPU 0 = Student (Unsloth trainer + vLLM student runner),
  GPU 1 = intermediate synchronous triage vLLM server, 120GB RAM for Redis.
* Mac M4 Max: Ollama asynchronous oracle (``qwen3.8:27b`` by default).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="S1_", env_file=".env", extra="ignore")

    # ------------------------------------------------------------------ runtime
    data_dir: Path = Path("./var/systemone")

    # ---------------------------------------------------------------- LAN API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Optional shared secret for LAN clients (sent as ``X-API-Key``).
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------ RAM datastore
    redis_url: str = "redis://localhost:6379/0"
    redis_namespace: str = "s1"
    replay_capacity: int = 10_000

    # ------------------------------------------------------------ hardware
    student_gpu: int = 0
    triage_gpu: int = 1
    # VRAM (MiB) below which GPU 0 is considered flushed.
    vram_flush_threshold_mb: int = 1024
    vram_flush_timeout_s: float = 120.0

    # ------------------------------------------------------ docker lifecycle
    docker_project: str = "systemone"
    trainer_container: str = "systemone-trainer"
    student_container: str = "systemone-student"
    triage_container: str = "systemone-triage"
    trainer_image: str = "systemone/trainer:latest"
    # Host path of the shared workspace volume (models, datasets, state).
    workspace_host_path: str = "./workspace"
    workspace_container_path: str = "/workspace"

    # --------------------------------------------------------- BYOM models
    student_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student_url: str = "http://localhost:8001/v1"
    student_served_name: str = "student"
    triage_url: str = "http://localhost:8002/v1"
    triage_model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    oracle_url: str = "http://mac-m4.local:11434"
    oracle_model: str = "qwen3.8:27b"
    oracle_vision_model: str | None = None
    # Adapter kinds: "openai" (vLLM / any OpenAI-compatible) or "ollama"
    # (plus any registered through the ``systemone.adapters`` entry points).
    # Optional YAML/JSON file overriding the per-role endpoints (see adapters/byom.py).
    byom_file: str | None = None
    student_adapter: str = "openai"
    triage_adapter: str = "openai"
    oracle_adapter: str = "ollama"
    request_timeout_s: float = 30.0
    oracle_timeout_s: float = 600.0

    # -------------------------------------------------------------- routing
    default_threshold: float = 0.80
    oracle_concurrency: int = 1
    student_max_tokens: int = 192
    # Blend of self-reported confidence vs. token-probability confidence.
    confidence_logprob_weight: float = 0.5

    # --------------------------------------------------------- training
    lora_rank: int = 16
    lora_alpha: int = 32
    train_epochs: int = 1
    train_batch_size: int = 4
    train_learning_rate: float = 2e-4
    train_max_seq_length: int = 4096
    # Number of new SFT samples that triggers an automatic training cycle.
    auto_train_min_samples: int = 256
    auto_train: bool = True
    auto_factory: bool = True
    # "adapter": vLLM loads the LoRA at runtime; "merged": vLLM restarts on
    # the merged 16-bit weights.
    reload_mode: str = "merged"

    # ------------------------------------------------------ synthetic factory
    factory_batch_size: int = 8
    factory_poll_interval_s: float = 5.0
    factory_judge: bool = True

    telemetry_interval_s: float = 1.0

    # ------------------------------------------------------- DPO loop
    # Judge score at/above which teacher corrections skip human review
    # (set to a value > 1 to require the Corrections Studio for everything).
    dpo_auto_approve_min_judge: float = 0.9
    dpo_auto_train_min_pairs: int = 64

    @property
    def workspace(self) -> Path:
        return self.data_dir / "workspace"

    def ensure_dirs(self) -> None:
        for sub in ("workspace/datasets", "workspace/models", "workspace/state", "eval_results", "domains"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
