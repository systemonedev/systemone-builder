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

from pydantic import AliasChoices, Field
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
    # Address the API/dashboard ports are published on (set by compose). On a
    # non-loopback bind the API refuses to start without a real api_key.
    bind_addr: str = "127.0.0.1"
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
    # Alternative flush criterion when the GPU is shared with a display or,
    # on WSL2, with Windows: enough free VRAM for the trainer and for vLLM's
    # --gpu-memory-utilization (0.85 x 24 GB = 20.9 GB).
    vram_required_free_mb: int = 21000
    # How long a (re)started vLLM student may take to load weights and pass
    # its health check before the lifecycle rolls back.
    student_start_timeout_s: float = 1800.0

    # ------------------------------------------------------ docker lifecycle
    docker_project: str = "systemone"
    trainer_container: str = "systemone-trainer"
    student_container: str = "systemone-student"
    triage_container: str = "systemone-triage"
    trainer_image: str = "systemone/trainer:latest"
    # Host-memory bounds for the one-shot trainer container.
    trainer_mem_limit: str = "48g"
    trainer_shm_size: str = "8g"
    # Host path of the shared workspace volume (models, datasets, state).
    workspace_host_path: str = "./workspace"
    workspace_container_path: str = "/workspace"

    # --------------------------------------------------------- BYOM models
    student_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student_url: str = "http://localhost:8001/v1"
    student_served_name: str = "student"
    triage_url: str = "http://localhost:8002/v1"
    triage_model: str = "Qwen/Qwen2.5-7B-Instruct"
    oracle_url: str = "http://mac-m4.local:11434"
    oracle_model: str = "qwen3.8:27b"
    oracle_vision_model: str | None = None
    # Adapter kinds: "openai" (vLLM / any OpenAI-compatible) or "ollama"
    # (plus any registered through the ``systemone_builder.adapters`` entry points).
    # Optional YAML/JSON file overriding the per-role endpoints (see adapters/byom.py).
    byom_file: str | None = None
    student_adapter: str = "openai"
    triage_adapter: str = "openai"
    oracle_adapter: str = "ollama"
    request_timeout_s: float = 30.0
    oracle_timeout_s: float = 600.0

    # ------------------------------------------- System One (Jev-compatible)
    # Local engine for POST /api/v1/systemone: reads each answer from one
    # forward pass of an OpenAI-compatible model. Unset = the triage server.
    system_one_local_url: str | None = None
    system_one_local_model: str | None = None
    system_one_concurrency: int = 16
    # Kenning, the local System One model (compose service ``kenning``).
    kenning_url: str = "http://kenning:8000"
    # Optional Cloudflare Clef server (compose profile "clef"): benchmark engine and teacher.
    clef_url: str = "http://clef:8000"
    # Kenning's files: models/, datasets/, exports/, bench_cache/ and active.json.
    # Unset = <data_dir>/workspace/kenning (the shared workspace volume in compose).
    kenning_home: Path | None = None
    # What answers POST /api/v1/systemone: "kenning" (the Kenning server) or
    # "logprob" (the label-token readout of a local LLM above).
    system_one_backend: str = "kenning"
    # Kenning jobs (Train / Verify pages): containers and GPUs they pause and use.
    kenning_gpu: int = 0
    kenning_cuda_device: str | None = None  # native Linux: "0" (see the compose file's GPU note)
    clef_gpu: int = 0
    kenning_container: str = "systemone-kenning"
    clef_container: str = "systemone-clef"
    kenning_image: str = "ghcr.io/systemonedev/systemone-kenning:main"  # compose sets it from S1_IMAGE_TAG
    kenning_train_container: str = "systemone-kenning-train"
    kenning_train_mem_limit: str = "32g"
    # Free VRAM (MiB) a training run needs once the GPU's services are paused
    # (DeBERTa-v3-large at --max-length 512 peaks at ~18.1 GiB).
    kenning_train_free_mb: int = 19500

    def kenning_dir(self) -> Path:
        return self.kenning_home or self.data_dir / "workspace" / "kenning"
    # TypeSafe Jev, the reference System One model we benchmark against. Its
    # key is TYPESAFE_API_KEY (the name TypeSafe's SDKs use, no S1_ prefix);
    # S1_API_KEY only protects this project's own API.
    typesafe_url: str = "https://api.typesafe.ai"
    typesafe_model: str = "jev-latest"
    typesafe_api_key: str | None = Field(default=None, validation_alias=AliasChoices("TYPESAFE_API_KEY", "S1_TYPESAFE_API_KEY"))
    # Fastino GLiDE, a hosted System One model speaking the same wire format
    # (model "fastino/GLiDE"). Benchmark engine only, never trained on. Key name
    # is FASTINO_API_KEY (what Fastino's docs use, no S1_ prefix).
    fastino_url: str = "https://api.fastino.ai"
    fastino_model: str = "fastino/GLiDE"
    fastino_api_key: str | None = Field(default=None, validation_alias=AliasChoices("FASTINO_API_KEY", "S1_FASTINO_API_KEY"))

    # -------------------------------------------------------------- routing
    default_threshold: float = 0.80
    oracle_concurrency: int = 1
    student_max_tokens: int = 192
    # Blend of self-reported confidence vs. token-probability confidence.
    confidence_logprob_weight: float = 0.5

    # --------------------------------------------------------- training
    # The generative distillation pipeline (vLLM student + triage, LoRA training,
    # data factory; compose profile "pipeline"). Off: the API never starts the
    # student or trains on its own, and the stack is Kenning-only (one GPU).
    pipeline: bool = False
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
