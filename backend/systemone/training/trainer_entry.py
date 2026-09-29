"""Unsloth QLoRA trainer - entrypoint of Container A (GPU 0).

This file is intentionally self-contained (no ``systemone`` imports): it is
copied into the trainer image and launched by the lifecycle orchestrator as a
one-shot container *after* the vLLM student has been stopped and GPU 0 VRAM
verified empty::

    python trainer_entry.py /workspace/runs/<run_id>/config.json

Config keys (written by ``systemone.training.lifecycle``)::

    run_id, mode ("sft" | "dpo"), base_model, init_from (path or null),
    train_file, output_dir, max_seq_length, lora_rank, lora_alpha,
    epochs, batch_size, grad_accum, learning_rate, dpo_beta, merge (bool), seed

Progress is reported on stdout as ``S1_METRIC {json}`` lines which the
orchestrator parses from the container log stream (loss curve on the
dashboard), and the final result is written to ``<output_dir>/result.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# All linear projection layers of Llama/Qwen/Mistral-style decoders.
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


PINNED_PACKAGES = ["unsloth", "unsloth_zoo", "trl", "datasets", "peft", "transformers", "bitsandbytes",
                   "accelerate", "torch"]


def environment_versions() -> dict:
    """Exact versions this run used - copy them into docker/trainer/requirements.txt."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in PINNED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def emit(kind: str, **data) -> None:
    print("S1_METRIC " + json.dumps({"kind": kind, "ts": time.time(), **data}), flush=True)


def pin_gpu(tag: str) -> None:
    """Pin this process to GPU ``S1_GPU_INDEX`` when isolation is not enforced.

    Compose (``device_ids``) and the orchestrator (``DeviceRequest``) expose a
    single GPU per container on native Linux, where index 0 inside the
    container is the right one. Docker Desktop / WSL2 exposes every GPU to every
    container regardless, so CUDA would default to GPU 0 for all of them and
    the triage server would land on the student's GPU. In that case pin
    explicitly. PCI bus order makes CUDA's numbering match NVML's.
    """
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    want = os.environ.get("S1_GPU_INDEX")
    if want is None or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        visible = [ln for ln in out.splitlines() if ln.startswith("GPU ")]
    except (OSError, subprocess.TimeoutExpired):
        visible = []
    if not visible:  # no nvidia-smi in the image: ask NVML directly
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = [str(i) for i in range(pynvml.nvmlDeviceGetCount())]
            pynvml.nvmlShutdown()
        except Exception:
            visible = []
    if len(visible) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = want
        print(f"[{tag}] {len(visible)} GPUs visible (container GPU isolation not enforced, e.g. WSL2): "
              f"pinning to GPU {want} via CUDA_VISIBLE_DEVICES", flush=True)


def main(config_path: str) -> int:
    pin_gpu("trainer")  # before torch / unsloth initialise CUDA
    cfg = json.loads(Path(config_path).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    emit("status", phase="loading", run_id=cfg["run_id"], mode=cfg["mode"])

    # Unsloth must be imported before transformers / trl to apply its patches.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001

    if cfg["mode"] == "dpo":
        from unsloth import PatchDPOTrainer

        PatchDPOTrainer()
    from datasets import load_dataset
    from transformers import TrainerCallback

    class MetricCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
            if logs:
                emit("log", step=state.global_step, max_steps=state.max_steps, epoch=state.epoch,
                     **{k: v for k, v in logs.items() if isinstance(v, (int, float))})

        def on_train_begin(self, args, state, control, **kwargs):
            emit("status", phase="training", max_steps=state.max_steps)

    model_path = cfg.get("init_from") or cfg["base_model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        target_modules=ALL_LINEAR,
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 3407),
    )
    dataset = load_dataset("json", data_files=cfg["train_file"], split="train")
    emit("status", phase="dataset", rows=len(dataset))

    common = dict(
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        output_dir=str(out / "checkpoints"),
        report_to="none",
        seed=cfg.get("seed", 3407),
    )

    if cfg["mode"] == "dpo":
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # PEFT: the frozen base acts as the reference
            args=DPOConfig(beta=cfg.get("dpo_beta", 0.1), max_length=cfg["max_seq_length"],
                           max_prompt_length=cfg["max_seq_length"] - 256, **common),
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[MetricCallback()],
        )
    else:
        from trl import SFTConfig, SFTTrainer

        # Conversational prompt/completion rows: loss only on the completion
        # (the action JSON), never on the static prompt prefix.
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(max_length=cfg["max_seq_length"], completion_only_loss=True, packing=False, **common),
            callbacks=[MetricCallback()],
        )

    stats = trainer.train()
    emit("status", phase="saving")
    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    result = {
        "environment": environment_versions(),
        "run_id": cfg["run_id"],
        "mode": cfg["mode"],
        "train_loss": float(stats.training_loss),
        "steps": int(stats.global_step),
        "runtime_s": float(stats.metrics.get("train_runtime", 0.0)),
        "adapter_dir": str(adapter_dir),
        "merged_dir": None,
    }
    if cfg.get("merge", True):
        emit("status", phase="merging")
        merged_dir = out / "merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        result["merged_dir"] = str(merged_dir)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    emit("result", **result)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: trainer_entry.py <config.json>", file=sys.stderr)
        sys.exit(2)
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as exc:  # surface the failure to the orchestrator
        emit("error", error=repr(exc), traceback=traceback.format_exc())
        sys.exit(1)
