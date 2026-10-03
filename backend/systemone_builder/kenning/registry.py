"""Kenning model registry, activation and export bundles (no torch needed).

Layout under ``Settings.kenning_dir()`` (the shared workspace volume)::

    models/<name>/       weights (safetensors), tokenizer, kenning.json
    datasets/*.jsonl     training rows (+ .manifest.json with sources and licences)
    exports/<name>.zip   export bundles
    active.json          {"model": "<name>"}: what the Kenning server serves

An export bundle is everything needed to run the model elsewhere: the model
directory, a generated model card (README.md), NOTICE.md with the licences of
the base model and every training source, and SHA256SUMS.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from systemone_builder.kenning.model import CONFIG_FILE, read_config

# The base model's licence and what its own fine-tuning data was (for NOTICE.md).
BASE_LICENSES = {
    "MoritzLaurer/ModernBERT-large-zeroshot-v2.0":
        "Apache-2.0; its zero-shot fine-tuning mix includes non-commercially licensed data (no '-c' variant exists)",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c":
        "MIT; fine-tuned on MNLI, FEVER-NLI (CC-BY-SA-3.0) and Mixtral-generated synthetic data (no non-commercial data)",
    "answerdotai/ModernBERT-large": "Apache-2.0; pretrained on web text, code and scientific articles",
    "microsoft/deberta-v3-large": "MIT",
}
# Bases whose own fine-tuning used non-commercially licensed data: models built on them
# are not released under Apache-2.0 (see weights_licence).
NC_BASES = {"MoritzLaurer/ModernBERT-large-zeroshot-v2.0"}
WEIGHTS_LICENCE = "apache-2.0"
LICENCE_TEXT = Path(__file__).with_name("LICENSE-Apache-2.0.txt")
WEIGHT_SUFFIXES = (".safetensors", ".bin")


class RegistryError(ValueError):
    pass


def _valid(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith(".") or name != Path(name).name:
        raise RegistryError(f"invalid model name {name!r}")
    return name


def model_dir(home: Path, name: str) -> Path:
    path = home / "models" / _valid(name)
    if not path.is_dir():
        raise RegistryError(f"no model named {name!r}")
    return path


def active(home: Path) -> str | None:
    f = home / "active.json"
    try:
        return json.loads(f.read_text()).get("model") if f.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def set_active(home: Path, name: str) -> None:
    model_dir(home, name)
    home.mkdir(parents=True, exist_ok=True)
    (home / "active.json").write_text(json.dumps({"model": name, "since": time.time()}))


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def manifest_for(cfg: dict[str, Any], home: Path) -> dict[str, Any] | None:
    """The training data manifest (sources, licences) if it can be found."""
    trained_on = cfg.get("trained_on")
    if not trained_on:
        return None
    name = Path(trained_on).name
    for candidate in (home / "datasets" / name, Path(trained_on)):
        m = candidate.with_suffix(".manifest.json")
        if m.exists():
            return json.loads(m.read_text())
    return None


def summary(home: Path, name: str) -> dict[str, Any]:
    path = model_dir(home, name)
    cfg = read_config(path)
    held = cfg.get("heldout") or {}
    return {
        "name": name,
        "display_name": cfg.get("name", name),
        "active": active(home) == name,
        "created": cfg.get("created"),
        "base_model": cfg.get("base_model"),
        "trained_on": cfg.get("trained_on"),
        "train_groups": cfg.get("train_groups"),
        "temperature": cfg.get("temperature"),
        "max_length": cfg.get("max_length"),
        "hyperparameters": cfg.get("hyperparameters"),
        "heldout": {k: (held.get(k) or {}).get("all") for k in ("zero_shot", "trained", "calibrated")},
        "heldout_by_type": held.get("calibrated"),
        "size_bytes": _size(path),
        "has_weights": any(f.suffix in WEIGHT_SUFFIXES for f in path.iterdir()),
        "manifest": manifest_for(cfg, home),
    }


def list_models(home: Path) -> list[dict[str, Any]]:
    root = home / "models"
    if not root.is_dir():
        return []
    out = [summary(home, p.name) for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    return sorted(out, key=lambda m: m.get("created") or "", reverse=True)


def delete_model(home: Path, name: str) -> None:
    path = model_dir(home, name)
    if active(home) == name:
        raise RegistryError("cannot delete the active model; activate another one first")
    shutil.rmtree(path)
    bundle = home / "exports" / f"{name}.zip"
    bundle.unlink(missing_ok=True)


# ------------------------------------------------------------------ export
def weights_licence(s: dict[str, Any]) -> tuple[str, str]:
    """(Hugging Face licence id, sentence for the card) for a model's weights."""
    base = s.get("base_model") or "unknown"
    if base in NC_BASES:
        return "other", (f"**Not released under Apache-2.0.** The base model `{base}` was fine-tuned on data that "
                         "includes non-commercially licensed sets, so these weights are for research and evaluation; "
                         "rebuild on a clean base (e.g. `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c`) to "
                         "redistribute.")
    return WEIGHTS_LICENCE, ("The weights are released under the **Apache License 2.0** (LICENSE). Upstream "
                             "licences of the base model and of every training source are listed in NOTICE.md; "
                             "some are share-alike (CC-BY-SA-3.0), so keep NOTICE.md with the weights.")


def _pct(x: Any) -> str:
    return "–" if x is None else f"{x:.3f}"


def model_card(name: str, s: dict[str, Any]) -> str:
    h = s["heldout"]
    base = s.get("base_model") or "unknown"
    spdx, licence_note = weights_licence(s)
    lines = [
        "---",
        f"license: {spdx}",
        f"base_model: {base}",
        "pipeline_tag: zero-shot-classification",
        "tags: [system-one, kenning, decision-model, calibrated]",
        "---",
        "",
        f"# {s.get('display_name') or name}",
        "",
        "Kenning is a **System One** decision model: given program state and typed questions "
        "(`noul` yes/no, `choice`, `score`) it returns typed answers with calibrated probabilities in one "
        "forward pass. It does not generate text. It was built with "
        "[SystemOne Builder](https://github.com/jesusdrodriguez/systemone-builder).",
        "",
        "## Model",
        "",
        f"- Base model: `{base}` ({BASE_LICENSES.get(base, 'see its model card')})",
        "- Architecture: cross-encoder; every candidate answer is scored against the state, and each question's "
        "distribution is `softmax(scores / T)` with per-type temperatures fitted on held-out data",
        f"- Temperatures: `{json.dumps(s.get('temperature'))}`",
        f"- Max tokens per (state, answer) pair: {s.get('max_length')}",
        f"- Created: {s.get('created')}",
        "",
        "## Training data",
        "",
    ]
    m = s.get("manifest")
    if m:
        lines += ["| Source | Rows | Licence |", "|---|---|---|"]
        for key, src in m.get("sources", {}).items():
            lines.append(f"| `{src.get('dataset', key)}` | {src.get('rows')} | {src.get('license')} |")
    else:
        lines.append(f"`{s.get('trained_on')}` (no manifest found)")
    lines += [
        "",
        "## Held-out results (split of the training pool)",
        "",
        "| | accuracy | Brier | ECE |",
        "|---|---|---|---|",
    ]
    for label, key in (("zero-shot (before training)", "zero_shot"), ("trained", "trained"),
                       ("trained + calibrated", "calibrated")):
        r = h.get(key) or {}
        lines.append(f"| {label} | {_pct(r.get('accuracy'))} | {_pct(r.get('brier'))} | {_pct(r.get('ece'))} |")
    lines += [
        "",
        "These are in-distribution numbers. Benchmark on your own data (and recalibrate on a few hundred labelled "
        "examples from it) before letting the model act automatically.",
        "",
        "## Use",
        "",
        "```python",
        "from systemone import Kenning, Noul, Choice",
        "",
        f'model = Kenning.from_pretrained("./{name}")       # in-process, pip install "systemone[local]"',
        'answer = model.system_one(state={"ticket": "I was charged twice."},',
        '                          questions={"billing": Noul("Is this about billing?")})',
        'print(answer.nouls["billing"].noul)',
        "```",
        "",
        "Or serve it with SystemOne Builder's `kenning` service and call `POST /v1/systemone` "
        "with `systemone.Client`.",
        "",
        "## Limitations",
        "",
        "- Calibration was fitted on the training distribution; probabilities on other data are scores until "
        "recalibrated.",
        "- Arithmetic, dates and long or contradictory states degrade accuracy.",
        "- Determinism: identical requests give identical answers on the same hardware and software; "
        "results can differ across GPUs or library versions.",
        "",
        "## Licence",
        "",
        licence_note,
        "",
        "Kenning implements a wire format compatible with TypeSafe AI's System One API. It is not affiliated with "
        "or endorsed by TypeSafe AI, and was not trained on TypeSafe outputs.",
        "",
    ]
    return "\n".join(lines)


def notice(s: dict[str, Any]) -> str:
    base = s.get("base_model") or "unknown"
    lines = ["# NOTICE", "", "This bundle contains model weights derived from:", "",
             f"- Base model `{base}` - licence: {BASE_LICENSES.get(base, 'see https://huggingface.co/' + base)}", ""]
    m = s.get("manifest")
    if m:
        lines += ["Fine-tuned on rows sampled from:", ""]
        for key, src in m.get("sources", {}).items():
            lines.append(f"- `{src.get('dataset', key)}` ({src.get('config', '')} / {src.get('split', '')}), "
                         f"{src.get('rows')} rows - licence: {src.get('license')}")
    lines += ["", "Check each licence's terms (attribution, share-alike) before redistributing.", ""]
    return "\n".join(lines)


def export_bundle(home: Path, name: str) -> Path:
    """Build (or reuse) ``exports/<name>.zip`` and return its path."""
    src = model_dir(home, name)
    out = home / "exports" / f"{name}.zip"
    # rebuilt when the model or the card/NOTICE generator (this file) is newer than the bundle
    newest = max([f.stat().st_mtime for f in src.rglob("*") if f.is_file()] + [Path(__file__).stat().st_mtime])
    if out.exists() and out.stat().st_mtime > newest:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    s = summary(home, name)
    tmp = out.with_suffix(".zip.part")
    sums: list[str] = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                arc = f"{name}/{f.relative_to(src).as_posix()}"
                z.write(f, arc)
                h = hashlib.sha256()
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                sums.append(f"{h.hexdigest()}  {arc}")
        if not (src / CONFIG_FILE).exists() and (src / "s1_config.json").exists():
            z.writestr(f"{name}/{CONFIG_FILE}", (src / "s1_config.json").read_text())
        z.writestr(f"{name}/README.md", model_card(name, s))
        z.writestr(f"{name}/NOTICE.md", notice(s))
        if weights_licence(s)[0] == WEIGHTS_LICENCE and LICENCE_TEXT.exists():
            z.writestr(f"{name}/LICENSE", LICENCE_TEXT.read_text(encoding="utf-8"))
        z.writestr(f"{name}/SHA256SUMS", "\n".join(sums) + "\n")
    tmp.replace(out)
    return out
