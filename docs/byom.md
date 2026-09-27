# Bring Your Own Model

Three roles, each reached through an adapter:

| Role | Default | Purpose |
|---|---|---|
| `student` | vLLM serving the fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` | the System-1 reflex model |
| `triage` | vLLM serving `Qwen/Qwen2.5-14B-Instruct-AWQ` | synchronous fallback on GPU 1 |
| `oracle` | Ollama `qwen3.8:27b` on the Mac | System-2 teacher, judge, vision, deep analysis |

## Configure

Configuration is resolved in layers, lowest to highest precedence:

1. `S1_*` environment variables
2. a `systemone.yaml` file whose path is set in `S1_BYOM_FILE`
3. runtime overrides made with `PUT /api/v1/byom/{role}`, persisted in Redis

```yaml
student:
  base_model: meta-llama/Llama-3.2-1B-Instruct   # what Unsloth fine-tunes
  url: http://student:8000/v1
triage:
  adapter: openai
  url: http://triage:8000/v1
  model: mistralai/Mistral-Small-24B-Instruct-2501
oracle:
  adapter: ollama
  url: http://192.168.1.50:11434
  model: llama3.3:70b
  vision_model: qwen2.5vl:32b
```

Check a student candidate before training:

```bash
systemone validate-model meta-llama/Llama-3.2-1B-Instruct
```

This checks that the model exists, whether it's gated, and that its `model_type` is supported
by Unsloth.

When the student's `base_model` changes, set `S1_STUDENT_BASE_MODEL` for the student
container too, so vLLM serves the same base until the first fine-tune lands.

## Adapters

| Kind | Works with |
|---|---|
| `openai` | vLLM, SGLang, llama.cpp `llama-server`, LM Studio, TGI, LocalAI; any `/v1/chat/completions` server. Uses streaming, `response_format: json_schema` guided decoding and logprobs when available. |
| `ollama` | Ollama native `/api/chat`. Uses JSON-schema `format`, images for vision, and logprobs on recent versions. |

### Writing a plugin

```python
from systemone.adapters.base import Generation, ModelAdapter

class MLXAdapter(ModelAdapter):
    kind = "mlx"
    async def generate(self, messages, *, json_schema=None, max_tokens=256, temperature=0.0,
                       images=None, logprobs=False, model=None) -> Generation: ...
    async def health(self): ...
    async def list_models(self): ...
```

```toml
[project.entry-points."systemone.adapters"]
mlx = "my_pkg.adapters:MLXAdapter"
```

Install the plugin into the API image, then set `adapter: mlx` for a role.
