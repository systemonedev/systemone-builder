# Contributing

Thanks for helping democratize System-1 distillation.

## Layout

```
backend/systemone_builder/     orchestrator, routing, factory, training, DPO, evaluation, API
backend/tests/         pytest suite (pure logic + real Redis)
frontend/              Next.js + Tailwind dashboard
docker/                images: api, student (vLLM), trainer (Unsloth), dashboard, redis.conf
mac/                   Ollama oracle setup for the Mac endpoint
examples/              reference clients (Playwright agent, Suricata reflex)
docs/                  user documentation
```

## Ground rules

* **No simulated hardware in product code.** Components talk to real NVML, Docker, Redis,
  vLLM and Ollama. Tests cover pure logic and run against a real Redis
  (`S1_TEST_REDIS_URL`). Hardware paths are validated on the reference rig.
* **Protect the hot path.** Anything on the student request path must stay allocation-light
  and must not break prompt-prefix stability (canonical JSON, `prompt_key_order`, scrubbing).
* **Contracts first.** New domains belong in `systemone/templates/*.json` and must pass
  `tests/test_phase7_dx.py::test_default_domains_and_templates_are_self_consistent`.
* **Adapters are plugins.** Prefer an entry-point plugin (`systemone_builder.adapters`) over adding
  a provider to core.

## Workflow

```bash
pip install -e "backend[dev]"
S1_TEST_REDIS_URL=redis://localhost:6379/15 pytest backend/tests
cd frontend && npm ci && npm run lint && npm run build
docker compose config -q
```

Open a pull request with a description of what changed and how you validated it. Include
the hardware you validated on if the change touches the lifecycle, vLLM or Unsloth.
