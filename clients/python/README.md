# systemone-client

`pip install systemone-client`, then `import systemone`.

The Python client for **System One** decision models. You ask typed questions about your program's
state and get typed answers with calibrated probabilities: yes/no, pick one, or rate on a scale. The
model doesn't generate text, so there's no parsing, no retries on malformed output and no prompt
templates.

```python
from systemone import Client, Noul

with Client("http://localhost:8093") as client:
    r = client.system_one(state={"email": {"from": "it-support@examp1e.com", "body": "Verify your password now"}},
                          questions={"phishing": Noul("Is this email a phishing attempt?")})

p = r.nouls["phishing"].noul          # probability of "yes": 0.78 with kenning-large-v0.4
```

It works with any server that speaks the System One wire format (`POST /v1/systemone`):

| Engine | Where it runs | How to reach it |
|---|---|---|
| **Kenning** (open, Apache-2.0) | your machine, ~2 GB of GPU memory, or CPU | [SystemOne Builder](https://github.com/systemonedev/systemone-builder)'s `kenning` service, or in-process with `Kenning.from_pretrained` |
| **Clef** (Cloudflare, Apache-2.0) | your machine, a 24 GB GPU | SystemOne Builder's optional `clef` service |
| **Jev** (TypeSafe AI, hosted) | TypeSafe's API | `Client("https://api.typesafe.ai", api_key=...)` with your TypeSafe key |

The same code runs against all three: switch with the base URL.

- [Install](#install)
- [Quickstart](#quickstart)
- [Questions](#questions)
- [Answers](#answers)
- [Acting on answers](#acting-on-answers)
- [Run Kenning in-process](#run-kenning-in-process)
- [Async](#async)
- [Configuration and errors](#configuration-and-errors)
- [Compatibility](#compatibility)
- [Contributing](#contributing)

## Install

```bash
pip install systemone-client              # the HTTP client (only dependency: httpx)
pip install "systemone-client[local]"     # + torch and transformers, to run Kenning in your own process
```

It needs Python 3.10 or later. For the latest unreleased code, install from source:

```bash
pip install "systemone-client @ git+https://github.com/systemonedev/systemone-builder#subdirectory=clients/python"
```

You also need a model to ask. The quickest way is SystemOne Builder, which starts Kenning on one GPU:

```bash
git clone https://github.com/systemonedev/systemone-builder && cd systemone-builder
cp .env.example .env                 # set S1_REDIS_PASSWORD (openssl rand -hex 24)
docker compose up -d                 # Kenning on 127.0.0.1:8093, dashboard on :3090
```

Or skip the server and [run Kenning in-process](#run-kenning-in-process).

## Quickstart

```python
from systemone import Client, Noul, Choice, Score

client = Client("http://localhost:8093")          # reuse one client: it keeps connections open

r = client.system_one(
    state={
        "ticket": {"subject": "Refund?", "body": "I was charged twice this month, please fix it."},
        "customer": {"plan": "pro", "tenure_months": 14},
    },
    questions={
        "billing": Noul("Is this ticket about billing?"),
        "team": Choice("Which team should handle it?", {
            "billing": "Charges, refunds, invoices",
            "technical": "Bugs, errors, outages",
            "other": "Anything else",
        }),
        "urgency": Score("How urgent is it?", ["can wait", "this week", "today"]),
    },
)

r.nouls["billing"].noul             # float in [0, 1]: probability of "yes"
r.choices["team"].choice            # "billing"
r.choices["team"].probabilities     # {"billing": 0.96, "technical": 0.01, "other": 0.03}
r.scores["urgency"].score           # probability-weighted level index, 0.0 .. 2.0 (here 0.49)
r.model, r.latency_ms               # which model answered, and how fast
```

All questions in one call are answered together, in one pass over the state. Ask everything you need
in a single request rather than one request per question.

**`state`** is whatever describes the situation: a string, or a JSON object (a dict; wrap lists and
other values in one). Pass your real structured data as it is; you don't need to write a prompt. Keep it focused: models read a
limited number of tokens (Kenning: 512 per question/answer pair), so send the fields that matter.

## Questions

| Type | Use it for | Arguments | Answer |
|---|---|---|---|
| `Noul(instructions, criteria=None)` | yes/no | `criteria`: optional text clarifying what counts as yes | `NoulAnswer.noul`: P(yes) |
| `Choice(instructions, criteria)` | pick exactly one of 2-255 options | `criteria`: `{option: description or None}` | `ChoiceAnswer.choice`, `.probabilities`, `.confidence` |
| `Score(instructions, criteria)` | an ordered scale of 2-10 levels | `criteria`: the levels, lowest first | `ScoreAnswer.score`, `.probabilities`, `.confidence`, `.legend` |

Writing good questions:

- **One decision per question.** "Is this phishing?" and "Is this urgent?" are two nouls, not one.
- **Describe choice options.** `{"billing": "Charges, refunds, invoices"}` is answered more
  accurately than `{"billing": None}`, especially when option names are terse or overlap.
- **Use `Score` only when the levels are ordered.** Severity, urgency or quality are scores. Team,
  category or language are choices: a score's average of "billing" and "technical" means nothing.
- **Question ids are yours.** Use stable ids such as `"billing"`; they are the keys of the answers.

Plain dicts work too, in the wire format: `{"type": "noul", "instructions": "..."}`.

## Answers

`system_one()` returns a `Response`:

| Attribute | Type | Meaning |
|---|---|---|
| `nouls` | `dict[str, NoulAnswer]` | `.noul`: probability of yes |
| `choices` | `dict[str, ChoiceAnswer]` | `.choice` (the most likely option), `.probabilities` (all options, summing to 1), `.confidence` |
| `scores` | `dict[str, ScoreAnswer]` | `.score` (expected level index), `.probabilities` per level, `.confidence`, `.legend` (index to level name) |
| `model` | `str` | the model that answered, e.g. `kenning-large-v0.4` |
| `usage` | `Usage` | `.input_tokens`, `.output_tokens` (always 0: nothing is generated) |
| `latency_ms` | `float \| None` | server-side time, when the server reports it |
| `raw` | `dict` | the response exactly as received |

`confidence` is `(max p - 1/n) / (1 - 1/n)`: 0 when every option is equally likely, 1 when one
option has all the probability. A noul has no `confidence`: its probability already is the
confidence.

## Acting on answers

The point of calibrated probabilities is to decide **what the model may do on its own**:

```python
p = r.nouls["phishing"].noul
if p >= 0.9:
    quarantine(email)                 # sure it's phishing: act
elif p <= 0.1:
    deliver(email)                    # sure it's not: close
else:
    send_to_analyst(email, p)         # unsure: a person, or a slower model, decides
```

- **A noul is a float.** Never write `if r.nouls["x"].noul:` because `bool(0.02)` is `True`. Compare it
  with thresholds.
- **Pick thresholds from the cost of mistakes**, not from 0.5. If missing a phishing email is far worse
  than a false alarm, lower the "close" threshold (e.g. `<= 0.02`) and accept more escalations.
- **A `score` can hide a split.** 0.5 / 0 / 0.5 over low / medium / high averages to "medium" but
  means "either low or high". Check `.probabilities` before acting on a middle value.
- **Calibration is measured, not promised.** Probabilities are calibrated on the model's training
  distribution. Before letting a model act unsupervised, measure it on a few hundred labelled examples
  of your own data. SystemOne Builder's Verify page and `systemone bench` do this.

## Run Kenning in-process

With `pip install "systemone-client[local]"` you can load Kenning without running a server: from the
Hugging Face Hub, or from a bundle exported on SystemOne Builder's Models page.

```python
from systemone import Kenning, Noul

model = Kenning.from_pretrained("systemonedev/kenning-large-v0.4")    # or "./my-exported-model"
r = model.system_one(state="Your invoice is attached, click here to pay now",
                     questions={"phishing": Noul("Is this a phishing attempt?")})
```

- The first call downloads the model (~0.9 GB) into the Hugging Face cache. Its `kenning.json`, which
  holds the fitted calibration temperatures, comes with it.
- It runs on CUDA when available, otherwise on CPU (slower).
- Answers are deterministic and match the Kenning server for the same request on the same hardware
  and library versions.
- `Kenning(path, device="cpu", max_length=512)` overrides the device or the token limit.

## Async

`AsyncClient` has the same interface:

```python
import asyncio
from systemone import AsyncClient, Noul

texts = ["WIN A FREE CRUISE, reply YES", "Lunch at 1pm?"]

async def main():
    async with AsyncClient("http://localhost:8093") as client:
        results = await asyncio.gather(*(
            client.system_one(state=t, questions={"spam": Noul("Is this spam?")}) for t in texts))

asyncio.run(main())
```

## Configuration and errors

```python
Client(base_url=None, api_key=None, timeout=30.0, model=None)
```

| Argument | Default | Notes |
|---|---|---|
| `base_url` | `$SYSTEMONE_BASE_URL`, else `http://localhost:8093` | the server root; the client posts to `{base_url}/v1/systemone` |
| `api_key` | `$SYSTEMONE_API_KEY` | sent as `Authorization: Bearer` and `X-API-Key` |
| `timeout` | 30 s | per request |
| `model` | server default | requested model id, for servers that host several (Jev: `jev-latest`) |

Common base URLs:

| Server | `base_url` | `api_key` |
|---|---|---|
| Kenning server (SystemOne Builder) | `http://localhost:8093` | none (localhost only) |
| SystemOne Builder API | `http://localhost:8090/api` | your `S1_API_KEY` |
| Clef server (SystemOne Builder, `clef` profile) | `http://localhost:8094` | none |
| TypeSafe Jev | `https://api.typesafe.ai` | your TypeSafe key |

Errors:

- `SystemOneError`: the server answered with an error. `.status` is the HTTP status and `.detail` the
  server's message. A 422 means the request was invalid, for example a `Choice` with one option.
- `httpx.TimeoutException` and `httpx.ConnectError`: the server was slow or unreachable.
- `ValueError`: the request is invalid before sending, for example no questions or bad criteria.

## Compatibility

The wire format is compatible with TypeSafe AI's System One API, and the question types use the same
names as its MIT-licensed SDK, so code moves between engines unchanged. This package is not affiliated
with or endorsed by TypeSafe AI or Cloudflare. Using Jev through it is governed by your own TypeSafe
agreement.

Versioning follows [SemVer](https://semver.org): until 1.0, minor versions may change the API, and the
[changelog](https://github.com/systemonedev/systemone-builder/blob/main/clients/python/CHANGELOG.md) lists
every change.

## Contributing

The package lives in [`clients/python`](https://github.com/systemonedev/systemone-builder/tree/main/clients/python)
of the SystemOne Builder repository. Issues and pull requests are welcome.

```bash
git clone https://github.com/systemonedev/systemone-builder && cd systemone-builder/clients/python
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                               # unit tests, no server or GPU needed
```

- **Keep the core light.** The HTTP client depends only on `httpx`. Anything heavier belongs in an
  extra, as `torch` does in `[local]`.
- **Test against a mock transport.** `tests/test_client.py` shows the pattern with
  `httpx.MockTransport`. Changes to `systemone.local` must keep the parity test with the Kenning server
  passing (`backend/tests` in the repository).
- **Wire-format changes** must stay compatible with Kenning, Clef and Jev, or be clearly opt-in.
- **Sign off your commits** (`git commit -s`, the Developer Certificate of Origin) and follow the
  [code of conduct](https://github.com/systemonedev/systemone-builder/blob/main/CODE_OF_CONDUCT.md). The
  repository's [CONTRIBUTING.md](https://github.com/systemonedev/systemone-builder/blob/main/CONTRIBUTING.md)
  has the rest.
- **Security issues:** report them privately, see
  [SECURITY.md](https://github.com/systemonedev/systemone-builder/blob/main/SECURITY.md).

Releases are cut by maintainers. They bump `version` in `pyproject.toml` and `systemone/__init__.py`,
add a `CHANGELOG.md` entry, and push a `client-vX.Y.Z` tag. CI then tests, builds and publishes to
PyPI through Trusted Publishing, with no stored tokens.

## Licence

Apache-2.0. Kenning weights published by [systemonedev](https://huggingface.co/systemonedev) are also
Apache-2.0; see each model's card and `NOTICE.md`.
