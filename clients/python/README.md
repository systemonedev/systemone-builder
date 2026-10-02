# systemone

Python client for **System One** decision models such as **Kenning**: ask typed questions about
program state, get typed answers with calibrated probabilities. No text generation, no parsing.

```bash
pip install "systemone @ git+https://github.com/jesusdrodriguez/systemone-builder#subdirectory=clients/python"
pip install "systemone[local] @ git+..."   # + torch/transformers, to run an exported model in-process
```

## Ask a server

```python
from systemone import Client, Noul, Choice, Score

with Client("http://localhost:8093") as client:   # a Kenning server (SystemOne Builder's `kenning` service)
    r = client.system_one(
        state={"ticket": {"subject": "Refund?", "body": "I was charged twice this month, please fix it."}},
        questions={
            "billing": Noul("Is this ticket about billing?"),
            "team": Choice("Which team should handle it?",
                           {"billing": "Charges, refunds, invoices", "technical": "Bugs, errors, outages",
                            "other": "Anything else"}),
            "urgency": Score("How urgent is it?", ["can wait", "this week", "today"]),
        },
    )

r.nouls["billing"].noul           # 0.97  - probability of "yes"
r.choices["team"].choice          # "billing", with .probabilities and .confidence
r.scores["urgency"].score         # 0.9   - probability-weighted level (0..2), see .probabilities
```

Through SystemOne Builder's API instead of the Kenning server directly:
`Client("http://localhost:8090/api", api_key=...)`. `SYSTEMONE_BASE_URL` and `SYSTEMONE_API_KEY`
set the defaults. `AsyncClient` has the same interface for asyncio.

## Run a model in-process

Export a model from the Builder's **Models** page, unzip it, then:

```python
from systemone import Kenning, Noul

model = Kenning.from_pretrained("./kenning-large-v0.1")
r = model.system_one(state="Your invoice is attached, click here to pay now",
                     questions={"phishing": Noul("Is this a phishing attempt?")})
```

Same answers as the server for the same request (same weights, same calibration, same hardware).

## Acting on answers

- `noul` is a **float**: gate with thresholds (`>= 0.9` act, `<= 0.1` close, otherwise escalate), never
  with `bool(noul)`.
- Calibration is fitted on the model's training data. Check it on a few hundred labelled examples from
  your own data before letting it act automatically.

## Compatibility

The wire format (`POST /v1/systemone`) is compatible with TypeSafe AI's System One API, and the question
types use the same names as its MIT-licensed SDK. This package is not affiliated with or endorsed by
TypeSafe AI.

Licence: Apache-2.0.
