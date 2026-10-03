# Security policy

SystemOne Builder runs models, training jobs and an HTTP API on your own hardware. We treat
anything that lets someone reach those services, read your data or keys, or run code they
should not as a security issue.

## Reporting a vulnerability

Please **do not open a public issue** for a vulnerability. Use GitHub's private
vulnerability reporting ("Security" tab → "Report a vulnerability") on this repository, or
email **systemonedev@gmail.com** with "SECURITY" in the subject. Include what you found, how to
reproduce it, and the version or commit.

We aim to acknowledge reports within 3 working days and to agree on a fix and disclosure
timeline with you. We credit reporters in the release notes unless you prefer otherwise.

## Supported versions

Only the latest release and `main` receive security fixes until 1.0.

## What the defaults assume

- Every service binds to `127.0.0.1`. The dashboard gateway keeps `S1_API_KEY` on the
  server. It checks the origin and `Host` header on every request, and it asks for a login
  when bound to a LAN address. Exposing any port beyond localhost is your decision. Put
  TLS and authentication in front of it.
- Model servers (`kenning`, `clef`, `student`, `triage`) have no authentication. Keep them on
  the Docker network or localhost.
- Keep secrets in `.env`, which git ignores, and never in the dashboard or in browser storage.
  `TYPESAFE_API_KEY` is only needed if you opt in to benchmarking against Jev.
- Model weights and code downloaded from Hugging Face run in your containers. Clef loads
  Python code shipped with its weights (`joint_schema_model.py`). Only load models from
  publishers you trust.

## In scope

The backend API, the dashboard and its gateway, the Docker images and compose file, the
`systemone` Python client, and export bundles.

## Out of scope

Model quality issues, such as wrong or poorly calibrated answers, are bugs, not
vulnerabilities. Please open a normal issue with a reproducible example.
