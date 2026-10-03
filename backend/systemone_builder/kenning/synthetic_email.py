"""Synthetic modern emails written by a local System 2 model (the teacher).

Public phishing corpora are mostly old (Enron-era spam, 419 scams, CEAS 2008), so a
model trained on them misses short, modern phishing - credential harvesting
behind a fake login page, invoice and bank-detail fraud, gift-card requests - and
confuses it with the legitimate security mail people really get. This module asks
an OpenAI-compatible model (by default the builder's triage server) to write
realistic emails for scenarios on both sides, including hard legitimate ones.

The label comes from the scenario that was requested, never from the teacher's
judgement. Outputs of an Apache-2.0 model (Qwen2.5-7B-Instruct by default).
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import httpx

PHISHING = [
    "a fake Microsoft 365 / Outlook alert that the mailbox is full or the password expires, with a link to 'keep' the password",
    "an unusual sign-in alert that asks the reader to verify their identity on a look-alike login page",
    "a fake shared document (DocuSign, SharePoint, Google Drive) that requires the email password to open",
    "an executive impersonation asking an employee to urgently buy gift cards and send the codes",
    "a vendor or supplier announcing changed bank details and asking for an outstanding payment to the new account",
    "a fake overdue invoice with a payment link or an attachment to open",
    "a parcel delivery notice (DHL, UPS, FedEx, USPS, Royal Mail) asking for a small customs or redelivery fee",
    "a streaming or subscription service saying payment failed and the card must be updated via a link",
    "a fake bank or PayPal security notice that the account is limited until identity and card are confirmed",
    "an HR or payroll message asking staff to log in to update direct-deposit details before payday",
    "an IT helpdesk message asking the reader to install an 'urgent security update' from a link",
    "an MFA push-fatigue follow-up asking the reader to approve the prompt or read back their code",
    "a tax refund or government benefit notice asking for bank details",
    "a cryptocurrency wallet or exchange warning asking for the recovery phrase to 'secure' funds",
    "a fake job offer asking for personal details and a small onboarding fee",
    "a voicemail or fax notification with a link to 'listen' that leads to a login page",
    "a social-media account violation notice threatening deletion unless the reader logs in",
    "a lottery, prize or inheritance notice asking for a processing fee",
    "a fake domain or website renewal notice with a payment link",
    "a request from 'the CEO' or 'legal' for a confidential wire transfer before end of day",
]
LEGIT = [
    "a colleague sharing meeting slides or notes, no links except an internal file share",
    "a genuine automated security alert from a real service about a new sign-in that the reader can review in their own account settings, without asking for credentials",
    "a password-reset email the reader requested themselves, saying to ignore it if they did not",
    "a one-time verification code email that says the code expires and to ignore it if not requested",
    "an accounts-payable confirmation that an invoice was paid through the usual vendor portal",
    "an IT announcement of scheduled maintenance that needs no action",
    "an order shipped / delivered notification from an online store",
    "a calendar invitation for a meeting",
    "an HR reminder about benefits enrolment pointing to the portal people normally use",
    "a newsletter from a software project or publication",
    "a customer support reply confirming a refund was processed",
    "a teammate asking a quick question about a project",
    "a CI / build notification from a developer tool",
    "a receipt for a subscription renewal the reader set up",
    "a bank statement-is-ready notice that says to log in through the usual app, without a link",
    "an event registration confirmation with the date and venue",
    "a manager approving a time-off request",
    "an internal security-awareness reminder that tells people never to share passwords",
    "a vendor sending a quote that was requested, attached as PDF",
    "a friend or family member making weekend plans",
]
STYLES = ["very short (1-2 sentences)", "short (2-4 sentences)", "medium (one paragraph)", "formal", "casual",
          "with a greeting and sign-off", "terse and urgent", "with a few typos"]

# Calm, professional lures paired with legitimate twins from the real service. Loud
# scams are easy; these teach the difference that matters: a look-alike sender, and
# a request to sign in, confirm or pay through a link, versus a notice that sends
# you to the account you already use (or needs nothing).
SUBTLE_PAIRS = [
    ("a calm, professional notice that a file was shared, which asks you to sign in with your work account on a link from a look-alike domain",
     "a genuine file-share notification from the colleague's own company domain, pointing to the usual shared drive"),
    ("a polite mailbox storage or password-expiry notice from a look-alike IT domain, offering a self-service link to keep the account working",
     "a genuine IT notice from the company's real domain about a password policy, telling people to change it in the usual settings page"),
    ("a quiet tax or payroll document notice (W-2, payslip) asking you to verify your login to download it",
     "a genuine payroll notice from the company's HR domain saying payslips are available in the usual HR portal"),
    ("a courteous subscription-renewal problem notice from a look-alike billing domain, asking to review payment details via a link",
     "a genuine receipt or renewal confirmation from the real vendor domain, with no request to enter details"),
    ("a short, discreet request from an executive's personal address to arrange a payment or share a phone number",
     "a genuine short request from a manager's company address about ordinary work, with no money or secrets involved"),
    ("a calm security-alert email from a look-alike domain asking you to confirm your account details",
     "a genuine security alert from the real service that says no action is needed if it was you, and to check activity in your account settings"),
    ("a professional MFA or device re-enrollment request from a look-alike domain with a link",
     "a genuine IT announcement that MFA enrollment is done in person or in the usual company portal, with no link to an unknown site"),
    ("a calm message from a colleague-like sender asking to send an invoice payment to new bank details because 'the portal is down'",
     "a genuine accounts-payable note confirming an invoice was paid through the normal vendor portal"),
]
SUBTLE_STYLES = ["calm and professional, no urgency words", "brief and businesslike", "friendly and polite",
                 "formal corporate tone"]

SCHEMA = {"type": "object", "properties": {"from": {"type": "string"}, "subject": {"type": "string"},
                                           "body": {"type": "string"}},
          "required": ["from", "subject", "body"]}


def prompt(scenario: str, style: str, malicious: bool) -> str:
    kind = ("a realistic modern phishing or scam email used in security-awareness training" if malicious
            else "a realistic, completely legitimate modern email")
    return (f"Write {kind}. Scenario: {scenario}. Style: {style}. Invent plausible names, companies and domains "
            "(no real people). Return JSON with the sender address in 'from', the 'subject' and the 'body'.")


async def generate(base_url: str, model: str, n_per_class: int, seed: int, concurrency: int = 16,
                   api_key: str | None = None, subtle_share: float = 0.0) -> list[dict[str, Any]]:
    """``n_per_class`` phishing + ``n_per_class`` legitimate emails; ``subtle_share`` of each
    class comes from the matched calm-lure / legitimate-twin pairs."""
    rng = random.Random(seed)
    n_subtle = int(n_per_class * subtle_share)
    jobs = [(True, rng.choice(PHISHING), rng.choice(STYLES)) for _ in range(n_per_class - n_subtle)] + \
           [(False, rng.choice(LEGIT), rng.choice(STYLES)) for _ in range(n_per_class - n_subtle)]
    for _ in range(n_subtle):
        lure, twin = rng.choice(SUBTLE_PAIRS)
        style = rng.choice(SUBTLE_STYLES)
        jobs += [(True, lure, style), (False, twin, style)]
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def one(client: httpx.AsyncClient, i: int, malicious: bool, scenario: str, style: str) -> dict[str, Any] | None:
        body = {"model": model, "temperature": 0.9, "top_p": 0.95, "seed": seed * 100_003 + i, "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt(scenario, style, malicious)}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "email", "schema": SCHEMA}}}
        async with sem:
            try:
                r = await client.post("/chat/completions", json=body)
                email = json.loads(r.json()["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, ValueError):
                return None
        if not all(isinstance(email.get(k), str) and email[k].strip() for k in ("from", "subject", "body")):
            return None
        return {"email": {k: email[k].strip()[:1500] for k in ("from", "subject", "body")},
                "malicious": malicious, "scenario": scenario}

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=120, headers=headers) as client:
        out = await asyncio.gather(*(one(client, i, *job) for i, job in enumerate(jobs)))
    seen: set[str] = set()
    rows = []
    for x in out:
        if x and x["email"]["body"] not in seen:
            seen.add(x["email"]["body"])
            rows.append(x)
    return rows
