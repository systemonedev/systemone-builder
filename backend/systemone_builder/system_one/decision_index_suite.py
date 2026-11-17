"""Jev Decision Index benchmark suite and chance-corrected, area-weighted scoring.

Aligns with the community Decision Index (`multimodalart/jev-decision-index` and
`apolinario/decision-index`):
- Evaluates typed System One decisions (`noul`, `choice`, `score`) across the 5 official
  Decision Index competence areas with their canonical weights:
    * Knowledge & Reasoning     (25.8%)
    * Language Understanding    (25.8%)
    * Retrieval & Classification (20.0%)
    * Tools & Automation        (18.3%)
    * Arts & Human Taste        (10.0%)
- Gold (☆) tasks carry a 1.2x weight within their area; standard tasks carry 1.0x.
- Each question's accuracy is chance-corrected against random guessing (1 / K options)
  and coverage-adjusted (errored or unanswered items score 0 on the [0, 100] scale):
      score_q = 100 * (acc_cov - 1/K) / (1 - 1/K)
- Supports importing an external `apolinario/decision-index` `selected-rows.jsonl(.gz)`
  artifact when present in `bench_cache/decision-index-selected-rows.jsonl.gz` (or via
  `S1_DECISION_INDEX_ROWS`), and ships a self-contained 100-item stratified suite across
  all 5 areas (15 tasks, ground-truth labels) so `systemone bench --suite decision_index`
  runs out of the box without a 17 GB dataset rebuild.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path
from typing import Any

from systemone_builder.system_one.bench import Item, Suite
from systemone_builder.system_one.contract import Question

AREA_WEIGHTS: dict[str, float] = {
    "knowledge_reasoning": 0.258,
    "language_understanding": 0.258,
    "retrieval_classification": 0.200,
    "tools_automation": 0.183,
    "arts_human_taste": 0.100,
}

# Tasks within each area: (task_id, area_id, is_gold_star)
TASK_META: dict[str, tuple[str, bool]] = {
    # Knowledge & Reasoning (25.8%)
    "kr_mmlu_pro": ("knowledge_reasoning", True),
    "kr_bbh_logic": ("knowledge_reasoning", True),
    "kr_winogrande": ("knowledge_reasoning", True),
    # Language Understanding (25.8%)
    "lu_anli": ("language_understanding", True),
    "lu_contract_nli": ("language_understanding", False),
    "lu_sarcasm": ("language_understanding", False),
    # Retrieval & Classification (20.0%)
    "rc_banking77": ("retrieval_classification", True),
    "rc_esci_relevance": ("retrieval_classification", False),
    "rc_secops_routing": ("retrieval_classification", False),
    # Tools & Automation (18.3%)
    "ta_bfcl_tool_select": ("tools_automation", True),
    "ta_when2call": ("tools_automation", False),
    "ta_apibank_param_check": ("tools_automation", False),
    # Arts & Human Taste (10.0%)
    "ah_forecast_plausibility": ("arts_human_taste", False),
    "ah_habermas_consensus": ("arts_human_taste", False),
    "ah_humicroedit_wit": ("arts_human_taste", False),
}


def num_options(q: Question) -> int:
    if q.type == "noul":
        return 2
    if isinstance(q.criteria, (dict, list)) and len(q.criteria) >= 2:
        return len(q.criteria)
    return 2


def chance_corrected_question_score(qm: dict[str, Any], k_options: int, total_asked: int | None = None) -> float | None:
    """Chance-corrected, coverage-adjusted score in [-100/(K-1), 100], where 0 = random guessing and 100 = perfect."""
    n_answered = int(qm.get("n") or 0)
    if n_answered == 0:
        return None
    raw_acc = qm.get("accuracy")
    if raw_acc is None:
        raw_acc = qm.get("exact_level")
    if raw_acc is None:
        return None
    k = max(k_options, 2)
    chance = 1.0 / k
    denom = max(total_asked or n_answered, n_answered, 1)
    # Coverage adjustment: unanswered/errored items contribute chance baseline (0 chance-corrected points)
    cov_ratio = n_answered / denom
    cc = ((float(raw_acc) - chance) / (1.0 - chance)) * 100.0 * cov_ratio
    return round(cc, 2)


def decision_index_breakdown(report: dict[str, Any], suite: Suite | None = None) -> dict[str, Any]:
    """Compute per-area chance-corrected scores and the weighted Jev Decision Index total (0..100)."""
    q_reports = report.get("questions") or {}
    area_weighted_sum: dict[str, float] = {a: 0.0 for a in AREA_WEIGHTS}
    area_weight_denom: dict[str, float] = {a: 0.0 for a in AREA_WEIGHTS}
    task_scores: dict[str, float] = {}

    for qid, qm in q_reports.items():
        if not qm.get("n"):
            continue
        q_obj = suite.questions.get(qid) if suite else None
        k = num_options(q_obj) if q_obj else (2 if qm.get("type") == "noul" else int(qm.get("k_options") or 4))
        cc = qm.get("chance_corrected_score")
        if cc is None:
            cc = chance_corrected_question_score(qm, k)
        if cc is None:
            continue
        task_key = qid.split("::")[0] if "::" in qid else qid
        task_scores[qid] = cc
        area, is_gold = TASK_META.get(task_key, ("retrieval_classification", False))
        w = 1.2 if is_gold else 1.0
        area_weighted_sum[area] = area_weighted_sum.get(area, 0.0) + cc * w
        area_weight_denom[area] = area_weight_denom.get(area, 0.0) + w

    area_scores: dict[str, float] = {}
    for area, w_sum in area_weighted_sum.items():
        denom = area_weight_denom.get(area, 0.0)
        if denom > 0:
            area_scores[area] = round(w_sum / denom, 2)

    if not area_scores:
        return {"index_score": None, "area_scores": {}, "task_scores": task_scores}

    total_w = sum(AREA_WEIGHTS[a] for a in area_scores if a in AREA_WEIGHTS)
    if total_w > 0:
        index_score = sum(area_scores[a] * AREA_WEIGHTS[a] for a in area_scores if a in AREA_WEIGHTS) / total_w
    else:
        index_score = sum(area_scores.values()) / len(area_scores)

    return {
        "index_score": round(index_score, 2),
        "area_scores": area_scores,
        "task_scores": task_scores,
    }


def _canonical_questions() -> dict[str, Question]:
    return {
        # 1. Knowledge & Reasoning (Gold ☆)
        "kr_mmlu_pro": Question(
            type="choice",
            instructions="Which option is the single best technical answer to the problem in the state?",
            criteria={
                "option_a": "Candidate answer A in the state",
                "option_b": "Candidate answer B in the state",
                "option_c": "Candidate answer C in the state",
                "option_d": "Candidate answer D in the state",
            },
        ),
        "kr_bbh_logic": Question(
            type="noul",
            instructions="Does the conclusion logically follow from the premises in the state without contradiction?",
            criteria={
                "true": "The conclusion necessarily follows from the stated premises and constraints",
                "false": "The conclusion contradicts the premises or relies on an invalid logical leap",
            },
        ),
        "kr_winogrande": Question(
            type="choice",
            instructions="Which entity does the ambiguous pronoun in the sentence refer to?",
            criteria={
                "entity_1": "The first entity named in the state",
                "entity_2": "The second entity named in the state",
            },
        ),
        # 2. Language Understanding
        "lu_anli": Question(
            type="choice",
            instructions="What is the natural language inference relationship between the context passage and the hypothesis?",
            criteria={
                "entailment": "The hypothesis is definitely true given the context passage",
                "neutral": "The hypothesis might be true or false; the context does not decide it",
                "contradiction": "The hypothesis is definitely false given the context passage",
            },
        ),
        "lu_contract_nli": Question(
            type="noul",
            instructions="Does the contract clause explicitly permit the action asked about in the inquiry?",
            criteria={
                "true": "The clause explicitly allows or grants the right to perform the action",
                "false": "The clause prohibits the action, requires prior written consent, or omits permission",
            },
        ),
        "lu_sarcasm": Question(
            type="noul",
            instructions="Is the speaker's reply sarcastic or ironic given the situation?",
            criteria={
                "true": "The speaker says the opposite of what they mean to mock or express frustration",
                "false": "The speaker's statement is literal and sincere",
            },
        ),
        # 3. Retrieval & Classification
        "rc_banking77": Question(
            type="choice",
            instructions="Which banking support category best matches the customer's inquiry?",
            criteria={
                "card_payment_not_recognised": "Customer sees an unfamiliar or fraudulent card charge",
                "transfer_not_received_by_recipient": "Outbound wire or bank transfer has not arrived yet",
                "declined_card_payment": "Customer's card transaction was rejected at checkout",
                "lost_or_stolen_card": "Customer lost their physical card or had it stolen",
                "exchange_rate_inquiry": "Customer is asking about FX conversion rates or fees",
            },
        ),
        "rc_esci_relevance": Question(
            type="score",
            instructions="How well does the candidate product match the shopper's search query?",
            criteria=[
                "Irrelevant: wrong product category or incompatible item",
                "Complement: accessory or companion item, not the main product requested",
                "Substitute: similar product that misses one specific constraint (brand, size, or model)",
                "Exact: satisfies every requirement of the shopper's query",
            ],
        ),
        "rc_secops_routing": Question(
            type="choice",
            instructions="How should the security gateway handle this inbound event?",
            criteria={
                "allow": "Legitimate operational traffic or benign message; allow through",
                "escalate": "Ambiguous or policy-borderline event requiring sandbox or analyst review",
                "block": "Confirmed credential theft, injection, malware, or spoofing attack; block immediately",
            },
        ),
        # 4. Tools & Automation
        "ta_bfcl_tool_select": Question(
            type="choice",
            instructions="Which registered tool should the agent invoke to fulfill the user's request?",
            criteria={
                "get_invoice_status": "Look up invoice payment state, due date, and line items by invoice_id",
                "issue_refund": "Refund a settled charge to the customer's original payment method",
                "update_shipping_address": "Modify the delivery address on an unshipped order",
                "quarantine_email": "Isolate a suspicious email message and revoke active links",
            },
        ),
        "ta_when2call": Question(
            type="noul",
            instructions="Should the assistant call an external tool right now instead of answering directly or asking for clarification?",
            criteria={
                "true": "A matching tool is available and all required arguments are present in the state",
                "false": "No tool is needed, or a required parameter is missing and must be asked from the user first",
            },
        ),
        "ta_apibank_param_check": Question(
            type="noul",
            instructions="Does the proposed tool call strictly satisfy the API schema and safety constraints for the state?",
            criteria={
                "true": "Every parameter matches the user's request, types are valid, and policy limits are respected",
                "false": "A parameter is hallucinated, has the wrong value/type, or violates a policy constraint",
            },
        ),
        # 5. Arts & Human Taste
        "ah_forecast_plausibility": Question(
            type="noul",
            instructions="Given the base rates and evidence in the state, is the event more likely than not (>50%) to occur before the deadline?",
            criteria={
                "true": "Current trajectory, historical base rate, and milestones indicate >50% likelihood",
                "false": "Base rates or blocking constraints make occurrence before the deadline unlikely (<50%)",
            },
        ),
        "ah_habermas_consensus": Question(
            type="choice",
            instructions="Which caucus statement best synthesizes both viewpoints into a fair consensus position?",
            criteria={
                "statement_a": "Statement A in the state",
                "statement_b": "Statement B in the state",
                "statement_c": "Statement C in the state",
            },
        ),
        "ah_humicroedit_wit": Question(
            type="choice",
            instructions="Which edited headline creates a sharper, funnier comedic contrast while staying coherent?",
            criteria={
                "edit_1": "Edited headline 1 in the state",
                "edit_2": "Edited headline 2 in the state",
            },
        ),
    }


def _build_builtin_items(seed: int = 42) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []

    # --- 1. Knowledge & Reasoning (21 items: 7 per task) ---
    mmlu_cases = [
        ("kr_m01", "A distributed consensus cluster uses Raft with N=5 nodes. What is the maximum number of simultaneous node failures the cluster can tolerate while still committing new log entries?",
         {"option_a": "1 node", "option_b": "2 nodes", "option_c": "3 nodes", "option_d": "4 nodes"}, "option_b"),
        ("kr_m02", "In TLS 1.3, which key exchange mechanism provides forward secrecy by default during the initial handshake?",
         {"option_a": "Static RSA key transport", "option_b": "Ephemeral Diffie-Hellman (ECDHE)", "option_c": "Pre-shared static Kerberos tickets", "option_d": "Plaintext session resumption"}, "option_b"),
        ("kr_m03", "A binary classifier has a 1% false-positive rate and 90% recall on a threat that occurs in 0.1% (1 in 1,000) of emails. Approximately what fraction of flagged emails are actually threats (precision)?",
         {"option_a": "About 8.3%", "option_b": "About 50.0%", "option_c": "About 90.0%", "option_d": "About 99.0%"}, "option_a"),
        ("kr_m04", "Which HTTP response header instructs browsers never to load the site over unencrypted HTTP for a specified duration?",
         {"option_a": "X-Frame-Options: DENY", "option_b": "Content-Security-Policy", "option_c": "Strict-Transport-Security (HSTS)", "option_d": "Access-Control-Allow-Origin"}, "option_c"),
        ("kr_m05", "What is the time complexity of finding the shortest paths from a single source in a weighted DAG with V vertices and E edges using topological sort?",
         {"option_a": "O(V * E)", "option_b": "O(V^3)", "option_c": "O(V + E)", "option_d": "O(E log V)"}, "option_c"),
        ("kr_m06", "In PostgreSQL MVCC, which isolation level prevents phantom reads and serialization anomalies without requiring explicit table locks?",
         {"option_a": "Read Uncommitted", "option_b": "Read Committed", "option_c": "Repeatable Read / Serializable SSI", "option_d": "Dirty Read"}, "option_c"),
        ("kr_m07", "When calibrating a two-class softmax cross-encoder with temperature scaling T > 0, how does changing T affect the argmax decision boundary at p = 0.5?",
         {"option_a": "It shifts the boundary toward the majority class", "option_b": "It leaves the p = 0.5 argmax boundary unchanged because sign((z1 - z0)/T) is invariant to T > 0", "option_c": "It inverts the predictions when T > 1", "option_d": "It forces all probabilities to 1.0"}, "option_b"),
    ]
    for cid, qtext, opts, ans in mmlu_cases:
        items.append(Item(id=cid, state={"task": "mmlu_pro", "problem": qtext, "candidates": opts}, labels={"kr_mmlu_pro": ans}))

    bbh_cases = [
        ("kr_b01", ["All production database writes require a change ticket.", "Service-A wrote to the production replicas only.", "Replicas are read-only except during replication sync."],
         "Service-A violated the production database write policy by answering a read query.", 0),
        ("kr_b02", ["Every email failing DMARC with policy p=reject must be quarantined before user delivery.", "Message #8841 failed both SPF and DMARC, and the sender domain publishes p=reject."],
         "Message #8841 must be quarantined before user delivery.", 1),
        ("kr_b03", ["If an API key has scope 'billing:write', it requires hardware MFA.", "Token T-99 has scopes ['metrics:read', 'billing:write']."],
         "Token T-99 requires hardware MFA.", 1),
        ("kr_b04", ["No contractor account may access customer PII without VP approval.", "Some engineering interns are contractor accounts.", "Alex is an engineering intern."],
         "Alex definitely has VP approval to access customer PII.", 0),
        ("kr_b05", ["Task C cannot start until both Task A and Task B finish.", "Task A takes 4 hours and Task B takes 7 hours, starting at 09:00 in parallel."],
         "The earliest Task C can start is 16:00.", 1),
        ("kr_b06", ["All signed container images in prod come from registry.internal.", "Image sha256:44a1 was pulled from docker.io without a signature."],
         "Image sha256:44a1 is permitted in prod as a signed internal image.", 0),
        ("kr_b07", ["A refund is auto-approved if and only if amount <= $50 and account_age_days >= 30.", "Order #192 requests a $42 refund on an account created 95 days ago."],
         "Order #192 qualifies for auto-approval.", 1),
    ]
    for cid, prems, concl, ans in bbh_cases:
        items.append(Item(id=cid, state={"task": "bbh_logic", "premises": prems, "conclusion": concl}, labels={"kr_bbh_logic": ans}))

    wino_cases = [
        ("kr_w01", "The firewall blocked the webhook from the billing service because it contained an expired HMAC signature.", "the firewall", "the webhook", "ambiguous pronoun: 'it'", "entity_2"),
        ("kr_w02", "The firewall blocked the webhook from the billing service because it was configured in strict deny-by-default mode.", "the firewall", "the webhook", "ambiguous pronoun: 'it'", "entity_1"),
        ("kr_w03", "The analyst escalated the alert to the incident commander because she lacked permission to isolate the production host.", "the analyst", "the incident commander", "ambiguous pronoun: 'she'", "entity_1"),
        ("kr_w04", "The analyst escalated the alert to the incident commander because she had the sole authority to authorize a region failover.", "the analyst", "the incident commander", "ambiguous pronoun: 'she'", "entity_2"),
        ("kr_w05", "The parser rejected the JSON payload from the legacy client because it contained unescaped control characters.", "the parser", "the JSON payload", "ambiguous pronoun: 'it'", "entity_2"),
        ("kr_w06", "The parser rejected the JSON payload from the legacy client because it enforces strict RFC 8259 compliance.", "the parser", "the JSON payload", "ambiguous pronoun: 'it'", "entity_1"),
        ("kr_w07", "The vendor sent an updated invoice to the finance team after they noticed the wire routing number was outdated.", "the vendor", "the finance team", "ambiguous pronoun: 'they' (who re-issued the invoice with their new bank routing)", "entity_1"),
    ]
    for cid, sent, e1, e2, note, ans in wino_cases:
        items.append(Item(id=cid, state={"task": "winogrande", "sentence": sent, "entity_1": e1, "entity_2": e2, "focus": note}, labels={"kr_winogrande": ans}))

    # --- 2. Language Understanding (21 items: 7 per task) ---
    anli_cases = [
        ("lu_a01", "Deployment v2.14 rolled out to canary at 14:10 UTC and was rolled back at 14:22 UTC after p99 latency doubled from 80ms to 165ms.",
         "Deployment v2.14 remained in production for the entire afternoon without issues.", "contradiction"),
        ("lu_a02", "Deployment v2.14 rolled out to canary at 14:10 UTC and was rolled back at 14:22 UTC after p99 latency doubled from 80ms to 165ms.",
         "The canary deployment of v2.14 lasted 12 minutes before being reverted.", "entailment"),
        ("lu_a03", "Deployment v2.14 rolled out to canary at 14:10 UTC and was rolled back at 14:22 UTC after p99 latency doubled from 80ms to 165ms.",
         "The latency spike in v2.14 was caused by an unindexed SQL join in the billing microservice.", "neutral"),
        ("lu_a04", "Under the Enterprise SLA, scheduled maintenance windows require 72 hours advance notice and may not exceed 30 minutes per month.",
         "An unannounced 45-minute maintenance outage violates the Enterprise SLA.", "entailment"),
        ("lu_a05", "All 40 counterfactual phishing twin pairs in the holdout split are grouped by pair_id so neither twin appears in the training split.",
         "One of the twins from each holdout pair was included in the training set.", "contradiction"),
        ("lu_a06", "The security gateway verified that SPF and DKIM passed for mail.partner-corp.com, but the body link pointed to login-partner-corp.xyz.",
         "Passing SPF and DKIM guarantees that links inside the email body point to the sender's domain.", "contradiction"),
        ("lu_a07", "Customer support ticket volume rose 18% on Monday morning following the mobile app update.",
         "Every ticket opened on Monday morning was submitted from an iOS device.", "neutral"),
    ]
    for cid, ctx, hyp, ans in anli_cases:
        items.append(Item(id=cid, state={"task": "anli", "context": ctx, "hypothesis": hyp}, labels={"lu_anli": ans}))

    contract_cases = [
        ("lu_c01", "Section 4.2 (Subprocessors): Licensee may not engage any third-party subprocessor to handle Customer Data without Licensor's prior written consent.",
         "May the Licensee immediately share Customer Data with a new analytics vendor without asking the Licensor first?", 0),
        ("lu_c02", "Section 7.1 (Open Source): Recipient is hereby granted a perpetual, worldwide, royalty-free license to reproduce, modify, and distribute the Software in source or binary form.",
         "Is the Recipient permitted to distribute modified binary builds of the Software?", 1),
        ("lu_c03", "Section 9.3 (Audit Rights): Upon 15 business days' written notice, Customer may conduct one security audit per calendar year during normal business hours.",
         "May the Customer conduct an annual security audit after providing 15 business days' written notice?", 1),
        ("lu_c04", "Section 5.4 (Data Retention): Vendor shall delete all copies of Production Logs within 30 days of contract termination unless retention is mandated by statutory law.",
         "May the Vendor retain Production Logs indefinitely after contract termination for internal model training?", 0),
        ("lu_c05", "Section 2.2 (Benchmarking): Customer may publish benchmark results comparing the Service against competing offerings provided no proprietary source code is disclosed.",
         "Is the Customer allowed to publish accuracy and latency benchmarks of the Service?", 1),
        ("lu_c06", "Section 6.1 (Assignment): Neither party may assign this Agreement without consent, except in connection with a merger or acquisition of all or substantially all of its assets.",
         "Can a party assign the Agreement to an unrelated third-party reseller outside of a merger or acquisition without consent?", 0),
        ("lu_c07", "Section 3.5 (Service Credits): If monthly uptime falls below 99.9%, Customer is entitled to a 10% service credit applied to the next billing cycle.",
         "Does a monthly uptime of 99.4% entitle the Customer to a 10% service credit?", 1),
    ]
    for cid, clause, inq, ans in contract_cases:
        items.append(Item(id=cid, state={"task": "contract_nli", "clause": clause, "inquiry": inq}, labels={"lu_contract_nli": ans}))

    sarcasm_cases = [
        ("lu_s01", "Context: PagerDuty fired 47 false-positive alerts at 03:15 AM because someone pushed an unvalidated regex rule.",
         "Oh fantastic, nothing builds team morale quite like 47 wake-up calls at 3 AM for a typo.", 1),
        ("lu_s02", "Context: The database migration completed in 4 minutes with zero dropped connections.",
         "Great job on the migration—zero dropped connections and right on schedule.", 0),
        ("lu_s03", "Context: An attacker sent a 'CEO gift card' email from boss-real-account99@gmail.com.",
         "Wow, our CEO definitely uses a random Gmail account to urgently request six Apple gift cards during a board meeting.", 1),
        ("lu_s04", "Context: Post-mortem review of an outage caused by an expired TLS certificate.",
         "We added a 30-day Prometheus alert for certificate expiration so this won't catch us off guard again.", 0),
        ("lu_s05", "Context: A vendor's 'urgent security patch' broke SSO login for the entire company on Friday at 4:55 PM.",
         "Truly love when a 'minor stability hotfix' at 4:55 PM on Friday turns into a weekend incident bridge.", 1),
        ("lu_s06", "Context: Code review comment on a clean PR that reduced p95 latency by 40%.",
         "Nice optimization—pre-allocating the token buffer cut p95 latency almost in half.", 0),
        ("lu_s07", "Context: An LLM agent took 14 seconds and $0.08 of tokens to decide whether a 500 error log is an error.",
         "Clearly we needed a 70-billion-parameter poet to ponder for 14 seconds whether HTTP 500 means an error.", 1),
    ]
    for cid, ctx, reply, ans in sarcasm_cases:
        items.append(Item(id=cid, state={"task": "sarcasm", "situation": ctx, "reply": reply}, labels={"lu_sarcasm": ans}))

    # --- 3. Retrieval & Classification (20 items) ---
    banking_cases = [
        ("rc_b01", "I see a $489.00 charge from 'ELECTRONICS-MEGA-STORE' on my statement from yesterday, but I have never shopped there and still have my card in my wallet.", "card_payment_not_recognised"),
        ("rc_b02", "I wired €2,400 to my landlord's IBAN three business days ago and it left my balance, but my landlord says it still hasn't arrived in their account.", "transfer_not_received_by_recipient"),
        ("rc_b03", "Every time I try to pay for my train ticket at the terminal, my card gets rejected even though I have $1,200 available balance.", "declined_card_payment"),
        ("rc_b04", "I left my wallet in a taxi at the airport twenty minutes ago—please freeze my debit card right away and send a replacement.", "lost_or_stolen_card"),
        ("rc_b05", "If I use my debit card in Tokyo next week, what markup do you charge over the mid-market JPY/USD interbank rate?", "exchange_rate_inquiry"),
        ("rc_b06", "There is a recurring charge of $29.99 to 'STREAM-PLUS-VIP' that I never signed up for or authorized.", "card_payment_not_recognised"),
        ("rc_b07", "My purse was stolen at the train station this morning with my physical Platinum card inside.", "lost_or_stolen_card"),
    ]
    for cid, msg, ans in banking_cases:
        items.append(Item(id=cid, state={"task": "banking77", "customer_message": msg}, labels={"rc_banking77": ans}))

    esci_cases = [
        ("rc_e01", "usb-c to displayport 1.4 cable 6ft 8k 60hz", "Braided USB-C to DisplayPort 1.4 Cable (6 ft, supports 8K@60Hz / 4K@144Hz, Thunderbolt 4 compatible)", 3),
        ("rc_e02", "usb-c to displayport 1.4 cable 6ft 8k 60hz", "USB-C to DisplayPort 1.4 Cable (3 ft, 8K@60Hz) — shorter 3-foot version", 2),
        ("rc_e03", "usb-c to displayport 1.4 cable 6ft 8k 60hz", "Velcro Cable Ties 50-Pack for Desk Wire Management", 1),
        ("rc_e04", "usb-c to displayport 1.4 cable 6ft 8k 60hz", "Stainless Steel French Press Coffee Maker 34 oz", 0),
        ("rc_e05", "men's waterproof trail running shoes size 10.5 gore-tex", "Salomon Speedcross 6 GTX Men's Trail Running Shoes, GORE-TEX Waterproof, Size 10.5 US", 3),
        ("rc_e06", "men's waterproof trail running shoes size 10.5 gore-tex", "Salomon Speedcross 6 Non-Waterproof Mesh Men's Trail Shoe, Size 10.5 US", 2),
        ("rc_e07", "men's waterproof trail running shoes size 10.5 gore-tex", "Bluetooth Mechanical Gaming Keyboard 75% Layout", 0),
    ]
    for cid, q, prod, ans in esci_cases:
        items.append(Item(id=cid, state={"task": "esci", "search_query": q, "candidate_product": prod}, labels={"rc_esci_relevance": ans}))

    secops_cases = [
        ("rc_s01", {"from": "security-noreply@microsoft-verify-login.xyz", "to": "cfo@acme.io", "spf": "fail", "dmarc": "fail",
                    "subject": "URGENT: Your Microsoft 365 password expires in 2 hours", "link": "https://microsoft-verify-login.xyz/auth"}, "block"),
        ("rc_s02", {"from": "ci-bot@acme.io", "to": "eng-alerts@acme.io", "spf": "pass", "dmarc": "pass",
                    "subject": "Build #1942 succeeded on branch main", "link": "https://ci.acme.io/builds/1942"}, "allow"),
        ("rc_s03", {"from": "billing@new-supplier-logistics.com", "to": "ap@acme.io", "spf": "pass", "dmarc": "pass",
                    "subject": "First invoice for PO #7712 (new vendor onboarding pending verification)", "link": "https://new-supplier-logistics.com/invoices/7712"}, "escalate"),
        ("rc_s04", {"from": "ceo.personal99@gmail.com", "to": "payroll@acme.io", "spf": "pass", "dmarc": "pass",
                    "subject": "Urgent direct deposit bank routing change before Friday's payroll run", "link": None}, "block"),
        ("rc_s05", {"from": "notifications@github.com", "to": "dev@acme.io", "spf": "pass", "dmarc": "pass",
                    "subject": "[acme/backend] PR #418 merged by @sarah", "link": "https://github.com/acme/backend/pull/418"}, "allow"),
        ("rc_s06", {"from": "partner@consulting-firm.org", "to": "legal@acme.io", "spf": "softfail", "dmarc": "none",
                    "subject": "Redlined NDA draft attached for tomorrow's call", "attachment": "NDA_Redline_v2.docx"}, "escalate"),
    ]
    for cid, ev, ans in secops_cases:
        items.append(Item(id=cid, state={"task": "secops_routing", "event": ev}, labels={"rc_secops_routing": ans}))

    # --- 4. Tools & Automation (19 items) ---
    bfcl_cases = [
        ("ta_b01", "Can you check whether invoice INV-2026-884 has been paid yet and when it is due?", "get_invoice_status"),
        ("ta_b02", "Please refund the duplicate $79.00 charge (ch_99281) back to the customer's Visa card.", "issue_refund"),
        ("ta_b03", "I typed the wrong apartment number on order ORD-5512—can you change the shipping address to 742 Evergreen Terrace, Apt 4B before it ships?", "update_shipping_address"),
        ("ta_b04", "Message msg-7731 is a confirmed credential-harvesting phish; isolate it from the user's inbox immediately.", "quarantine_email"),
        ("ta_b05", "Look up the line items and payment status for invoice INV-2026-109.", "get_invoice_status"),
        ("ta_b06", "Quarantine email msg-4409 right away so nobody clicks the malicious OAuth link.", "quarantine_email"),
        ("ta_b07", "Customer was billed twice on charge ch_11042; issue a full refund of $35.00.", "issue_refund"),
    ]
    for cid, req, ans in bfcl_cases:
        items.append(Item(id=cid, state={"task": "bfcl_tool_select", "user_request": req}, labels={"ta_bfcl_tool_select": ans}))

    when2call_cases = [
        ("ta_w01", "User: 'Please check the status of invoice INV-9041.' Available tool: get_invoice_status(invoice_id: str).", 1),
        ("ta_w02", "User: 'Can you issue a refund for my order?' Available tool: issue_refund(charge_id: str, amount_usd: float). Note: neither charge_id nor amount_usd has been provided yet.", 0),
        ("ta_w03", "User: 'What is the difference between SPF and DKIM in email authentication?' Available tool: quarantine_email(message_id: str).", 0),
        ("ta_w04", "User: 'Quarantine email message msg-88210 immediately.' Available tool: quarantine_email(message_id: str).", 1),
        ("ta_w05", "User: 'Update the shipping address on order ORD-301 to 100 Market St, Suite 200, San Francisco, CA 94105.' Available tool: update_shipping_address(order_id: str, address: str).", 1),
        ("ta_w06", "User: 'Change my shipping address please.' Available tool: update_shipping_address(order_id: str, address: str). Note: user has not stated the order_id or new address.", 0),
    ]
    for cid, st, ans in when2call_cases:
        items.append(Item(id=cid, state={"task": "when2call", "context": st}, labels={"ta_when2call": ans}))

    apibank_cases = [
        ("ta_a01", {"user_request": "Refund $45.00 on charge ch_5521", "policy": "Max auto-refund is $100.00",
                    "proposed_call": {"tool": "issue_refund", "args": {"charge_id": "ch_5521", "amount_usd": 45.0}}}, 1),
        ("ta_a02", {"user_request": "Refund $45.00 on charge ch_5521", "policy": "Max auto-refund is $100.00",
                    "proposed_call": {"tool": "issue_refund", "args": {"charge_id": "ch_5521", "amount_usd": 450.0}}}, 0),
        ("ta_a03", {"user_request": "Check status of invoice INV-771", "policy": "Read-only lookup",
                    "proposed_call": {"tool": "get_invoice_status", "args": {"invoice_id": "INV-771"}}}, 1),
        ("ta_a04", {"user_request": "Refund $250.00 on charge ch_8810", "policy": "Max auto-refund is $100.00 without manager override",
                    "proposed_call": {"tool": "issue_refund", "args": {"charge_id": "ch_8810", "amount_usd": 250.0}}}, 0),
        ("ta_a05", {"user_request": "Quarantine email msg-319", "policy": "Isolate specified message_id only",
                    "proposed_call": {"tool": "quarantine_email", "args": {"message_id": "msg-999-hallucinated"}}}, 0),
        ("ta_a06", {"user_request": "Quarantine email msg-319", "policy": "Isolate specified message_id only",
                    "proposed_call": {"tool": "quarantine_email", "args": {"message_id": "msg-319"}}}, 1),
    ]
    for cid, st, ans in apibank_cases:
        items.append(Item(id=cid, state={"task": "apibank_param_check", **st}, labels={"ta_apibank_param_check": ans}))

    # --- 5. Arts & Human Taste (19 items) ---
    forecast_cases = [
        ("ah_f01", "Project Atlas has completed 9 of 10 milestones with 3 weeks remaining, zero open P0 bugs, and historical teams at 90% completion 3 weeks out ship on time 88% of the time.", 1),
        ("ah_f02", "A startup with 0 regulatory filings submitted plans to launch a licensed commercial bank in 14 days, where the statutory charter approval process takes a minimum of 9 months.", 0),
        ("ah_f03", "Server fleet migration is at 94% completion with automated cutover tested and 10 days left before the end-of-quarter deadline (pace: 2% per day).", 1),
        ("ah_f04", "Hardware shipment is stuck in customs hold with an estimated 21-day clearance backlog, and the conference demo is in 48 hours with no local spares.", 0),
        ("ah_f05", "Open-source release v1.0 has passed all CI suites, documentation is merged, and the release tag is scheduled for tomorrow morning.", 1),
        ("ah_f06", "A proposal requires unanimous ratification by 12 independent standards committees within 5 days, and 4 of the committees do not meet again until next quarter.", 0),
        ("ah_f07", "Quarterly cloud spend is at $98k on day 28 of 30 with a steady $3.5k/day burn rate; question: will monthly spend exceed $100k before month end?", 1),
    ]
    for cid, ev, ans in forecast_cases:
        items.append(Item(id=cid, state={"task": "forecast_plausibility", "evidence": ev}, labels={"ah_forecast_plausibility": ans}))

    habermas_cases = [
        ("ah_h01",
         "Viewpoint 1 (Security): We should block all external link clicks until every URL is detonated in a full browser sandbox.\nViewpoint 2 (Product Ops): A 5-second delay on every email link breaks time-sensitive customer workflows.",
         {
             "statement_a": "Disable all email security scanning so links open instantaneously.",
             "statement_b": "Force a 10-second full browser detonation on every internal and external link without exception.",
             "statement_c": "Use a fast (<35 ms) calibrated System One model to auto-allow verified safe links and reserve deep sandbox detonation for the uncertain middle band.",
         },
         "statement_c"),
        ("ah_h02",
         "Viewpoint 1 (Developers): Mandatory manual review for every dependency bump slows down security patching.\nViewpoint 2 (SRE): Unreviewed major version upgrades have caused two production outages this quarter.",
         {
             "statement_a": "Auto-merge minor and patch security updates that pass full CI regression suites, while requiring human review and canary rollout for major version changes.",
             "statement_b": "Ban all dependency updates for the rest of the year.",
             "statement_c": "Auto-merge every major version bump straight to production on Friday afternoons without tests.",
         },
         "statement_a"),
        ("ah_h03",
         "Viewpoint 1 (Privacy): Customer support logs contain PII and must not be sent to third-party APIs.\nViewpoint 2 (Support Leads): Triage agents need automated routing and urgency scoring to handle ticket volume.",
         {
             "statement_a": "Send all raw unredacted customer PII to external public APIs anyway.",
             "statement_b": "Run a self-hosted Apache-2.0 System One model on local infrastructure with deterministic PII redaction so tickets are routed automatically without data leaving the VPC.",
             "statement_c": "Shut down automated ticket routing and make customers wait 5 days.",
         },
         "statement_b"),
        ("ah_h04",
         "Viewpoint 1 (Finance): GPU cluster costs are growing too fast because every routine classification calls a 70B LLM.\nViewpoint 2 (AI Team): Accuracy drops if we replace complex reasoning tasks with tiny regexes.",
         {
             "statement_a": "Route 80% of high-volume structured classification to a 400M cross-encoder reflex at $0.50/100k items and escalate only low-confidence cases to the 70B model.",
             "statement_b": "Delete all AI features and use grep.",
             "statement_c": "Use the 70B model with 16x best-of-N sampling on every health-check ping.",
         },
         "statement_a"),
        ("ah_h05",
         "Viewpoint 1 (Remote Staff): Asynchronous written updates keep deep-work blocks uninterrupted across time zones.\nViewpoint 2 (New Hires): Without live pairing sessions, onboarding questions take days to resolve.",
         {
             "statement_a": "Require 8 hours of mandatory all-hands video calls every day.",
             "statement_b": "Keep status updates asynchronous in writing while scheduling dedicated opt-in pairing and onboarding office hours twice a week.",
             "statement_c": "Forbid all live communication between engineers.",
         },
         "statement_b"),
        ("ah_h06",
         "Viewpoint 1 (SOC Analysts): Low-threshold alerts flood the queue with false positives.\nViewpoint 2 (CISO): Raising the threshold blindly risks missing subtle BEC and credential phishing.",
         {
             "statement_a": "Calibrate the model's probabilities on held-out data, set asymmetric thresholds (auto-quarantine >=0.90, auto-deliver <=0.02), and send only the calibrated middle band to analysts.",
             "statement_b": "Turn off all phishing alerts so analysts have an empty queue.",
             "statement_c": "Send every single inbound email in the company to manual SOC review.",
         },
         "statement_a"),
    ]
    for cid, views, stmts, ans in habermas_cases:
        items.append(Item(id=cid, state={"task": "habermas_consensus", "viewpoints": views, "candidates": stmts}, labels={"ah_habermas_consensus": ans}))

    humor_cases = [
        ("ah_w01", "Original headline: 'Scientists Discover New Planet Made Entirely of Diamonds'",
         "Scientists Discover New Planet Made Entirely of Unread Jira Tickets",
         "Scientists Discover New Planet Made of Rocks and Gas", "edit_1"),
        ("ah_w02", "Original headline: 'Local Man Completes Marathon in Under Three Hours'",
         "Local Man Completes marathon in 175 minutes",
         "Local Man Completes 5-Minute 'Quick Sync' Meeting in Under Three Hours", "edit_2"),
        ("ah_w03", "Original headline: 'Tech Giant Unveils Smart Refrigerator That Orders Groceries'",
         "Tech Giant Unveils Smart Refrigerator That Judges Your 2 AM Cheese Habits",
         "Tech Giant Unveils Refrigerator With Cold Compartments", "edit_1"),
        ("ah_w04", "Original headline: 'Archaeologists Uncover Ancient Roman Road Beneath City Center'",
         "Archaeologists Uncover Ancient Soil Beneath City Sidewalk",
         "Archaeologists Uncover Ancient TODO Comment From 2014 Still Blocking Production Deploy", "edit_2"),
        ("ah_w05", "Original headline: 'National Weather Service Issues High Wind Advisory'",
         "National Weather Service Issues Advisory After Senior Architect Starts Explaining Microservices",
         "National Weather Service Reports Moderate Breeze Expected Tuesday", "edit_1"),
        ("ah_w06", "Original headline: 'Study Finds Coffee Drinkers Live Longer'",
         "Study Finds Coffee Drinkers Mostly Just Awake Longer During Morning Standup",
         "Study Finds Beverages Contain Liquid", "edit_1"),
    ]
    for cid, orig, e1, e2, ans in humor_cases:
        items.append(Item(id=cid, state={"task": "humicroedit_wit", "original": orig, "edit_1": e1, "edit_2": e2}, labels={"ah_humicroedit_wit": ans}))

    rng.shuffle(items)
    return items


def _load_external_decision_index_rows(path: Path, n: int, seed: int) -> tuple[dict[str, Question], list[Item]]:
    """Load an external `apolinario/decision-index` `selected-rows.jsonl(.gz)` file if provided."""
    opener = gzip.open if path.name.endswith(".gz") else open
    raw_rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                raw_rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(raw_rows)
    if n > 0:
        raw_rows = raw_rows[:n]
    questions: dict[str, Question] = {}
    items: list[Item] = []
    for idx, r in enumerate(raw_rows):
        qid = str(r.get("id") or r.get("item_id") or f"di_{idx:04d}")
        q_dict = r.get("question") or next(iter((r.get("questions") or {}).values()), None)
        if not q_dict:
            continue
        q_obj = Question.model_validate(q_dict)
        questions[qid] = q_obj
        items.append(Item(id=qid, state=r.get("state"), labels={qid: r.get("label", r.get("target"))}))
    return questions, items


async def decision_index_suite(n: int = 100, seed: int = 42, cache_dir: Path | None = None) -> Suite:
    """Build the Jev Decision Index benchmark suite from real `apolinario/decision-index` rows."""
    ext_env = os.environ.get("S1_DECISION_INDEX_ROWS")
    ext_path = Path(ext_env) if ext_env else (cache_dir / "decision-index-selected-rows.jsonl.gz" if cache_dir else None)
    if not ext_path or not ext_path.exists():
        raise RuntimeError(
            "Real Jev Decision Index rows not found. Set S1_DECISION_INDEX_ROWS or place "
            "`decision-index-selected-rows.jsonl.gz` in bench_cache/ (from `apolinario/decision-index`). "
            "For the internal self-written 100-item probe, use `--suite decision_probe`."
        )
    questions, items = _load_external_decision_index_rows(ext_path, n, seed)
    return Suite(
        name="decision_index",
        description=f"Jev Decision Index (imported from {ext_path.name}; {len(items)} items, chance-corrected [0..100] scoring)",
        questions=questions,
        items=items,
        gate=None,
    )


async def decision_probe_suite(n: int = 100, seed: int = 42) -> Suite:
    """Internal 100-item self-written probe across 5 skill areas (not comparable to `multimodalart/jev-decision-index`)."""
    questions = _canonical_questions()
    items = _build_builtin_items(seed=seed)
    if 0 < n < len(items):
        items = items[:n]
    return Suite(
        name="decision_probe",
        description=(
            f"Internal self-written decision probe: {len(items)} items across 5 weighted areas "
            "(not comparable to the external Jev Decision Index leaderboard)"
        ),
        questions=questions,
        items=items,
        gate=None,
    )
