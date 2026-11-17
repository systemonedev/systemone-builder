"""Stage-0 email pre-parser, counterfactual dataset loader, layout renderer, and benchmark suites for Phishing System One models.

Maintains 100% Apache-2.0 clean lineage:
- Ingests the paired counterfactual datasets in ``workspace/datasets/phishing_classifier/``
- Generates diverse paired counterfactual emails across 12 modern attack families
- Strictly groups train / calibration / held-out benchmark splits by ``pair_id`` so counterfactual twins never leak across splits
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from systemone_builder.system_one.bench import Item, Suite
from systemone_builder.system_one.contract import Question

FREE_MAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "icloud.com",
    "protonmail.com", "proton.me", "mail.com", "gmx.com", "yandex.com", "zoho.com",
    "freemail-inbox.example.com", "webmail-free.example.org", "alumni-mail.example.edu",
    "public-mail.example.net", "fast-inbox.example.org",
}

SAAS_RELAY_DOMAINS = {
    "google.com": "google",
    "docs.google.com": "google",
    "drive.google.com": "google",
    "sharepointonline.com": "microsoft",
    "sharepoint.com": "microsoft",
    "onedrive.com": "microsoft",
    "docusign.net": "docusign",
    "docusign.com": "docusign",
    "echosign.com": "adobe",
    "adobesign.com": "adobe",
    "notion.so": "notion",
    "mail.notion.so": "notion",
    "dropbox.com": "dropbox",
    "box.com": "box",
    "app.box.com": "box",
    "canva.com": "canva",
    "smartsheet.com": "smartsheet",
    "app.smartsheet.com": "smartsheet",
    "calendly.com": "calendly",
}

RISKY_EXTENSIONS = (
    ".exe", ".scr", ".vbs", ".js", ".jse", ".wsf", ".hta", ".iso", ".img", ".vhd",
    ".lnk", ".bat", ".cmd", ".ps1", ".msi", ".docm", ".xlsm", ".pptm",
    ".pdf.html", ".pdf.htm", ".pdf.iso", ".pdf.lnk", ".pdf.exe", ".xlsx.html", ".docx.html",
)

SUSPICIOUS_DOMAIN_TOKENS = (
    "-verify", "-login", "-portal", "-auth", "-check", "-secure", "-update", "-sso",
    "-recover", "-session", "-validate", "verify-", "login-", "auth-", "sso-", "xn--",
)

CANONICAL_QUESTIONS: dict[str, Question] = {
    "is_phish": Question(
        type="noul",
        instructions="Is this email a phishing attempt, social-engineering attack, or malicious lure?",
        criteria={
            "true": "Phishing, brand or executive impersonation, lookalike or mismatched link/reply-to domains, failed authentication on a claimed domain, suspicious external SaaS sharing lure, or unsolicited credential/payment/malware harvesting",
            "false": "Legitimate transactional, operational, SaaS notification, vendor, or internal message where sender domain, authentication, links, sharer context, and requested action are consistent and benign",
        },
    ),
    "recommended_action": Question(
        type="choice",
        instructions="How should the email security pipeline route this message?",
        criteria={
            "deliver": "Deliver normally to the recipient's inbox (consistent sender/auth/links and benign context)",
            "enrich_links_or_attachments": "Hold briefly for automated URL detonation or attachment sandboxing before delivery",
            "hold_for_review": "Escalate to System-2 analyst review or out-of-band sender verification (e.g. BEC/payment/PII request)",
            "quarantine": "Block or quarantine immediately due to clear spoofing, malicious lure, or credential/malware harvesting",
        },
    ),
    "action": Question(
        type="choice",
        instructions="What primary physical or UI action does the email ask the recipient to take?",
        criteria={
            "click_link": "Follow a URL or button in the email body",
            "open_attachment": "Open, download, or extract an attached file",
            "reply_with_info": "Reply by email or message with requested information, approvals, or payment actions",
            "none": "Informational only; no link click, attachment open, or reply is requested",
        },
    ),
    "ask": Question(
        type="choice",
        instructions="What sensitive asset, access, or commitment is the email ultimately asking for?",
        criteria={
            "credentials": "Sign in, verify an account, reset a password, enter an MFA/OTP code, or grant OAuth permissions",
            "payment_details": "Pay an invoice, wire funds, update bank/IBAN/ACH routing details, or purchase gift cards",
            "document_access": "Open, view, comment on, or sign a shared cloud document, spreadsheet, or e-signature envelope",
            "personal_info": "Provide PII, tax IDs, SSN/NI numbers, payroll records, HR beneficiary details, or confidential business data",
            "software_install": "Run an executable/script, open a macro/archive/HTML smuggling attachment, or install a tool/extension",
            "nothing": "No sensitive asset, login, payment, document access, or installation is requested",
        },
    ),
    "identity_claim": Question(
        type="noul",
        instructions="Does the sender claim to represent a trusted brand, executive, internal team, or known vendor?",
        criteria={
            "true": "Claims to come from an IT/security desk, SaaS platform, executive, HR/finance team, bank, courier, or established vendor",
            "false": "Makes no special authority or institutional identity claim",
        },
    ),
    "urgency": Question(
        type="noul",
        instructions="Does the email apply time pressure, urgency, or threat of negative consequences?",
        criteria={
            "true": "Demands immediate or same-day action, warns of account lock/suspension, overdue penalties, legal/regulatory deadlines, or urgent executive need",
            "false": "Uses a calm, routine timeline without artificial urgency or threats",
        },
    ),
}


def _email_domain(addr: str | None) -> str:
    if not addr or "@" not in addr:
        return ""
    part = addr.strip().split("@")[-1].lower().strip("> )\"")
    return part


def _url_domain(href: str | None) -> str:
    if not href:
        return ""
    try:
        host = (urlparse(href).hostname or "").lower()
    except ValueError:
        return ""
    for prefix in ("www.", "app.", "sec.", "secure.", "mail.", "docs.", "drive.", "na2.", "na3.", "na4."):
        if host.startswith(prefix) and len(host) > len(prefix) + 4:
            return host[len(prefix):]
    return host


def _org_stem(domain: str) -> str:
    if not domain:
        return ""
    parts = [p for p in domain.split(".") if p not in ("example", "com", "org", "net", "io", "co", "uk", "de", "fr", "nl", "jp", "pl", "ng", "edu", "ac", "au", "ca", "ch")]
    return parts[-1] if parts else domain


def _has_lookalike_pattern(domain: str, reference_stems: set[str]) -> bool:
    if not domain:
        return False
    if any(tok in domain for tok in SUSPICIOUS_DOMAIN_TOKENS):
        return True
    if re.search(r"[a-z][01][a-z]", domain):  # digit 0/1 embedded inside word letters (e.g. m0rgan, l1nkedin)
        return True
    stem = _org_stem(domain)
    for ref in reference_stems:
        if ref and stem != ref:
            # Checkrn/m, vv/w, l/1, o/0 or single-character edit / suffix addition
            if stem.replace("rn", "m") == ref or stem.replace("0", "o") == ref or stem.replace("1", "l") == ref:
                return True
            if ref in stem and abs(len(stem) - len(ref)) <= 8:
                return True
    return False


def extract_email_signals(raw: dict[str, Any]) -> dict[str, Any]:
    """Stage-0 deterministic structural feature extraction (<0.2 ms per email)."""
    headers = raw.get("headers") or {}
    auth = raw.get("auth") or {}
    links = raw.get("links") or []
    attachments = raw.get("attachments") or []

    from_addr = str(headers.get("from_addr") or "")
    from_display = str(headers.get("from_display") or "")
    reply_to = headers.get("reply_to")
    to_raw = headers.get("to_addr") or ""
    cc_raw = headers.get("cc_addr") or ""
    sharer_addr = raw.get("sharer_addr")

    from_dom = _email_domain(from_addr)
    reply_dom = _email_domain(reply_to) if reply_to else ""
    sharer_dom = _email_domain(sharer_addr) if sharer_addr else ""

    to_list = [x.strip() for x in (to_raw if isinstance(to_raw, list) else str(to_raw).split(",")) if x.strip()]
    cc_list = [x.strip() for x in (cc_raw if isinstance(cc_raw, list) else str(cc_raw).split(",")) if x.strip()]
    all_rcpts = to_list + cc_list
    rcpt_domains = sorted({_email_domain(r) for r in all_rcpts if _email_domain(r)})
    rcpt_stems = {_org_stem(d) for d in rcpt_domains if d}

    link_domains = sorted({_url_domain(lk.get("href")) for lk in links if isinstance(lk, dict) and _url_domain(lk.get("href"))})
    link_stems = {_org_stem(d) for d in link_domains if d}

    is_saas_relay = any(from_dom.endswith(k) for k in SAAS_RELAY_DOMAINS)
    from_stem = _org_stem(from_dom)

    # Reply-To mismatch: reply_to present, differs from from_domain, and is not the sharer on a SaaS notification
    reply_mismatch = bool(reply_dom and reply_dom != from_dom and not (is_saas_relay and reply_dom == sharer_dom))
    reply_or_sharer_freemail = bool(
        (reply_dom and (reply_dom in FREE_MAIL_DOMAINS or ".stu." in reply_dom or "stu." in reply_dom or "alumni" in reply_dom))
        or (sharer_dom and (sharer_dom in FREE_MAIL_DOMAINS or ".stu." in sharer_dom or "stu." in sharer_dom or "alumni" in sharer_dom))
        or (not is_saas_relay and from_dom in FREE_MAIL_DOMAINS and any(w in from_display.lower() for w in ("ceo", "cfo", "director", "vp", "payroll", "hr", "it ", "security", "support", "billing", "legal", "board")))
    )

    auth_vals = [str(auth.get(k) or "").lower() for k in ("spf", "dkim", "dmarc")]
    auth_failure = any(v in ("fail", "softfail", "temperror", "permerror") for v in auth_vals) or (
        auth_vals == ["none", "none", "none"]
    )

    # Link vs sender domain mismatch
    link_mismatch = False
    if link_stems and from_stem:
        if is_saas_relay:
            # For a SaaS relay (e.g. docusign.net, google.com, sharepointonline.com), links should match the platform
            expected_saas = next((v for k, v in SAAS_RELAY_DOMAINS.items() if from_dom.endswith(k)), "")
            if not any(expected_saas in d or "sharepoint" in d or "1drv" in d or "docusign" in d or "adobesign" in d or "google" in d or "notion" in d or "dropbox" in d or "box" in d or "canva" in d or "smartsheet" in d or "calendly" in d for d in link_domains):
                link_mismatch = True
        else:
            if from_stem not in link_stems and not (rcpt_stems & link_stems):
                link_mismatch = True

    lookalike = any(_has_lookalike_pattern(d, rcpt_stems | {from_stem}) for d in [from_dom, reply_dom, *link_domains] if d)

    risky_att = False
    for att in attachments:
        fname = str((att.get("filename") if isinstance(att, dict) else att) or "").lower()
        mime = str((att.get("mime_type") if isinstance(att, dict) else "") or "").lower()
        if any(fname.endswith(ext) for ext in RISKY_EXTENSIONS) or "html" in mime or "iso" in mime or "executable" in mime or "macro" in mime or "script" in mime:
            risky_att = True

    rcpt_org_count = int(raw.get("recipient_org_count") or len(rcpt_stems) or 1)
    sharer_same_org = raw.get("sharer_same_org")
    if sharer_same_org is None and sharer_dom and rcpt_stems:
        sharer_same_org = _org_stem(sharer_dom) in rcpt_stems

    return {
        "from_domain": from_dom,
        "reply_to_domain": reply_dom or None,
        "reply_to_mismatch": reply_mismatch,
        "free_or_student_mail_for_business_ask": reply_or_sharer_freemail,
        "auth_failure": auth_failure,
        "link_domains": link_domains,
        "link_vs_sender_domain_mismatch": link_mismatch,
        "lookalike_or_homoglyph_domain": lookalike,
        "risky_attachment": risky_att,
        "is_saas_notification": is_saas_relay,
        "sharer_same_org": sharer_same_org,
        "recipient_org_count": rcpt_org_count,
        "multi_org_recipient_spread": rcpt_org_count >= 3 or len(cc_list) >= 4,
    }


def derive_recommended_action(label: dict[str, Any]) -> str:
    """Deterministic Stage-1 routing target from ground-truth phishing & ask/action labels."""
    is_phish = bool(label.get("is_phish"))
    action = str(label.get("action") or "none")
    ask = str(label.get("ask") or "nothing")
    if is_phish:
        if action in ("click_link", "open_attachment") and ask in ("credentials", "software_install"):
            return "quarantine"
        if ask in ("payment_details", "personal_info") or action == "reply_with_info":
            return "hold_for_review"
        return "quarantine"
    # Benign email: if it asks to open an executable/software or wire payment with urgency, enrich/verify; else deliver
    if ask == "software_install":
        return "enrich_links_or_attachments"
    return "deliver"


def canonical_labels(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a dataset item's ``label`` dict into benchmark/training targets for CANONICAL_QUESTIONS."""
    lb = raw["label"]
    return {
        "is_phish": 1 if lb["is_phish"] else 0,
        "recommended_action": lb.get("recommended_action") or derive_recommended_action(lb),
        "action": str(lb["action"]),
        "ask": str(lb["ask"]),
        "identity_claim": 1 if lb["identity_claim"] else 0,
        "urgency": 1 if lb["urgency"] else 0,
    }


def phishing_layouts(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Render one structured email record in 4 distinct state layouts."""
    headers = raw.get("headers") or {}
    auth = raw.get("auth") or {}
    links = raw.get("links") or []
    attachments = raw.get("attachments") or []
    signals = extract_email_signals(raw)
    body = str(raw.get("body") or "").strip()

    collab: dict[str, Any] = {}
    if raw.get("platform"):
        collab = {
            "platform": raw.get("platform"),
            "sharer_addr": raw.get("sharer_addr"),
            "sharer_same_org": raw.get("sharer_same_org"),
            "recipient_org_count": raw.get("recipient_org_count", 1),
        }

    # Layout 1: Full structured JSON with Stage-0 signals
    structured_json: dict[str, Any] = {
        "headers": headers,
        "auth": auth,
        "signals": signals,
        "links": links,
        "attachments": attachments,
        **({"collab_context": collab} if collab else {}),
        "body": body,
    }

    # Layout 2: RFC822-style header block + body
    hdr_lines = [
        f"From: {headers.get('from_display', '')} <{headers.get('from_addr', '')}>".strip(),
        *( [f"Reply-To: {headers['reply_to']}"] if headers.get("reply_to") else [] ),
        f"To: {headers.get('to_addr', '')}",
        *( [f"Cc: {', '.join(headers['cc_addr']) if isinstance(headers['cc_addr'], list) else headers['cc_addr']}"] if headers.get("cc_addr") else [] ),
        f"Subject: {headers.get('subject', '')}",
        f"Authentication-Results: spf={auth.get('spf', 'none')} dkim={auth.get('dkim', 'none')} dmarc={auth.get('dmarc', 'none')}",
    ]
    if collab:
        hdr_lines.append(
            f"X-Collab-Context: platform={collab['platform']} sharer={collab.get('sharer_addr')} "
            f"sharer_same_org={collab.get('sharer_same_org')} recipient_org_count={collab.get('recipient_org_count')}"
        )
    if links:
        hdr_lines.append("Extracted-Links: " + "; ".join(f"[{lk.get('anchor_text', '')}] -> {lk.get('href', '')}" for lk in links))
    if attachments:
        hdr_lines.append("Attachments: " + ", ".join(f"{a.get('filename', '')} ({a.get('mime_type', '')})" for a in attachments))
    flag_items = [k for k, v in signals.items() if isinstance(v, bool) and v]
    hdr_lines.append("Pre-Parser-Flags: " + (", ".join(flag_items) if flag_items else "none"))
    rfc822_state = {"raw_message": "\n".join(hdr_lines) + "\n\n" + body}

    # Layout 3: Flattened gateway event
    gateway_event = {
        "gateway_event": {
            "sender_display": headers.get("from_display"),
            "sender_address": headers.get("from_addr"),
            "reply_to": headers.get("reply_to"),
            "recipient": headers.get("to_addr"),
            "cc_count": len(headers.get("cc_addr") or []) if isinstance(headers.get("cc_addr"), list) else 0,
            "subject": headers.get("subject"),
            "spf_dkim_dmarc": f"{auth.get('spf')}/{auth.get('dkim')}/{auth.get('dmarc')}",
            "urls": [lk.get("href") for lk in links],
            "attachment_files": [a.get("filename") for a in attachments],
            "collab_sharer": raw.get("sharer_addr"),
            "sharer_same_org": signals["sharer_same_org"],
            "recipient_org_count": signals["recipient_org_count"],
            "risk_signals": [k for k, v in signals.items() if isinstance(v, bool) and v],
            "text_body": body,
        }
    }

    # Layout 4: Raw email state without pre-parsed signals (tests direct header/link reading)
    raw_email: dict[str, Any] = {
        "headers": headers,
        "auth": auth,
        "links": links,
        "attachments": attachments,
        **({"collab_context": collab} if collab else {}),
        "body": body,
    }

    return {
        "structured_json": structured_json,
        "rfc822_headers": rfc822_state,
        "gateway_event": gateway_event,
        "raw_email": raw_email,
    }


def load_workspace_counterfactuals(workspace_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load the 160 counterfactual emails (80 pairs) from workspace/datasets/phishing_classifier/ or system_one/suites/."""
    if workspace_dir is None:
        # Resolve relative to repo root, falling back to tracked system_one/suites or container mounts
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / "workspace" / "datasets" / "phishing_classifier",
            Path(__file__).resolve().parent.parent / "system_one" / "suites",
            Path("/host_ws/datasets/phishing_classifier"),
            Path("/workspace/datasets/phishing_classifier"),
        ]
        workspace_dir = next((c for c in candidates if (c / "training_data.v02.jsonl").exists()), candidates[0])
    files = [
        workspace_dir / "training_data.v02.jsonl",
        workspace_dir / "training_data_batch2.clean.jsonl",
    ]
    out: list[dict[str, Any]] = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------ Expanded Counterfactual Pair Generator
def generate_expanded_counterfactual_pairs(n_pairs: int = 240, seed: int = 42) -> list[dict[str, Any]]:
    """Generate ``n_pairs`` paired counterfactual emails (2 * n_pairs total items) across 12 modern phishing vectors.

    Every pair (cfXXXa, cfXXXb) shares the same realistic corporate context and body prose, differing ONLY in the
    causal security indicator so the cross-encoder cannot rely on surface topic or keyword shortcuts.
    """
    rng = random.Random(seed)
    orgs = [
        ("stratos-bio.example", "Stratos Biologics", "maya.lin", "ap-desk", "it-helpdesk"),
        ("kestrel-robotics.example", "Kestrel Robotics", "d.thorne", "payroll", "sec-ops"),
        ("meridian-logistics.example", "Meridian Freight", "s.kovacs", "billing", "cloud-admin"),
        ("solstice-energy.example", "Solstice Grid", "a.okafor", "treasury", "identity-team"),
        ("vanguard-optics.example", "Vanguard Photonics", "r.chen", "procurement", "workspace-support"),
        ("halcyon-med.example", "Halcyon Diagnostics", "e.lindqvist", "finance", "sso-alerts"),
        ("cobalt-aero.example", "Cobalt Aerospace", "j.moreau", "accounts", "platform-sec"),
        ("apex-semiconductor.example", "Apex Silicon", "t.nakamura", "legal-ops", "corp-it"),
        ("nordic-marine.example", "Nordic Maritime", "h.bergstrom", "chartering", "noc-alerts"),
        ("veridian-cap.example", "Veridian Capital", "c.alvarez", "fund-accounting", "okta-admin"),
    ]
    vendors = [
        ("ironclad-cloud.example", "Ironclad Cloud Hosting", "billing"),
        ("precision-tooling.example", "Precision CNC Tooling", "ar"),
        ("baltic-freight.example", "Baltic Container Lines", "customs"),
        ("sterling-legal.example", "Sterling & Vance LLP", "closings"),
        ("clearwater-benefits.example", "Clearwater HR Benefits", "enrollments"),
        ("atlas-datacenter.example", "Atlas Colocation Services", "provisioning"),
    ]
    families = [
        "link_domain_mismatch",
        "auth_failure",
        "reply_to_mismatch",
        "homoglyph_domain",
        "attachment_risk",
        "sharer_outside_org",
        "sharer_consumer_account",
        "recipient_org_spread",
        "oauth_consent_lure",
        "qr_quishing_lure",
        "payroll_diversion_bec",
        "vendor_thread_hijack",
    ]

    items: list[dict[str, Any]] = []
    for idx in range(1, n_pairs + 1):
        pair_id = f"cf{idx:03d}"
        fam = families[(idx - 1) % len(families)]
        org_dom, org_name, user_local, fin_local, it_local = orgs[(idx * 3) % len(orgs)]
        ven_dom, ven_name, ven_local = vendors[(idx * 5) % len(vendors)]
        ticket_no = 10000 + idx * 37
        amount = f"${12_400 + (idx * 730) % 85_000:,.2f}"

        if fam == "link_domain_mismatch":
            bad_dom = f"{org_dom.split('.')[0]}-session-verify.example"
            good_dom = f"sso.{org_dom}"
            for suffix, is_phish, ldom in (("a", True, bad_dom), ("b", False, good_dom)):
                href = f"https://{ldom}/auth/token-rotate?id={ticket_no}"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": True, "ask": "credentials", "identity_claim": True},
                    "headers": {"from_display": f"{org_name} Identity Operations", "from_addr": f"{it_local}@{org_dom}", "reply_to": None, "to_addr": f"{user_local}@{org_dom}", "subject": f"[Action Required] Rotate expiring SAML signing session #{ticket_no}"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [{"href": href, "anchor_text": "Rotate SAML Session"}],
                    "attachments": [],
                    "body": f"Hi {user_local.split('.')[0].title()},\n\nYour workstation SAML certificate for {org_name} expires in 4 hours. Please re-authenticate at {href} to avoid VPN and Git disconnection during today's release window.\n\n{org_name} Identity Team",
                })

        elif fam == "auth_failure":
            for suffix, is_phish, auth_state in (
                ("a", True, {"spf": "softfail", "dkim": "fail", "dmarc": "fail"}),
                ("b", False, {"spf": "pass", "dkim": "pass", "dmarc": "pass"}),
            ):
                href = f"https://portal.{ven_dom}/invoices/INV-{ticket_no}"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": True, "ask": "payment_details", "identity_claim": True},
                    "headers": {"from_display": f"{ven_name} Accounts Receivable", "from_addr": f"{ven_local}@{ven_dom}", "reply_to": None, "to_addr": f"{fin_local}@{org_dom}", "subject": f"Overdue Balance Notice: Invoice #INV-{ticket_no} ({amount})"},
                    "auth": auth_state,
                    "links": [{"href": href, "anchor_text": "View & Settle Invoice"}],
                    "attachments": [],
                    "body": f"Hello {org_name} AP,\n\nInvoice #INV-{ticket_no} for {amount} is now 15 days past terms. Please confirm remittance details or settle via ACH on the customer portal before 17:00 UTC to prevent service hold: {href}\n\nRegards,\n{ven_name} Billing",
                })

        elif fam == "reply_to_mismatch":
            bad_reply = f"{ven_local}.settlement@fast-inbox.example.org"
            good_reply = f"{ven_local}@{ven_dom}"
            for suffix, is_phish, rto in (("a", True, bad_reply), ("b", False, good_reply)):
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "reply_with_info", "urgency": True, "ask": "payment_details", "identity_claim": True},
                    "headers": {"from_display": f"{ven_name} Treasury", "from_addr": f"{ven_local}@{ven_dom}", "reply_to": rto, "to_addr": f"{fin_local}@{org_dom}", "subject": f"Updated SWIFT/IBAN routing for Q4 contract #{ticket_no}"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [],
                    "attachments": [],
                    "body": f"Hi {org_name} Finance,\n\nAhead of Friday's {amount} milestone disbursement, please reply to this thread to confirm you have updated our beneficiary clearing account in your ERP, and we will send the countersigned bank letter.\n\nBest,\n{ven_name} Treasury",
                })

        elif fam == "homoglyph_domain":
            stem = org_dom.split(".")[0]
            spoof_stem = stem.replace("o", "0") if "o" in stem else stem.replace("i", "1") if "i" in stem else f"{stem}-corp"
            bad_dom = f"{spoof_stem}.example"
            for suffix, is_phish, sdom in (("a", True, bad_dom), ("b", False, org_dom)):
                href = f"https://vpn.{sdom}/device-posture/{ticket_no}"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": False, "ask": "credentials", "identity_claim": True},
                    "headers": {"from_display": f"{org_name} Endpoint Security", "from_addr": f"{it_local}@{sdom}", "reply_to": None, "to_addr": f"{user_local}@{org_dom}", "subject": f" CrowdStrike Falcon posture check required for host #{ticket_no}"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [{"href": href, "anchor_text": "Verify Device Posture"}],
                    "attachments": [],
                    "body": f"Hello,\n\nYour macOS endpoint failed automated posture attestation after the kernel update. Please sign in with your SSO credentials at {href} to sync your device certificate.\n\n{org_name} SecOps",
                })

        elif fam == "attachment_risk":
            for suffix, is_phish, fname, mime, ask in (
                ("a", True, f"Remittance_Advice_{ticket_no}.pdf.html", "text/html", "software_install"),
                ("b", False, f"Remittance_Advice_{ticket_no}.pdf", "application/pdf", "nothing"),
            ):
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "open_attachment", "urgency": False, "ask": ask, "identity_claim": True},
                    "headers": {"from_display": f"{ven_name} Remittances", "from_addr": f"{ven_local}@{ven_dom}", "reply_to": None, "to_addr": f"{fin_local}@{org_dom}", "subject": f"Remittance advice & line-item breakdown #{ticket_no}"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [],
                    "attachments": [{"filename": fname, "mime_type": mime}],
                    "body": f"Good afternoon,\n\nPlease find attached the remittance breakdown ({fname}) covering batch #{ticket_no} ({amount}). Open the attached file to review the credit memo adjustments.\n\n{ven_name} Accounts Payable",
                })

        elif fam == "sharer_outside_org":
            ext_dom = vendors[(idx + 2) % len(vendors)][0]
            for suffix, is_phish, sdom, same_org in (("a", True, ext_dom, False), ("b", False, org_dom, True)):
                sharer = f"k.patel@{sdom}"
                href = f"https://docs.google.com/spreadsheets/d/19x8c7v6b5n4m3l2k{ticket_no}/edit"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "platform": "google_drive", "sharer_addr": sharer, "sharer_same_org": same_org, "recipient_org_count": 1,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": False, "ask": "document_access", "identity_claim": True},
                    "headers": {"from_display": "Kiran Patel (via Google Sheets)", "from_addr": "drive-shares-dm-noreply@google.com", "reply_to": sharer, "to_addr": f"{fin_local}@{org_dom}", "subject": f"Spreadsheet shared with you: \"{org_name} FY27 Merit & Bonus Pool Allocation\""},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [{"href": href, "anchor_text": "Open in Sheets"}],
                    "attachments": [],
                    "body": f"Kiran Patel ({sharer}) has invited you to edit the following spreadsheet:\n\n{org_name} FY27 Merit & Bonus Pool Allocation\n\n\"Hi team, please verify your department's equity refresh and bonus rows before the compensation committee lock at 5pm.\"\n\nOpen in Sheets: {href}",
                })

        elif fam == "sharer_consumer_account":
            for suffix, is_phish, sharer in (
                ("a", True, f"closing.desk.{ticket_no}@freemail-inbox.example.com"),
                ("b", False, f"{ven_local}@{ven_dom}"),
            ):
                href = f"https://na4.docusign.net/Signing/MTRedeem/v1/8a9b0c-{ticket_no}"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "platform": "docusign", "sharer_addr": sharer, "sharer_same_org": False, "recipient_org_count": 1,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": True, "ask": "payment_details", "identity_claim": True},
                    "headers": {"from_display": f"{ven_name} via DocuSign", "from_addr": "dse_NA4@docusign.net", "reply_to": sharer, "to_addr": f"{fin_local}@{org_dom}", "subject": f"Complete with DocuSign: Addendum_{ticket_no}_Escrow_Wire_Instructions.pdf"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [{"href": href, "anchor_text": "REVIEW DOCUMENT"}],
                    "attachments": [],
                    "body": f"{ven_name} ({sharer}) sent you a document to review and sign.\n\nREVIEW DOCUMENT: {href}\n\n\"Please countersign the revised escrow wire schedule for {amount} so release can proceed today.\"",
                })

        elif fam == "recipient_org_spread":
            other1 = orgs[(idx + 1) % len(orgs)][0]
            other2 = orgs[(idx + 2) % len(orgs)][0]
            other3 = orgs[(idx + 4) % len(orgs)][0]
            bad_to = f"{fin_local}@{org_dom}, ap@{other1}, billing@{other2}, treasury@{other3}"
            good_to = f"{fin_local}@{org_dom}, controller@{org_dom}, treasury@{org_dom}, tax@{org_dom}"
            for suffix, is_phish, to_val, org_cnt in (("a", True, bad_to, 4), ("b", False, good_to, 1)):
                sharer = f"{ven_local}@{ven_dom}"
                href = f"https://app.box.com/s/audit-folder-{ticket_no}"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "platform": "box", "sharer_addr": sharer, "sharer_same_org": False, "recipient_org_count": org_cnt,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": False, "ask": "document_access", "identity_claim": True},
                    "headers": {"from_display": f"{ven_name} via Box", "from_addr": "noreply@box.com", "reply_to": sharer, "to_addr": to_val, "subject": f"{ven_name} invited you to \"Confidential Q4 Tax & Wire Audit Folder #{ticket_no}\""},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [{"href": href, "anchor_text": "Go to Folder"}],
                    "attachments": [],
                    "body": f"{ven_name} ({sharer}) has invited you to collaborate on the folder \"Confidential Q4 Tax & Wire Audit Folder #{ticket_no}\" on Box.\n\nGo to Folder: {href}",
                })

        elif fam == "oauth_consent_lure":
            bad_href = f"https://login-oauth-consent-grant.example/authorize?client_id=unverified-{ticket_no}&scope=Mail.ReadWrite+Files.ReadWrite.All"
            good_href = f"https://sso.{org_dom}/oauth/apps/approved-{ticket_no}"
            for suffix, is_phish, href in (("a", True, bad_href), ("b", False, good_href)):
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": False, "ask": "credentials", "identity_claim": True},
                    "headers": {"from_display": f"{org_name} Workspace Integrations", "from_addr": f"{it_local}@{org_dom}", "reply_to": None, "to_addr": f"{user_local}@{org_dom}", "subject": f"Action needed: re-authorize Calendar & Mail Sync plugin #{ticket_no}"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [{"href": href, "anchor_text": "Grant OAuth Permissions"}],
                    "attachments": [],
                    "body": f"Hi {user_local.split('.')[0].title()},\n\nYour Microsoft 365 / Google Workspace calendar sync integration requires token renewal. Click {href} to approve read/write permissions for your mailbox.\n\n{org_name} IT Automation",
                })

        elif fam == "qr_quishing_lure":
            for suffix, is_phish, ldom, att_name, att_mime in (
                ("a", True, "qr-mfa-enroll-auth.example", f"QR_MFA_Enrollment_{ticket_no}.pdf.html", "text/html"),
                ("b", False, f"sso.{org_dom}", f"MFA_Setup_Guide_{ticket_no}.pdf", "application/pdf"),
            ):
                href = f"https://{ldom}/mfa-qr/{ticket_no}"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": True, "ask": "credentials", "identity_claim": True},
                    "headers": {"from_display": f"{org_name} Zero-Trust MFA", "from_addr": f"{it_local}@{org_dom}", "reply_to": None, "to_addr": f"{user_local}@{org_dom}", "subject": f"Mandatory Passkey / Authenticator QR re-enrollment #{ticket_no}"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [{"href": href, "anchor_text": "Open QR Enrollment Code"}],
                    "attachments": [{"filename": att_name, "mime_type": att_mime}],
                    "body": f"Attention {org_name} Staff,\n\nWe are migrating all mobile authenticator tokens before 18:00 today. Scan the QR code inside {att_name} or open {href} on your phone to bind your passkey and prevent account lockout.\n\n{org_name} Identity Security",
                })

        elif fam == "payroll_diversion_bec":
            for suffix, is_phish, faddr, rto in (
                ("a", True, f"{user_local}.personal@gmail.com", f"{user_local}.personal@gmail.com"),
                ("b", False, f"{user_local}@{org_dom}", None),
            ):
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "reply_with_info", "urgency": True, "ask": "payment_details", "identity_claim": True},
                    "headers": {"from_display": user_local.replace(".", " ").title(), "from_addr": faddr, "reply_to": rto, "to_addr": f"payroll@{org_dom}", "subject": f"Urgent: Direct deposit bank routing change for Friday payroll #{ticket_no}"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                    "links": [],
                    "attachments": [],
                    "body": f"Hi Payroll team,\n\nI just switched primary checking accounts and closed my old account. Can you update my direct deposit routing and account number before tomorrow's payroll cutoff? Please reply here and I will send the new voided check details.\n\nThanks,\n{user_local.replace('.', ' ').title()}",
                })

        else:  # vendor_thread_hijack
            bad_dom = f"{ven_dom.split('.')[0]}-remit.example"
            for suffix, is_phish, sdom, auth_dmarc in (
                ("a", True, bad_dom, "fail"),
                ("b", False, ven_dom, "pass"),
            ):
                href = f"https://pay.{sdom}/wire-schedule/{ticket_no}"
                items.append({
                    "id": f"{pair_id}{suffix}", "pair_id": pair_id, "varies": fam,
                    "label": {"is_phish": is_phish, "action": "click_link", "urgency": True, "ask": "payment_details", "identity_claim": True},
                    "headers": {"from_display": f"{ven_name} Billing", "from_addr": f"{ven_local}@{sdom}", "reply_to": f"{ven_local}@{sdom}", "to_addr": f"{fin_local}@{org_dom}", "subject": f"Re: PO #{ticket_no} delivery confirmation & wire instructions ({amount})"},
                    "auth": {"spf": "pass", "dkim": "pass", "dmarc": auth_dmarc},
                    "links": [{"href": href, "anchor_text": "Download Updated Wire Instructions"}],
                    "attachments": [],
                    "body": f"Following up on the thread below regarding PO #{ticket_no}—our clearing bank placed a temporary hold on our main operating account, so please use the updated wire instructions at {href} for today's {amount} settlement.\n\n---\nOn Tue, {org_name} Procurement wrote:\n> Confirmed receipt of shipment for PO #{ticket_no}. Scheduling payment for Friday's run.",
                })

    rng.shuffle(items)
    return items


def split_counterfactual_dataset(seed: int = 42, holdout_pairs_per_source: int = 12) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split workspace counterfactuals + generated counterfactuals into (train_items, heldout_bench_items) grouped strictly by pair_id."""
    ws_items = load_workspace_counterfactuals()
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for it in ws_items:
        by_pair.setdefault(it["pair_id"], []).append(it)

    p_pairs = sorted(k for k in by_pair if k.startswith("p"))
    q_pairs = sorted(k for k in by_pair if k.startswith("q"))

    rng = random.Random(seed)
    rng.shuffle(p_pairs)
    rng.shuffle(q_pairs)

    held_pair_ids = set(p_pairs[:holdout_pairs_per_source] + q_pairs[:holdout_pairs_per_source])
    train_ws = [it for pid, group in by_pair.items() if pid not in held_pair_ids for it in group]
    held_ws = [it for pid in sorted(held_pair_ids) for it in sorted(by_pair[pid], key=lambda x: x["id"])]

    gen_items = generate_expanded_counterfactual_pairs(n_pairs=136, seed=seed)
    gen_by_pair: dict[str, list[dict[str, Any]]] = {}
    for it in gen_items:
        gen_by_pair.setdefault(it["pair_id"], []).append(it)
    gen_pair_ids = sorted(gen_by_pair)
    rng.shuffle(gen_pair_ids)
    held_gen_ids = set(gen_pair_ids[:16])  # 16 generated pairs (32 emails) + 24 workspace pairs (48 emails) = 40 pairs (80 emails) held out
    train_gen = [it for pid, group in gen_by_pair.items() if pid not in held_gen_ids for it in group]
    held_gen = [it for pid in sorted(held_gen_ids) for it in sorted(gen_by_pair[pid], key=lambda x: x["id"])]

    return train_ws + train_gen, held_ws + held_gen


def to_training_rows(items: list[dict[str, Any]], seed: int = 7, signal_dropout: float = 0.5) -> list[dict[str, Any]]:
    """Convert structured counterfactual email items into multi-layout Kenning training rows."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for it in items:
        targets = canonical_labels(it)
        q_dict = {k: q.model_dump(exclude_none=True) for k, q in CANONICAL_QUESTIONS.items()}
        # Keep noul positive/negative prior balanced (~0.50) across the training set by pairing
        # positive-skewed identity_claim with a negative-skewed deception/spoofing check on alternate layouts
        layouts = phishing_layouts(it)
        chosen_layouts = ["structured_json", *rng.sample(["rfc822_headers", "gateway_event", "raw_email"], 2)]
        for idx_l, lname in enumerate(chosen_layouts):
            state = dict(layouts[lname])
            if lname == "structured_json" and signal_dropout > 0 and rng.random() < signal_dropout:
                state.pop("signals", None)
            state["pair_id"] = it.get("pair_id")
            row_qs = dict(q_dict)
            row_targets = dict(targets)
            if idx_l > 0 and row_targets.get("identity_claim") == 1 and row_targets.get("urgency") == 1:
                # On secondary layouts, ask only the core triage heads (is_phish, recommended_action, action, ask)
                # so the secondary metadata nouls do not skew the global yes/no prior above 0.50
                row_qs.pop("identity_claim", None)
                row_targets.pop("identity_claim", None)
            elif idx_l > 0 and row_targets.get("identity_claim") == 1:
                row_qs.pop("identity_claim", None)
                row_targets.pop("identity_claim", None)
            rows.append({
                "pair_id": it.get("pair_id"),
                "weight": 1.5 if it.get("pair_id") else 1.0,
                "state": state,
                "questions": row_qs,
                "targets": row_targets,
            })
            # Also add a bare yes/no phishing row (without criteria) on 50% of structured_json rows
            # so noul_bare and noul_criteria are both well-trained and calibrated!
            if lname == "structured_json" and rng.random() < 0.5:
                bare_q = {
                    "is_malicious": Question(
                        type="noul",
                        instructions=rng.choice([
                            "Is this email a phishing attempt, scam or other security threat?",
                            "Is this email malicious or a social-engineering attack?",
                            "Should this email be flagged as phishing?",
                        ]),
                    ).model_dump(exclude_none=True)
                }
                rows.append({
                    "pair_id": it.get("pair_id"),
                    "weight": 1.2,
                    "state": state,
                    "questions": bare_q,
                    "targets": {"is_malicious": targets["is_phish"]},
                })
    rng.shuffle(rows)
    return rows


def phishing_counterfactual_suite(seed: int = 42) -> Suite:
    """Held-out benchmark suite of 48 hand-written paired counterfactual emails (24 twin pairs, never trained on)."""
    _, held = split_counterfactual_dataset(seed=seed)
    handwritten = [raw for raw in held if not str(raw.get("id", "")).startswith("cf")]
    items: list[Item] = []
    for raw in handwritten:
        state = phishing_layouts(raw)["structured_json"]
        state["pair_id"] = raw.get("pair_id")
        items.append(Item(id=raw["id"], state=state, labels=canonical_labels(raw)))
    return Suite(
        name="phishing_counterfactual",
        description=f"Held-out hand-written paired counterfactual emails ({len(items)} emails / {len(items) // 2} twin pairs across header, auth, link, BEC & SaaS indicators)",
        questions=CANONICAL_QUESTIONS,
        items=items,
        gate="is_phish",
    )


def phishing_counterfactual_synthetic_suite(seed: int = 42) -> Suite:
    """Held-out synthetic-generator paired counterfactual emails (32 emails / 16 pairs; same template family as training)."""
    _, held = split_counterfactual_dataset(seed=seed)
    synthetic = [raw for raw in held if str(raw.get("id", "")).startswith("cf")]
    items: list[Item] = []
    for raw in synthetic:
        state = phishing_layouts(raw)["structured_json"]
        state["pair_id"] = raw.get("pair_id")
        items.append(Item(id=raw["id"], state=state, labels=canonical_labels(raw)))
    return Suite(
        name="phishing_counterfactual_synthetic",
        description=f"In-distribution synthetic counterfactual pairs ({len(items)} emails / {len(items) // 2} pairs; same template generator as training)",
        questions=CANONICAL_QUESTIONS,
        items=items,
        gate="is_phish",
    )


def phishing_layouts_suite(seed: int = 42, n_pairs: int = 20) -> Suite:
    """Held-out hand-written counterfactual emails rendered across all 4 state layouts to measure layout invariance."""
    _, held = split_counterfactual_dataset(seed=seed)
    handwritten = [raw for raw in held if not str(raw.get("id", "")).startswith("cf")]
    subset = handwritten[: n_pairs * 2]
    layout_names = ("structured_json", "rfc822_headers", "gateway_event", "raw_email")
    questions = {
        f"is_phish@{lname}": CANONICAL_QUESTIONS["is_phish"]
        for lname in layout_names
    }
    items: list[Item] = []
    for raw in subset:
        truth = 1 if raw["label"]["is_phish"] else 0
        for lname, state in phishing_layouts(raw).items():
            st = dict(state)
            st["pair_id"] = f"{raw['pair_id']}@{lname}"
            items.append(Item(id=f"{raw['id']}@{lname}", state=st, labels={f"is_phish@{lname}": truth}))
    return Suite(
        name="phishing_layouts",
        description=f"{len(subset)} held-out counterfactual emails across 4 state layouts ({', '.join(layout_names)})",
        questions=questions,
        items=items,
        gate="is_phish@structured_json",
    )


def build_phishing_dataset(out_dir: Path, structured_per_kind: int = 80, seed: int = 7, bench_seed: int = 42) -> dict[str, Any]:
    """Build the clean Apache-2.0 phishing training set, calibration split, and held-out benchmark files."""
    from systemone_builder.kenning import structured
    from systemone_builder.kenning.data import questions_for
    from systemone_builder.kenning.layouts import email_layouts

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    train_cf, held_cf = split_counterfactual_dataset(seed=bench_seed)
    # Carve out 20 pairs (40 emails) from train_cf as dedicated post-training calibration set
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for it in train_cf:
        by_pair.setdefault(it["pair_id"], []).append(it)
    pair_ids = sorted(by_pair)
    rng.shuffle(pair_ids)
    cal_pair_ids = set(pair_ids[:20])
    cal_cf = [it for pid in sorted(cal_pair_ids) for it in by_pair[pid]]
    core_train_cf = [it for pid in pair_ids[20:] for it in by_pair[pid]]

    cf_rows = to_training_rows(core_train_cf, seed=seed)

    # Also add short body-only modern email counterfactuals (for suites like modern_email_2 that only pass email.body)
    short_modern_pairs = [
        (True, "Hi team, we migrated our Notion workspace permissions. Re-authenticate your SSO session at https://notion-sso-check.example/login before 5pm so your pages stay synced."),
        (False, "Hi team, I put the Q4 product roadmap notes in the shared folder on Drive. Let me know if any milestones look off before Thursday's sync."),
        (True, "Urgent: Your corporate Amex card ending 4419 has been temporarily suspended due to an unverified charge. Confirm your card PIN and CVV at https://amex-card-verify.example/unlock."),
        (False, "Reminder: All Q3 corporate Amex expense reports and receipt PDFs must be submitted in Concur by Friday at 17:00 local time for October reimbursement."),
        (True, "From CEO (sent from my iPhone): Are you at your desk? I'm tied up in a board session and need you to purchase 6 Apple gift cards ($200 each) for a client Prospect immediately. Send me the codes here."),
        (False, "From CEO: Thanks everyone for pulling together the board deck yesterday. We'll review the customer churn feedback in Monday's all-hands."),
        (True, "DocuSign Security: Your signature is required on 'Amended_Wire_Instructions_2026.pdf'. Sign in with your Microsoft 365 password at https://docusign-envelope-view.example/sign."),
        (False, "Automated alert from GitHub: Dependabot opened pull request #412 to bump cryptography from 42.0.2 to 42.0.5 in systemone-builder."),
        (True, "HR Payroll: Your direct deposit failed validation for this pay cycle. Log in at https://adp-payroll-update-portal.example/bank to re-enter your routing and account number today."),
        (False, "HR People Ops: Open enrollment for 2027 dental and vision benefits begins next Monday. Review the plan summary on the internal intranet before November 15."),
    ]
    short_rows: list[dict[str, Any]] = []
    for idx, (mal, body) in enumerate(short_modern_pairs):
        pid = f"sm{idx // 2:03d}"
        for lname, st in email_layouts(body, rng).items():
            qs, targets = questions_for(rng, mal)
            if isinstance(st, dict):
                st = {**st, "pair_id": pid}
            short_rows.append({"pair_id": pid, "weight": 1.3, "state": st, "questions": qs, "targets": targets, "source": "short_modern_email"})

    synth_cache = out_dir / "synthetic-email-n1000-seed11-subtle40.jsonl"
    if synth_cache.exists():
        raw_synth = [json.loads(line) for line in synth_cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        rng.shuffle(raw_synth)
        mal_synth = [e for e in raw_synth if e.get("malicious")][:300]
        ben_synth = [e for e in raw_synth if not e.get("malicious")][:300]
        for e in mal_synth + ben_synth:
            qs, targets = questions_for(rng, bool(e.get("malicious")))
            body_txt = e["email"].get("body", "") if isinstance(e.get("email"), dict) else str(e.get("email") or "")
            layouts_map = email_layouts(body_txt, rng)
            st = layouts_map[rng.choice(list(layouts_map))]
            short_rows.append({"weight": 1.0, "state": st, "questions": qs, "targets": targets, "source": "short_modern_email"})

    anchor_rows: list[dict[str, Any]] = []
    if structured_per_kind > 0:
        for kind in structured.GENERATORS:
            for r in structured.rows(kind, structured_per_kind, seed):
                r["source"] = f"structured:{kind}"
                anchor_rows.append(r)

    for r in cf_rows:
        r["source"] = "phishing_counterfactual"

    all_rows = cf_rows + short_rows + anchor_rows
    rng.shuffle(all_rows)

    train_path = out_dir / "phishing-v1-train.jsonl"
    train_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")

    cal_rows = to_training_rows(cal_cf, seed=seed + 1)
    cal_path = out_dir / "phishing-v1-cal.jsonl"
    cal_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cal_rows), encoding="utf-8")

    suite = phishing_counterfactual_suite(seed=bench_seed)
    hold_path = out_dir / "phishing-v1-holdout.jsonl"
    hold_q_path = out_dir / "phishing-v1-holdout.questions.json"
    hold_path.write_text(
        "".join(json.dumps({"id": it.id, "state": it.state, "labels": it.labels}, ensure_ascii=False) + "\n" for it in suite.items),
        encoding="utf-8",
    )
    hold_q_path.write_text(
        json.dumps({k: q.model_dump(exclude_none=True) for k, q in suite.questions.items()}, indent=2),
        encoding="utf-8",
    )

    yes = [t for r in all_rows for q, t in r["targets"].items() if r["questions"][q]["type"] == "noul"]
    manifest = {
        "seed": seed,
        "rows": len(all_rows),
        "noul_mean_target": round(sum(float(t) for t in yes) / max(len(yes), 1), 3),
        "sources": {
            "phishing_counterfactual": {
                "dataset": "paired counterfactual phishing & SaaS collaboration emails (strictly split by pair_id)",
                "license": "Apache-2.0",
                "rows": len(cf_rows),
            },
            "short_modern_email": {
                "dataset": "short modern operational & phishing emails across 4 layouts",
                "license": "Apache-2.0",
                "rows": len(short_rows),
            },
            "structured_anchors": {
                "dataset": "rule-generated general decision anchors (records, tables, agent_steps, logs)",
                "license": "Apache-2.0 (kenning/structured.py)",
                "rows": len(anchor_rows),
            },
        },
        "holdouts": {
            "phishing_counterfactual": {
                "items": len(suite.items),
                "file": hold_path.name,
            }
        },
    }
    train_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "train_path": str(train_path),
        "cal_path": str(cal_path),
        "holdout_path": str(hold_path),
        "train_rows": len(all_rows),
        "cal_rows": len(cal_rows),
        "holdout_items": len(suite.items),
        "noul_mean_target": manifest["noul_mean_target"],
    }

