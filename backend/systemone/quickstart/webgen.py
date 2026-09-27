"""Dummy web-automation dataset for the Zero-to-One Quickstart (Module H).

Procedurally generates realistic (page, goal, history) -> next-action samples
covering the reflexes a GUI agent needs: filling logins, searching, cookie
banners, validation-error recovery, navigation, confirmation modals,
checkboxes, canvas targets that need CLICK_XY, and genuinely ambiguous pages
that must ESCALATE. Pages carry volatile noise (timestamps, request ids,
framework ids, cache-busters) so the fuzzy scrubber is exercised too.

Every sample is passed through the real ``computer_use`` extractor, so the
states match production traffic byte-for-byte.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from systemone.domains.builtin import COMPUTER_USE
from systemone.extraction.pipeline import Observation, StateExtractor

SITES = ["acme-crm", "orbit-hr", "northwind-erp", "helios-billing", "lumen-cms", "atlas-ops", "quanta-lab", "vega-support"]
USERS = ["admin", "ops.lead", "jdoe", "analyst", "svc-deploy", "m.garcia"]
QUERIES = ["invoice 4471", "overdue tickets", "Q3 revenue", "user alice", "failed jobs", "renewal contracts"]
SECTIONS = ["Settings", "Reports", "Billing", "Users", "Audit log", "Integrations", "Dashboard", "Profile"]


@dataclass
class Sample:
    observation: Observation
    action: dict[str, Any]
    scenario: str


def _noise(rng: random.Random) -> str:
    ts = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
    return (
        f'<footer><p>Last sync {ts} · request {uuid.UUID(int=rng.getrandbits(128))}</p>'
        f'<span id="ember{rng.randint(100, 9999)}">build {rng.getrandbits(64):016x}</span></footer>'
    )


def _nav(rng: random.Random, k: int = 3) -> str:
    links = rng.sample(SECTIONS, k)
    return "<nav>" + "".join(f'<a href="/{s.lower().replace(" ", "-")}?_={rng.randint(10**12, 10**13)}">{s}</a>' for s in links) + "</nav>"


def _page(site: str, body: str, rng: random.Random, nav: bool = True) -> str:
    return f"<html><head><title>{site}</title><script>window.__s={rng.random()}</script></head><body>{_nav(rng) if nav else ''}{body}{_noise(rng)}</body></html>"


def _ids(obs_state: dict[str, Any], role: str, name_contains: str) -> int:
    for n in obs_state["viewport_tree"]:
        if n["role"] == role and name_contains.lower() in n.get("name", "").lower():
            return n["id"]
    raise LookupError(f"{role} {name_contains!r} not in tree")


# ----------------------------------------------------------------- scenarios
def login(rng: random.Random) -> tuple[Observation, Callable[[dict[str, Any]], dict[str, Any]], str]:
    site, user = rng.choice(SITES), rng.choice(USERS)
    stage = rng.choice(["user", "password", "submit"])
    ulabel, plabel, btn = rng.choice([("Username", "Password", "Sign in"), ("Email", "Password", "Log in"), ("User ID", "Passphrase", "Continue")])
    uval = user if stage != "user" else ""
    pval = "hunter2" if stage == "submit" else ""
    body = (
        f"<h1>{site} sign-in</h1><form>"
        f'<label for="u">{ulabel}</label><input id="u" value="{uval}">'
        f'<label for="p">{plabel}</label><input id="p" type="password" value="{pval}">'
        f'<button id="ember{rng.randint(100, 999)}">{btn}</button></form><a href="/forgot">Forgot password?</a>'
    )
    hist = {"user": [], "password": [f"TYPE(1, '{user}')"], "submit": [f"TYPE(1, '{user}')", "TYPE(2, '<SECRET>')"]}[stage]
    obs = Observation(kind="html", data=_page(site, body, rng, nav=False), url=f"https://{site}.internal/login?next=%2F&ts={rng.randint(10**9, 2 * 10**9)}",
                      goal=f"Sign in as {user}", temporal_buffer=hist)

    def label(st: dict[str, Any]) -> dict[str, Any]:
        if stage == "user":
            return {"action": "TYPE", "target_id": _ids(st, "textbox", ulabel), "text": user, "confidence_score": 0.95}
        if stage == "password":
            return {"action": "TYPE", "target_id": _ids(st, "textbox", plabel), "text": "<SECRET>", "confidence_score": 0.93}
        return {"action": "CLICK", "target_id": _ids(st, "button", btn), "confidence_score": 0.96}

    return obs, label, f"login:{stage}"


def search(rng: random.Random):
    site, q = rng.choice(SITES), rng.choice(QUERIES)
    filled = rng.random() < 0.5
    body = (f'<h1>{rng.choice(SECTIONS)}</h1><input type="search" aria-label="Search" value="{q if filled else ""}">'
            f'<button>Search</button><table><tr><td>{rng.randint(1, 900)} results</td></tr></table>')
    obs = Observation(kind="html", data=_page(site, body, rng), url=f"https://{site}.internal/app", goal=f"Search for {q}",
                      temporal_buffer=[f"TYPE(4, '{q}')"] if filled else [])

    def label(st):
        if not filled:
            return {"action": "TYPE", "target_id": _ids(st, "searchbox", "search"), "text": q, "confidence_score": 0.94}
        return {"action": "CLICK", "target_id": _ids(st, "button", "search"), "confidence_score": 0.92}

    return obs, label, f"search:{'submit' if filled else 'type'}"


def cookie_banner(rng: random.Random):
    site = rng.choice(SITES)
    accept = rng.choice(["Accept all", "Accept cookies", "I agree", "Allow all"])
    body = (f'<div role="dialog" aria-label="Cookie consent"><p>We use cookies.</p>'
            f'<button>{accept}</button><button>Manage preferences</button></div><h1>Welcome</h1>')
    obs = Observation(kind="html", data=_page(site, body, rng), url=f"https://{site}.internal/", goal=f"Open {rng.choice(SECTIONS)}")
    return obs, lambda st: {"action": "CLICK", "target_id": _ids(st, "button", accept), "confidence_score": 0.9}, "cookie_banner"


def validation_error(rng: random.Random):
    site = rng.choice(SITES)
    field = rng.choice(["Email", "Company name", "Phone"])
    value = {"Email": "ops@example.com", "Company name": "Acme Corp", "Phone": "+1 555 0100"}[field]
    body = (f'<h1>New contact</h1><label for="n">Full name</label><input id="n" value="Dana Smith">'
            f'<label for="f">{field}</label><input id="f" value="">'
            f'<div role="alert">{field} is required</div><button>Save</button>')
    obs = Observation(kind="html", data=_page(site, body, rng), url=f"https://{site}.internal/contacts/new",
                      goal=f"Create a contact for Dana Smith with {field.lower()} {value}", temporal_buffer=["TYPE(4, 'Dana Smith')", "CLICK(7)"])
    return obs, lambda st: {"action": "TYPE", "target_id": _ids(st, "textbox", field), "text": value, "confidence_score": 0.9}, "validation_error"


def navigate(rng: random.Random):
    site = rng.choice(SITES)
    target = rng.choice(SECTIONS)
    others = [s for s in SECTIONS if s != target]
    links = rng.sample(others, 4) + [target]
    rng.shuffle(links)
    body = "<h1>Home</h1><ul>" + "".join(f'<li><a href="/{s.lower().replace(" ", "-")}">{s}</a></li>' for s in links) + "</ul>"
    obs = Observation(kind="html", data=_page(site, body, rng, nav=False), url=f"https://{site}.internal/home", goal=f"Go to {target}")
    return obs, lambda st: {"action": "CLICK", "target_id": _ids(st, "link", target), "confidence_score": 0.95}, "navigate"


def confirm_modal(rng: random.Random):
    site = rng.choice(SITES)
    item = rng.choice(["draft report", "API key", "old invoice", "test user"])
    intend = rng.random() < 0.6
    body = (f'<h1>Items</h1><dialog open><h2>Delete {item}?</h2><p>This cannot be undone.</p>'
            f"<button>Cancel</button><button>Delete</button></dialog>")
    goal = f"Delete the {item}" if intend else f"Review the {item} without changing anything"
    obs = Observation(kind="html", data=_page(site, body, rng), url=f"https://{site}.internal/items", goal=goal, temporal_buffer=["CLICK(9)"])
    btn = "Delete" if intend else "Cancel"
    return obs, lambda st: {"action": "CLICK", "target_id": _ids(st, "button", btn), "confidence_score": 0.9 if intend else 0.88}, f"modal:{btn.lower()}"


def terms_checkbox(rng: random.Random):
    site = rng.choice(SITES)
    checked = rng.random() < 0.5
    body = (f'<h1>Finish setup</h1><input type="checkbox" id="t" {"checked" if checked else ""}>'
            f'<label for="t">I accept the terms of service</label><button {"disabled" if not checked else ""}>Create workspace</button>')
    obs = Observation(kind="html", data=_page(site, body, rng), url=f"https://{site}.internal/onboarding", goal="Finish creating the workspace")

    def label(st):
        if not checked:
            return {"action": "CLICK", "target_id": _ids(st, "checkbox", "terms"), "confidence_score": 0.92}
        return {"action": "CLICK", "target_id": _ids(st, "button", "create"), "confidence_score": 0.93}

    return obs, label, f"terms:{'submit' if checked else 'check'}"


def canvas_target(rng: random.Random):
    site = rng.choice(SITES)
    x, y = rng.randint(200, 1000), rng.randint(150, 650)
    elements = [
        {"role": "heading", "name": "Throughput", "bbox": [40, 20, 300, 32]},
        {"role": "button", "name": "Export", "bbox": [1100, 20, 90, 32], "selector": "#export"},
        {"role": "img", "name": f"Anomaly marker at {x},{y}", "bbox": [x - 6, y - 6, 12, 12]},
    ]
    obs = Observation(kind="elements", data=elements, url=f"https://{site}.internal/metrics", viewport=(1280, 800),
                      goal="Open the anomaly marker on the chart")
    return obs, lambda st: {"action": "CLICK_XY", "coordinates": {"x": x, "y": y}, "confidence_score": 0.85}, "canvas_xy"


def ambiguous(rng: random.Random):
    site = rng.choice(SITES)
    body = "<h1>Something went wrong</h1><p>Unexpected response.</p><a href='/'>Home</a>"
    goal = rng.choice(["Approve the pending wire transfer", "Rotate the production database password", "Merge the release branch"])
    obs = Observation(kind="html", data=_page(site, body, rng, nav=False), url=f"https://{site}.internal/error", goal=goal)
    return obs, lambda st: {"action": "ESCALATE", "confidence_score": 0.3}, "escalate"


SCENARIOS = [login, login, search, cookie_banner, validation_error, navigate, navigate, confirm_modal, terms_checkbox, canvas_target, ambiguous]


def generate(n: int, seed: int = 7) -> list[dict[str, Any]]:
    """Returns ``[{state, action, scenario}]`` with extractor-normalized states."""
    rng = random.Random(seed)
    ex = StateExtractor(COMPUTER_USE)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        obs, label, scenario = rng.choice(SCENARIOS)(rng)
        res = ex.extract(obs)
        action = {**label(res.state), "supported_actions": list(COMPUTER_USE.supported_actions)}
        v = COMPUTER_USE.validate_action(action, res.state)
        if not v.ok or v.hallucinated:
            raise AssertionError(f"generator produced invalid sample for {scenario}: {v.errors + v.grounding_errors}")
        key = res.state_hash + str(action)
        if key in seen:
            continue
        seen.add(key)
        out.append({"state": res.state, "action": v.action, "scenario": scenario})
    return out
