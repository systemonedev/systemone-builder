"""Fuzzy state caching: scrub volatile data to protect the vLLM prefix cache.

vLLM's automatic prefix caching reuses KV blocks only while the token prefix
is byte-identical. DOM snapshots and logs are full of values that change on
every observation (timestamps, request ids, CSRF tokens, cache-busters, flow
ids...). Left in place they shift every following token and turn each request
into a cache miss. The scrubber replaces them with stable placeholders.

Entities that the model must *act on* (e.g. an attacker IP) can optionally be
tokenized through a reversible :class:`EntityVault` - the prompt sees
``<IP_1>`` and the action is rehydrated with the real value afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    replacement: str


def _r(name: str, pattern: str, replacement: str, flags: int = 0) -> Rule:
    return Rule(name, re.compile(pattern, flags), replacement)


# Order matters: most specific first.
DEFAULT_RULES: list[Rule] = [
    _r("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "<JWT>"),
    _r("uuid", r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "<UUID>"),
    _r(
        "iso_ts",
        r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b",
        "<TS>",
    ),
    _r(
        "syslog_ts",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b",
        "<TS>",
    ),
    _r(
        "clf_ts",
        r"\b\d{2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{4}:\d{2}:\d{2}:\d{2}(?: [+-]\d{4})?",
        "<TS>",
    ),
    _r("clock", r"(?<![\d.:])\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AaPp][Mm])?\b", "<TIME>"),
    _r("epoch", r"\b1[5-9]\d{8}(?:\d{3})?(?:\.\d+)?\b", "<EPOCH>"),
    _r(
        "reltime",
        r"\b(?:\d+|a|an)\s+(?:second|sec|minute|min|hour|hr|day|week|month|year)s?\s+ago\b|\bjust now\b",
        "<RELTIME>",
        re.IGNORECASE,
    ),
    _r("hex", r"\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{16,}\b", "<HEX>"),
    # Opaque tokens: long mixed-case alnum strings containing digits.
    _r("token", r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]{24,}\b", "<TOKEN>"),
    # Framework-generated element ids / classes (ember123, react-select-5, css-1x2y3z).
    _r("gen_id", r"\b(?:ember|react-select-|css-|jsx-|sc-|mui-|radix-|headlessui-[a-z]+-)[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b", "<GEN_ID>"),
]

# URL query parameters whose values are cache-busters / session state.
VOLATILE_QUERY_PARAMS = {
    "_", "t", "ts", "timestamp", "cb", "cache", "cachebuster", "v", "ver", "rnd", "rand", "nonce",
    "sid", "session", "sessionid", "token", "csrf", "csrf_token", "xsrf", "state", "code", "sig",
    "signature", "expires", "x-amz-signature", "x-amz-date", "x-amz-credential", "utm_source",
    "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid", "requestid",
}

IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
COUNTER_RE = re.compile(r"\((\d+)\)|\b(\d+)\s+(?=(?:new|unread|notifications?|messages?|items?)\b)", re.IGNORECASE)


class EntityVault:
    """Reversible entity tokenization (``203.0.113.9`` -> ``<IP_1>``)."""

    def __init__(self) -> None:
        self.forward: dict[str, str] = {}
        self.reverse: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def tokenize(self, kind: str, value: str) -> str:
        if value in self.forward:
            return self.forward[value]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        token = f"<{kind}_{self._counts[kind]}>"
        self.forward[value] = token
        self.reverse[token] = value
        return token

    def rehydrate(self, obj: Any) -> Any:
        if not self.reverse:
            return obj
        if isinstance(obj, str):
            for tok, val in self.reverse.items():
                obj = obj.replace(tok, val)
            return obj
        if isinstance(obj, list):
            return [self.rehydrate(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.rehydrate(v) for k, v in obj.items()}
        return obj

    def to_dict(self) -> dict[str, str]:
        return dict(self.reverse)

    @classmethod
    def from_dict(cls, reverse: dict[str, str]) -> "EntityVault":
        v = cls()
        for tok, val in reverse.items():
            v.reverse[tok] = val
            v.forward[val] = tok
            kind = tok.strip("<>").rsplit("_", 1)[0]
            v._counts[kind] = max(v._counts.get(kind, 0), int(tok.strip("<>").rsplit("_", 1)[1]))
        return v


@dataclass
class ScrubPolicy:
    rules: list[Rule] = field(default_factory=lambda: list(DEFAULT_RULES))
    # Dict keys removed entirely (volatile metadata carrying no signal).
    drop_fields: set[str] = field(default_factory=set)
    # Dict keys whose string values are never rewritten.
    preserve_fields: set[str] = field(default_factory=set)
    # Keys whose values are URLs (query-param scrubbing applies).
    url_fields: set[str] = field(default_factory=lambda: {"url", "href", "src", "referer", "referrer"})
    normalize_counters: bool = True
    tokenize_ips: bool = False
    tokenize_emails: bool = True
    max_string_len: int = 512
    extra_volatile_params: set[str] = field(default_factory=set)
    # User-supplied regexes from the Prompt-to-Workflow engine.
    custom_patterns: list[tuple[str, str]] = field(default_factory=list)

    def compiled_rules(self) -> list[Rule]:
        extra = [_r(f"custom_{i}", p, repl) for i, (p, repl) in enumerate(self.custom_patterns)]
        return extra + self.rules


class FuzzyScrubber:
    def __init__(self, policy: ScrubPolicy | None = None) -> None:
        self.policy = policy or ScrubPolicy()
        self._rules = self.policy.compiled_rules()
        self._volatile = VOLATILE_QUERY_PARAMS | {p.lower() for p in self.policy.extra_volatile_params}

    # --------------------------------------------------------------- strings
    def scrub_text(self, text: str, vault: EntityVault | None = None) -> str:
        p = self.policy
        if vault is not None and p.tokenize_ips:
            text = IP_RE.sub(lambda m: vault.tokenize("IP", m.group(0)), text)
        if vault is not None and p.tokenize_emails:
            text = EMAIL_RE.sub(lambda m: vault.tokenize("EMAIL", m.group(0)), text)
        for rule in self._rules:
            text = rule.pattern.sub(rule.replacement, text)
        if p.normalize_counters:
            text = COUNTER_RE.sub(lambda m: "(<N>)" if m.group(1) else "<N> ", text)
        if len(text) > p.max_string_len:
            text = text[: p.max_string_len] + "…"
        return text

    def scrub_url(self, url: str, vault: EntityVault | None = None) -> str:
        try:
            parts = urlsplit(url)
        except ValueError:
            return self.scrub_text(url, vault)
        query = [(k, "<VAR>" if k.lower() in self._volatile else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        fragment = "" if re.search(r"\d{3,}|[A-Za-z0-9_-]{20,}", parts.fragment or "") else parts.fragment
        rebuilt = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, safe="<>"), fragment))
        return self.scrub_text(rebuilt, vault)

    # --------------------------------------------------------------- objects
    def scrub(self, obj: Any, vault: EntityVault | None = None, _key: str | None = None) -> Any:
        p = self.policy
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in p.drop_fields:
                    continue
                out[k] = self.scrub(v, vault, k)
            return out
        if isinstance(obj, list):
            return [self.scrub(v, vault, _key) for v in obj]
        if isinstance(obj, str):
            if _key in p.preserve_fields:
                return obj
            if _key in p.url_fields:
                return self.scrub_url(obj, vault)
            return self.scrub_text(obj, vault)
        return obj

    def placeholders_in(self, obj: Any) -> set[str]:
        return set(re.findall(r"<[A-Z_]+(?:_\d+)?>", str(obj)))


def iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)
