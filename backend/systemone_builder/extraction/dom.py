"""DOM / accessibility-tree pre-processor for the Computer-Use domain.

Accepted observation formats (``ComputerUseObservation.kind``):

``html``      Raw page HTML. Parsed with the stdlib parser; ARIA roles, labels
              and visibility rules are resolved heuristically.
``ax_tree``   Playwright ``page.accessibility.snapshot()`` / CDP-style nested
              ``{role, name, value, children, ...}`` tree.
``elements``  Flat element list produced by :data:`BROWSER_COLLECTOR_JS`
              (recommended: includes bounding boxes for CLICK_XY fallbacks
              and CSS selectors for execution).

Every format is reduced to the compact ``viewport_tree`` of the contract.
Element ids are assigned by document order *after* filtering, so the same
page layout always yields the same ids - a key ingredient for prefix-cache
stability. The id -> selector/bbox index is returned separately and never
enters the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from systemone_builder.extraction.fuzzy import EntityVault, FuzzyScrubber, ScrubPolicy

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox", "listbox", "option",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "switch", "slider", "spinbutton",
    "treeitem", "gridcell",
}
CONTEXT_ROLES = {"heading", "alert", "alertdialog", "dialog", "status", "img", "banner", "navigation"}
KEEP_ROLES = INTERACTIVE_ROLES | CONTEXT_ROLES

VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
SKIP_TAGS = {"script", "style", "template", "noscript", "head", "svg", "iframe"}

INPUT_ROLES = {
    "button": "button", "submit": "button", "reset": "button", "image": "button",
    "checkbox": "checkbox", "radio": "radio", "range": "slider", "number": "spinbutton",
    "search": "searchbox", "hidden": None,
}


def tag_role(tag: str, attrs: dict[str, str]) -> str | None:
    if attrs.get("role"):
        return attrs["role"].split()[0].lower()
    if tag == "a":
        return "link" if "href" in attrs else None
    if tag == "button":
        return "button"
    if tag == "input":
        return INPUT_ROLES.get(attrs.get("type", "text").lower(), "textbox")
    if tag == "textarea":
        return "textbox"
    if tag == "select":
        return "listbox" if "multiple" in attrs else "combobox"
    if tag == "option":
        return "option"
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return "heading"
    if tag == "img":
        return "img" if attrs.get("alt") else None
    if tag == "dialog":
        return "dialog"
    if tag == "summary":
        return "button"
    return None


def _hidden(tag: str, attrs: dict[str, str]) -> bool:
    style = attrs.get("style", "").replace(" ", "").lower()
    return (
        "hidden" in attrs
        or attrs.get("aria-hidden") == "true"
        or "display:none" in style
        or "visibility:hidden" in style
        or (tag == "input" and attrs.get("type", "").lower() == "hidden")
    )


@dataclass
class _El:
    tag: str
    attrs: dict[str, str]
    role: str | None
    path: str
    text: list[str] = field(default_factory=list)


@dataclass
class _Frame:
    key: str  # e.g. "div[2]"
    hidden: bool
    el: _El | None
    ident: str | None  # DOM id whose text is collected (aria-labelledby)
    label_for: str | None  # <label for=...>
    text: list[str] = field(default_factory=list)


class _DomParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Frame] = []
        self.elements: list[_El] = []
        self.labels_for: dict[str, list[str]] = {}
        self.id_text: dict[str, str] = {}
        self._counters: list[dict[str, int]] = [{}]

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k.lower(): (v or "") for k, v in attrs_list}
        counts = self._counters[-1]
        counts[tag] = counts.get(tag, 0) + 1
        key = f"{tag}[{counts[tag]}]"
        path = "/".join([f.key for f in self.stack] + [key])
        hidden = (bool(self.stack) and self.stack[-1].hidden) or tag in SKIP_TAGS or _hidden(tag, attrs)
        el: _El | None = None
        role = tag_role(tag, attrs)
        if not hidden and role in KEEP_ROLES:
            el = _El(tag, attrs, role, path)
            self.elements.append(el)
        if tag in VOID_TAGS:
            return
        label_for = attrs.get("for") if tag == "label" else None
        self.stack.append(_Frame(key, hidden, el, attrs.get("id") or None, label_for))
        self._counters.append({})

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i].key.startswith(f"{tag}["):
                for frame in self.stack[i:]:
                    self._close(frame)
                del self.stack[i:]
                del self._counters[i + 1 :]
                break

    def close(self) -> None:
        super().close()
        for frame in self.stack:
            self._close(frame)
        self.stack.clear()

    def _close(self, frame: _Frame) -> None:
        text = " ".join(frame.text)
        if frame.ident:
            self.id_text[frame.ident] = text
        if frame.label_for:
            self.labels_for.setdefault(frame.label_for, []).append(text)

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or (self.stack and self.stack[-1].hidden):
            return
        for frame in self.stack:
            frame.text.append(text)
            if frame.el is not None:
                frame.el.text.append(text)


def _accessible_name(el: _El, parser: _DomParser) -> str:
    a = el.attrs
    if a.get("aria-label"):
        return a["aria-label"]
    if a.get("aria-labelledby"):
        parts = [parser.id_text.get(i, "") for i in a["aria-labelledby"].split()]
        if any(parts):
            return " ".join(p for p in parts if p)
    if a.get("id") and a["id"] in parser.labels_for:
        return " ".join(parser.labels_for[a["id"]])
    if el.tag == "img":
        return a.get("alt", "")
    if el.tag == "input" and a.get("type", "").lower() in ("submit", "button", "reset"):
        return a.get("value", "") or a.get("type", "").capitalize()
    text = " ".join(el.text)
    return text or a.get("placeholder", "") or a.get("title", "") or a.get("name", "")


def _states(el_attrs: dict[str, Any]) -> list[str] | None:
    st = [s for s in ("disabled", "checked", "selected", "required", "readonly") if s in el_attrs and el_attrs[s] not in (False, None, "false")]
    if el_attrs.get("aria-expanded") == "true" or el_attrs.get("expanded") is True:
        st.append("expanded")
    if el_attrs.get("aria-checked") == "true":
        st.append("checked")
    if el_attrs.get("focused") is True:
        st.append("focused")
    return st or None


@dataclass
class ExtractedDom:
    viewport_tree: list[dict[str, Any]]
    # id -> execution handle (selector / xpath / bbox) kept out of the prompt
    index: dict[int, dict[str, Any]]


class DomExtractor:
    def __init__(
        self,
        scrubber: FuzzyScrubber | None = None,
        max_nodes: int = 150,
        max_name_len: int = 80,
        viewport: tuple[int, int] | None = None,
    ) -> None:
        self.scrubber = scrubber or FuzzyScrubber(ScrubPolicy())
        self.max_nodes = max_nodes
        self.max_name_len = max_name_len
        self.viewport = viewport

    # ----------------------------------------------------------- normalize
    def _node(self, role: str, name: str, value: str | None, bbox: Any, states: list[str] | None, vault: EntityVault | None) -> dict[str, Any]:
        name = self.scrubber.scrub_text(" ".join(str(name or "").split()), vault)[: self.max_name_len]
        node: dict[str, Any] = {"role": role, "name": name}
        if value not in (None, ""):
            node["value"] = self.scrubber.scrub_text(str(value), vault)[: self.max_name_len]
        if bbox:
            node["bbox"] = [int(round(float(v))) for v in bbox]
        if states:
            node["states"] = sorted(states)
        return node

    def _in_viewport(self, bbox: list[int] | None) -> bool:
        if not bbox or not self.viewport:
            return True
        x, y, w, h = bbox
        vw, vh = self.viewport
        return w > 0 and h > 0 and x + w > 0 and y + h > 0 and x < vw and y < vh

    def _finalize(self, raw: list[tuple[dict[str, Any], dict[str, Any]]]) -> ExtractedDom:
        raw = [(n, h) for n, h in raw if self._in_viewport(n.get("bbox"))]
        # Drop nameless non-interactive context nodes and exact duplicates.
        seen: set[tuple[Any, ...]] = set()
        kept = []
        for n, h in raw:
            if n["role"] not in INTERACTIVE_ROLES and not n["name"]:
                continue
            sig = (n["role"], n["name"], tuple(n.get("bbox") or ()), h.get("selector"))
            if sig in seen:
                continue
            seen.add(sig)
            kept.append((n, h))
        if len(kept) > self.max_nodes:
            interactive = [k for k in kept if k[0]["role"] in INTERACTIVE_ROLES]
            context = [k for k in kept if k[0]["role"] not in INTERACTIVE_ROLES]
            budget = max(self.max_nodes - len(interactive), 0)
            keep_ids = {id(k) for k in interactive[: self.max_nodes]} | {id(k) for k in context[:budget]}
            kept = [k for k in kept if id(k) in keep_ids]
        tree, index = [], {}
        for i, (n, h) in enumerate(kept, start=1):
            tree.append({"id": i, **n})
            index[i] = {**h, "bbox": n.get("bbox")}
        return ExtractedDom(tree, index)

    # ------------------------------------------------------------- formats
    def from_html(self, html: str, vault: EntityVault | None = None) -> ExtractedDom:
        p = _DomParser()
        p.feed(html)
        p.close()
        raw = []
        for el in p.elements:
            bbox = None
            if el.attrs.get("data-s1-bbox"):
                bbox = [float(v) for v in el.attrs["data-s1-bbox"].split(",")]
            value = el.attrs.get("value") if el.role in ("textbox", "searchbox", "combobox", "spinbutton", "slider") else None
            if el.role == "textbox" and el.attrs.get("type", "").lower() == "password":
                value = "<SECRET>" if value else None
            node = self._node(el.role or "generic", _accessible_name(el, p), value, bbox, _states(el.attrs), vault)
            handle = {"xpath": "/" + el.path, "tag": el.tag}
            if el.attrs.get("id"):
                handle["selector"] = f"#{el.attrs['id']}"
            raw.append((node, handle))
        return self._finalize(raw)

    def from_ax_tree(self, root: dict[str, Any], vault: EntityVault | None = None) -> ExtractedDom:
        raw = []

        def walk(node: dict[str, Any], path: str) -> None:
            role = str(node.get("role", "")).lower()
            if role in KEEP_ROLES:
                bbox = node.get("bbox") or node.get("boundingBox")
                if isinstance(bbox, dict):
                    bbox = [bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)]
                n = self._node(role, node.get("name", ""), node.get("value"), bbox, _states(node), vault)
                raw.append((n, {"ax_path": path}))
            for i, child in enumerate(node.get("children") or []):
                walk(child, f"{path}/{i}")

        walk(root, "")
        return self._finalize(raw)

    def from_elements(self, elements: list[dict[str, Any]], vault: EntityVault | None = None) -> ExtractedDom:
        raw = []
        for e in elements:
            role = str(e.get("role", "")).lower()
            if role not in KEEP_ROLES or e.get("visible") is False:
                continue
            n = self._node(role, e.get("name", ""), e.get("value"), e.get("bbox"), e.get("states"), vault)
            raw.append((n, {k: e[k] for k in ("selector", "xpath", "tag") if k in e}))
        return self._finalize(raw)


# Injected with Playwright: ``elements = await page.evaluate(BROWSER_COLLECTOR_JS)``
BROWSER_COLLECTOR_JS = r"""
() => {
  const ROLE = (el) => {
    const r = el.getAttribute('role'); if (r) return r.split(' ')[0].toLowerCase();
    const t = el.tagName.toLowerCase(), type = (el.getAttribute('type') || 'text').toLowerCase();
    if (t === 'a' && el.hasAttribute('href')) return 'link';
    if (t === 'button' || t === 'summary') return 'button';
    if (t === 'input') return ({button:'button',submit:'button',reset:'button',image:'button',checkbox:'checkbox',
      radio:'radio',range:'slider',number:'spinbutton',search:'searchbox',hidden:null})[type] ?? 'textbox';
    if (t === 'textarea') return 'textbox';
    if (t === 'select') return el.multiple ? 'listbox' : 'combobox';
    if (/^h[1-6]$/.test(t)) return 'heading';
    if (t === 'img' && el.alt) return 'img';
    if (t === 'dialog') return 'dialog';
    return null;
  };
  const NAME = (el) => {
    const l = el.getAttribute('aria-label'); if (l) return l;
    const lb = el.getAttribute('aria-labelledby');
    if (lb) return lb.split(' ').map(id => document.getElementById(id)?.innerText || '').join(' ');
    if (el.labels && el.labels.length) return Array.from(el.labels).map(x => x.innerText).join(' ');
    if (el.tagName === 'IMG') return el.alt;
    return (el.innerText || el.value || el.placeholder || el.title || '').trim().slice(0, 200);
  };
  const SEL = (el) => {
    if (el.id && !/\d{3,}/.test(el.id)) return '#' + CSS.escape(el.id);
    const parts = [];
    for (let n = el; n && n.nodeType === 1 && parts.length < 6; n = n.parentElement) {
      let i = 1; for (let s = n.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === n.tagName) i++;
      parts.unshift(n.tagName.toLowerCase() + ':nth-of-type(' + i + ')');
    }
    return parts.join(' > ');
  };
  const out = [];
  for (const el of document.querySelectorAll('*')) {
    const role = ROLE(el); if (!role) continue;
    const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    const visible = r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
    const states = [];
    if (el.disabled) states.push('disabled');
    if (el.checked) states.push('checked');
    if (document.activeElement === el) states.push('focused');
    out.push({role, name: NAME(el), value: (el.type === 'password' ? (el.value ? '<SECRET>' : '') : (el.value ?? null)),
      bbox: [r.x, r.y, r.width, r.height], visible, states, selector: SEL(el), tag: el.tagName.toLowerCase()});
  }
  return out;
}
"""
