"""Computer-Use Action Schema with spatial fallbacks (spec section 5.1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

COMPUTER_USE_ACTIONS = ["CLICK", "TYPE", "CLICK_XY", "ESCALATE"]


class ViewportNode(BaseModel):
    id: int
    role: str
    name: str = ""
    value: str | None = None
    # [x, y, width, height] in viewport pixels - enables CLICK_XY fallbacks.
    bbox: list[int] | None = None
    states: list[str] | None = None  # e.g. ["disabled", "checked", "focused"]


class ComputerUseState(BaseModel):
    url: str
    temporal_buffer: list[str] = Field(default_factory=list)
    viewport_tree: list[ViewportNode] = Field(default_factory=list)
    # Raw viewport screenshot (base64 PNG) for GUIs without accessibility tags.
    # Never serialized into the student prompt; consumed by the vision parser.
    screenshot_b64: str | None = Field(default=None, exclude=True)


class Coordinates(BaseModel):
    x: int
    y: int


class ComputerUseAction(BaseModel):
    confidence_score: float = Field(ge=0.0, le=1.0)
    action: str
    target_id: int | None = None
    coordinates: Coordinates | None = None
    text: str | None = None  # payload for TYPE
    supported_actions: list[str] = Field(default_factory=lambda: list(COMPUTER_USE_ACTIONS))

    @model_validator(mode="after")
    def _check(self) -> "ComputerUseAction":
        if self.action not in self.supported_actions:
            raise ValueError(f"action {self.action!r} not in supported_actions")
        if self.action in ("CLICK", "TYPE") and self.target_id is None:
            raise ValueError(f"{self.action} requires target_id")
        if self.action == "TYPE" and self.text is None:
            raise ValueError("TYPE requires text")
        if self.action == "CLICK_XY" and self.coordinates is None:
            raise ValueError("CLICK_XY requires coordinates")
        return self


def grounding_errors(state: dict[str, Any], action: dict[str, Any]) -> list[str]:
    """Hallucination checks: the action must reference things that exist."""
    errs: list[str] = []
    ids = {n.get("id") for n in state.get("viewport_tree", [])}
    tid = action.get("target_id")
    if tid is not None and tid not in ids:
        errs.append(f"target_id {tid} not present in viewport_tree")
    coords = action.get("coordinates")
    if action.get("action") == "CLICK_XY" and coords:
        if coords.get("x", -1) < 0 or coords.get("y", -1) < 0:
            errs.append("coordinates outside viewport")
    return errs
