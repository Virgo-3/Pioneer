"""Minimal HTTP adapters for the OpenAI Responses and TypeSafe System One APIs."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
SYSTEM_INSTRUCTIONS = (
    "You are Pioneer, a thoughtful conversational assistant. State uncertainty plainly. "
    "When recommending a consequential action, describe what can be undone, what cannot, "
    "and whether waiting for specific information could improve the decision. "
    "Do not claim that a model probability is a guarantee."
)


class ProviderError(Exception):
    pass


@dataclass(frozen=True)
class ProviderResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    response_id: str | None = None


@dataclass(frozen=True)
class TurnPlan:
    reply: str
    decision_requested: bool
    case_json: str | None
    missing: list[str]
    model: str
    input_tokens: int
    output_tokens: int
    response_id: str | None = None


TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "decision_requested": {"type": "boolean"},
        "case_json": {"type": ["string", "null"]},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply", "decision_requested", "case_json", "missing"],
    "additionalProperties": False,
}

TURN_INSTRUCTIONS = """You are Pioneer, a conversational decision assistant. Return JSON matching the supplied schema.
For ordinary conversation, set decision_requested false, case_json null, missing []. Answer naturally in reply.
When the user is choosing whether or how to act, set decision_requested true. Never claim to have calculated the best action yourself.
Only supply case_json when the USER's messages explicitly provide a complete numerical decision case: mutually exclusive states and probabilities summing to one, actions and payoff in each state, and, when evaluating a future signal, its likelihood in each state and the costs of waiting. Use the documented case shape: {"title":string,"units":string,"states":{name:probability},"actions":{name:{"cost":number,"outcomes":{state:{"payoff":number,"undo":number,"undo_cost":number}}}},"wait":{"delay_cost":number,"information_cost":number,"signals":{signal:{state:probability}}}}. Omit optional fields without explicit inputs. Do not invent a do-nothing payoff, a prior, a utility, a signal accuracy, or an undo value. Convert explicitly given percentages to fractions.
If the inputs are incomplete, use case_json null and ask for the most important missing assumptions in reply and missing. Prefer a short focused question over a long questionnaire. If case_json is supplied, reply should only introduce the assumptions; the application will calculate and append the recommendation.
Jev scores, when supplied, are qualitative routing hints. They are not outcome probabilities or evidence for numerical case fields."""


def _post(url: str, key: str, payload: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Truncate remote error text; never include an authorization header.
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise ProviderError(f"API returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"Could not reach the API: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProviderError("API returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ProviderError("API returned an unexpected response")
    return result


def ask_openai(messages: list[dict[str, str]], *, model: str | None = None) -> ProviderResult:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("Set OPENAI_API_KEY to chat with OpenAI.")
    selected_model = model or os.environ.get("PIONEER_OPENAI_MODEL", "gpt-6-astra")
    response = _post(OPENAI_ENDPOINT, key, {
        "model": selected_model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": messages,
        "store": False,
    })
    parts: list[str] = []
    for item in response.get("output", []):
        if isinstance(item, dict):
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    parts.append(content["text"])
    text = "\n".join(parts).strip()
    if not text:
        raise ProviderError("OpenAI returned no text; no conversation turn was recorded.")
    usage = response.get("usage") or {}
    return ProviderResult(text, str(response.get("model", selected_model)),
                          _tokens(usage, "input_tokens"), _tokens(usage, "output_tokens"),
                          response.get("id"))


def compose_turn(messages: list[dict[str, str]], *, model: str | None = None,
                 triage: dict[str, Any] | None = None) -> TurnPlan:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("Set OPENAI_API_KEY to chat with OpenAI.")
    selected_model = model or os.environ.get("PIONEER_OPENAI_MODEL", "gpt-6-astra")
    instructions = TURN_INSTRUCTIONS
    if triage:
        instructions += "\nJev triage (qualitative only): " + json.dumps(triage["scores"], sort_keys=True)
    response = _post(OPENAI_ENDPOINT, key, {
        "model": selected_model,
        "instructions": instructions,
        "input": messages,
        "text": {"format": {"type": "json_schema", "name": "pioneer_turn", "strict": True, "schema": TURN_SCHEMA}},
        "store": False,
    })
    parts: list[str] = []
    for item in response.get("output", []):
        if isinstance(item, dict):
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    parts.append(content["text"])
    if not parts:
        raise ProviderError("OpenAI returned no structured reply.")
    try:
        data = json.loads("".join(parts))
    except json.JSONDecodeError as exc:
        raise ProviderError("OpenAI returned invalid structured JSON.") from exc
    if (not isinstance(data, dict) or not isinstance(data.get("reply"), str)
            or not isinstance(data.get("decision_requested"), bool)
            or data.get("case_json") is not None and not isinstance(data.get("case_json"), str)
            or not isinstance(data.get("missing"), list)
            or any(not isinstance(item, str) for item in data["missing"])):
        raise ProviderError("OpenAI returned an invalid turn plan.")
    usage = response.get("usage") or {}
    return TurnPlan(data["reply"], data["decision_requested"], data["case_json"], data["missing"],
                    str(response.get("model", selected_model)), _tokens(usage, "input_tokens"),
                    _tokens(usage, "output_tokens"), response.get("id"))


def triage_jev(text: str, *, model: str | None = None) -> dict[str, Any]:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ProviderError("Set TYPESAFE_API_KEY to use Jev triage.")
    response = _post(JEV_ENDPOINT, key, {
        "model": model or os.environ.get("PIONEER_JEV_MODEL", "jev-latest"),
        "state": {"proposed_action": text},
        "questions": {
            "decision_request": {"type": "noul", "instructions": "Is the user asking to choose an action or decide whether to act now or wait?"},
            "time_sensitive": {"type": "noul", "instructions": "Does the proposed action have a stated near-term deadline or a clear cost of delaying it?"},
            "hard_to_reverse": {"type": "noul", "instructions": "Would carrying out the proposed action be difficult or costly to undo?"},
            "missing_information": {"type": "noul", "instructions": "Is a specific missing fact likely to change which action is best?"},
        },
    })
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise ProviderError("Jev returned no answers")
    scores: dict[str, float] = {}
    for name in ("decision_request", "time_sensitive", "hard_to_reverse", "missing_information"):
        answer = answers.get(name)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ProviderError(f"Jev returned an invalid {name} answer")
        value = answer.get("noul")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ProviderError(f"Jev returned an invalid {name} probability")
        scores[name] = float(value)
    usage = response.get("usage") or {}
    return {"model": str(response.get("model", model or "jev-latest")), "scores": scores,
            "input_tokens": _tokens(usage, "input_tokens"), "output_tokens": _tokens(usage, "output_tokens")}


def _tokens(usage: Any, key: str) -> int:
    value = usage.get(key, 0) if isinstance(usage, dict) else 0
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
