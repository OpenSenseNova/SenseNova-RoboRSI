# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Validate public output before delivery; never interpret private reasoning."""

import json

from jsonschema import Draft202012Validator

MAX_ADDITIONAL_RETRIES = 5


def compaction_request(request):
    metadata = request.get("client_metadata") or {}
    if not isinstance(metadata, dict):
        return False
    turn = metadata.get("x-codex-turn-metadata", {})
    if isinstance(turn, str):
        try:
            turn = json.loads(turn)
        except ValueError:
            turn = {}
    return metadata.get("request_kind") == "compaction" or (
        isinstance(turn, dict) and turn.get("request_kind") == "compaction"
    )


def terminal_rejection(status, response):
    if status in {400, 401, 402, 403}:
        return True

    def codes(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"code", "type", "reason"} and isinstance(child, str):
                    yield child.lower()
                elif key in {"error", "details", "incomplete_details"}:
                    yield from codes(child)
        elif isinstance(value, list):
            for child in value:
                yield from codes(child)

    return any(
        any(
            token in code
            for token in (
                "content_policy",
                "content_filter",
                "authentication",
                "invalid_api_key",
                "insufficient_quota",
                "billing_hard_limit",
                "forbidden",
            )
        )
        for code in codes(response)
    )


def _tools(request):
    result = {}

    def visit(specs, prefix=""):
        for spec in specs if isinstance(specs, list) else []:
            if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
                continue
            name = prefix + spec["name"]
            if spec.get("type") == "namespace":
                visit(spec.get("tools"), name + ".")
            elif spec.get("type") in {"function", "custom"}:
                result[name] = spec

    visit(request.get("tools"))
    for item in (
        request.get("input", []) if isinstance(request.get("input"), list) else []
    ):
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            visit(item.get("tools"))
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON argument")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON argument field")
        result[key] = value
    return result


def output_error(request, response):
    if not isinstance(response, dict):
        return "response_not_object"
    if response.get("error"):
        return "provider_error"
    if response.get("status") != "completed":
        details = response.get("incomplete_details")
        return (
            details.get("reason") or "response_not_completed"
            if isinstance(details, dict)
            else "response_not_completed"
        )
    if not isinstance(response.get("id"), str) or not response["id"]:
        return "response_id_missing"
    output = response.get("output")
    if not isinstance(output, list):
        return "output_not_list"
    calls, texts = [], []
    for item in output:
        if not isinstance(item, dict):
            return "output_item_not_object"
        kind = item.get("type")
        if kind == "reasoning":
            continue
        if kind in {"function_call", "custom_tool_call"}:
            calls.append(item)
        elif kind == "message":
            content = item.get("content")
            if not isinstance(content, list):
                return "message_content_invalid"
            for part in content:
                if not isinstance(part, dict):
                    return "message_part_invalid"
                if part.get("type") == "refusal":
                    return "explicit_refusal"
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        return "output_text_invalid"
                    texts.append(part["text"])
        else:
            return "unsupported_output_item"
    if compaction_request(request):
        if calls:
            return "compaction_contains_tool_call"
        return None if "".join(texts).strip() else "compaction_public_summary_missing"
    tools, ids = _tools(request), set()
    for call in calls:
        name = call.get("name")
        if not isinstance(name, str):
            return "tool_name_invalid"
        namespace = call.get("namespace")
        if namespace:
            if not isinstance(namespace, str):
                return "tool_namespace_invalid"
            name = namespace + "." + name
        if name not in tools:
            return "tool_name_not_declared"
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in ids:
            return "tool_call_id_invalid"
        ids.add(call_id)
        declaration = tools[name]
        if call["type"] == "function_call":
            if declaration.get("type") != "function":
                return "tool_call_kind_invalid"
            try:
                arguments = json.loads(
                    call["arguments"],
                    parse_constant=_reject_constant,
                    object_pairs_hook=_unique_object,
                )
            except (KeyError, TypeError, ValueError):
                return "tool_arguments_invalid_JSON"
            if not Draft202012Validator(declaration.get("parameters", {})).is_valid(
                arguments
            ):
                return "tool_arguments_schema_invalid"
        elif declaration.get("type") != "custom" or not isinstance(
            call.get("input"), str
        ):
            return "custom_tool_input_invalid"
    return None if calls or "".join(texts).strip() else "public_output_missing"


def classify(status, response, request, *, transport_complete, transport_error=None):
    if transport_error == "cancelled_or_disconnected":
        return {
            "retryable": False,
            "unknown": True,
            "reason": "caller_cancelled_or_disconnected",
            "valid": False,
        }
    if terminal_rejection(status, response):
        return {
            "retryable": False,
            "unknown": False,
            "reason": "terminal_provider_rejection",
            "valid": False,
        }
    if not transport_complete:
        return {
            "retryable": True,
            "unknown": True,
            "reason": "transport_outcome_unknown",
            "valid": False,
        }
    error = response.get("error") if isinstance(response, dict) else None
    if isinstance(error, dict) and error.get("outcome_unknown") is True:
        return {
            "retryable": True,
            "unknown": True,
            "reason": "upstream_outcome_unknown",
            "valid": False,
        }
    if not 200 <= status < 300:
        return {
            "retryable": status in {408, 429, 500, 502, 503, 504},
            "unknown": False,
            "reason": "HTTP_" + str(status),
            "valid": False,
        }
    error = output_error(request, response)
    return {
        "retryable": bool(error) and error != "explicit_refusal",
        "unknown": False,
        "reason": error,
        "valid": error is None,
    }
