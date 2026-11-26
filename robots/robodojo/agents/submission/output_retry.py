# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""One retry owner for a buffered model turn, before any tool output delivery."""

import asyncio
import gzip
import hashlib
import json
import time
import zlib
from copy import deepcopy
from pathlib import Path

from roborsi.planner.provider_capture import ProviderCapture, parse_response_body

from .skillpack.capture_guard import GuardedCapture as OriginalCapture
from .skillpack.capture_guard import write
from .skillpack.context_windows import ContextWindows
from .turn_output_retry import MAX_ADDITIONAL_RETRIES, classify, compaction_request


class _Context:
    """Let explicit native text compaction use its original, tool-free input."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def prepare(self, request, compact):
        if compaction_request(request):
            return (
                request,
                {"native_cli_compaction": True, "actor_context_limits_unchanged": True},
                deepcopy(request["input"]),
            )
        return await self.original.prepare(request, compact)


class GuardedCapture(OriginalCapture):
    def __init__(self, *args, **kwargs):
        input_limit = kwargs.pop("context_input_token_limit", None)
        super().__init__(*args, **kwargs)
        if input_limit is not None:
            self.context = ContextWindows(
                Path(__file__).resolve().parent / "skillpack",
                input_token_limit=input_limit,
            )
        self._output_decisions = {}
        self.context = _Context(self.context)

    def _publish_capture_record(self, path, record, items):
        attempt = self._attempt_buffers.get(record["request_id"])
        request_path = path.parent / f"{record['request_id']:06d}.request.json"
        request = (
            attempt["request"]
            if attempt
            else json.loads(request_path.read_text()).get("forwarded", {})
        )
        raw = self._capture_raw_bytes(path)
        try:
            encoding = record.get("content_encoding", "").lower()
            if encoding == "gzip":
                raw = gzip.decompress(raw)
            elif encoding == "deflate":
                raw = zlib.decompress(raw)
            parsed = parse_response_body(
                raw,
                record["content_type"],
                endpoint=record["endpoint"],
                include_response=True,
            )
            response = parsed.get("response")
            if response is None and "text/event-stream" not in record["content_type"]:
                response = json.loads(raw)
        except (ValueError, OSError, EOFError, zlib.error):
            response = None
        decision = classify(
            record["status_code"],
            response,
            request,
            transport_complete=record.get("transport_complete", False),
            transport_error=record.get("transport_error"),
        )
        if decision["valid"] and not isinstance(response.get("usage"), dict):
            decision = {
                "valid": False,
                "retryable": False,
                "unknown": False,
                "reason": "usage_unknown",
            }
        self._output_decisions[record["request_id"]] = decision
        # Preserve the wire measurement separately from public-output acceptance.
        record["provider_response_completed"] = record.get("response_completed", False)
        record["output_validation"] = decision
        if not decision["valid"]:
            record["response_completed"] = False
            record["complete"] = False
        super()._publish_capture_record(path, record, items)

    def _hold(self, kind, evidence, *, cohort=False):
        decision = self._output_decisions.get(evidence.get("request_id"), {})
        if decision.get("retryable") and kind in {
            "provider_terminal_or_protocol_error",
            "usage_unknown",
        }:
            return  # A known invalid attempt is held only after its shared budget.
        return super()._hold(kind, evidence, cohort=cohort)

    async def _buffered_forward(self, scope, payload):
        request_sha = hashlib.sha256(payload).hexdigest()
        first_request_id = None
        for attempt_number in range(MAX_ADDITIONAL_RETRIES + 1):
            if self._retention_failed:
                return await self._buffered_error("capture_or_retention_failed", {})
            remaining = (
                None if self.case_deadline is None else self.case_deadline - time.time()
            )
            if remaining is not None and remaining <= 0:
                return await self._buffered_error(
                    "case_wall_budget", {"deadline": self.case_deadline}
                )
            if self._native_terminal_observed():
                return await self._buffered_error("native_terminal_observed", {})
            messages = []

            async def collect(message):
                messages.append(message)

            before = len(self.rows)
            self._attempts += 1
            forward = ProviderCapture._forward(self, scope, payload, collect)
            try:
                if remaining is None:
                    await forward
                else:
                    await asyncio.wait_for(forward, timeout=remaining)
            except asyncio.TimeoutError:
                return await self._buffered_error(
                    "case_wall_budget", {"deadline": self.case_deadline}
                )
            if self.case_deadline is not None and time.time() >= self.case_deadline:
                return await self._buffered_error(
                    "case_wall_budget", {"deadline": self.case_deadline}
                )
            if len(self.rows) != before + 1:
                return await self._buffered_error(
                    "missing_attempt_accounting",
                    {"rows_before": before, "rows_after": len(self.rows)},
                )
            row = self.rows[-1]
            if self._native_terminal_observed():
                return await self._buffered_error(
                    "native_terminal_observed", {"request_id": row["request_id"]}
                )
            first_request_id = first_request_id or row["request_id"]
            decision = self._output_decisions[row["request_id"]]
            evidence = {
                "at": time.time(),
                "request_id": row["request_id"],
                "first_request_id": first_request_id,
                "request_sha256": request_sha,
                "attempt": attempt_number + 1,
                "max_additional_retries": MAX_ADDITIONAL_RETRIES,
                "usage": row.get("usage"),
                "failed_output_forwarded": False,
                "budget_reset": False,
                "case_deadline": self.case_deadline,
                **decision,
            }
            with (self.case / "output-retries.jsonl").open("a") as stream:
                stream.write(json.dumps(evidence) + "\n")
            if row.get("capture_memory_bound_exceeded") or self._retention_failed:
                return await self._buffered_error(
                    "capture_or_retention_failed", evidence
                )
            if decision["valid"] and row.get("completed"):
                return messages
            if not decision["retryable"]:
                return await self._buffered_error(
                    "output_outcome_unknown"
                    if decision["unknown"]
                    else "output_terminal_rejection",
                    evidence,
                )
            if attempt_number == MAX_ADDITIONAL_RETRIES:
                return await self._buffered_error("output_retry_exhausted", evidence)
            write(self.case / "output-retry.latest.json", evidence)
        raise AssertionError("bounded output retry loop fell through")
