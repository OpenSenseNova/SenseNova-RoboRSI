# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Readiness polling observes an externally cancelled case."""

from __future__ import annotations

from threading import Event

import pytest

from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline
from roborsi.utils.rpc import wait_for_ready


def test_readiness_polling_stops_when_case_is_cancelled() -> None:
    cancelled = Event()
    deadline = CaseDeadline(10.0, cancel_event=cancelled)

    class UnreadyClient:
        def call(self, method: str, *, timeout_s: float) -> None:
            assert method == "healthz"
            cancelled.set()
            raise ConnectionError("not ready")

    with pytest.raises(CaseCancelled, match="case cancelled"):
        wait_for_ready(
            UnreadyClient(),
            timeout_s=5.0,
            poll_interval_s=0.01,
            remaining_s=deadline.remaining,
        )
