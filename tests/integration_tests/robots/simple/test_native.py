# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in native interface smoke; it does not measure task success rate."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action
from robots.simple.catalog import Catalog
from robots.simple.roborsi_adapter import SimpleAdapter

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBORSI_RUN_SIMPLE_NATIVE") != "1",
    reason="requires explicit native SIMPLE allocation",
)


@pytest.mark.timeout(1200)
def test_three_native_rows():
    import numpy as np
    from PIL import Image

    root = Path(os.environ["ROBORSI_SIMPLE_EVIDENCE_DIR"])
    root.mkdir(parents=True, exist_ok=False)
    catalog = Catalog(Path(os.environ["ROBORSI_SIMPLE_CATALOG"]))
    task = catalog.tasks()[0]
    media = CaseMediaChannel(root / "media")
    adapter = SimpleAdapter(
        root, ArtifactPublisher(root, "native-smoke"), catalog=catalog
    )
    try:
        started = adapter.start(task, "native-smoke")
        observation = started.observation
        for _ in range(3):
            image = Image.open(
                io.BytesIO(
                    media.read(
                        observation.media[0],
                        observation.episode_id,
                        observation.step_index,
                    )
                )
            )
            assert np.asarray(image).std() > 2
            row = [*observation.data["state"], 0.0, 0.0, 0.0, 0.0]
            result = adapter.step(
                Action(
                    benchmark="simple",
                    episode_id="native-smoke",
                    step_index=observation.step_index,
                    data={"actions": [row]},
                )
            )
            observation = result.observation
            if result.terminated or result.truncated:
                adapter.finish()
                break
        assert 1 <= observation.data["native_env_steps"] <= 3
    finally:
        adapter.close()
    assert adapter.daemon.poll() is not None
