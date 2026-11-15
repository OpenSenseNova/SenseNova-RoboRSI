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

"""Try forbidden accesses from inside a real Agent container."""

import importlib.util
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from roborsi_hold_agent import HoldAgent

from roborsi.evaluation.messages import NativeResult
from roborsi.utils.config import require_project_env


class IsolationAgent(HoldAgent):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.config = json.loads((directory / "probe.json").read_text())
        self.checked = False

    def reset(self, task):
        super().reset(task)
        self.task = task

    def act(self, observation):
        action = super().act(observation)
        if not self.checked:
            self.checked = True
            denied = {}

            def attempt(name, operation):
                try:
                    operation()
                except OSError:
                    denied[name] = True
                else:
                    denied[name] = False

            other, score = (
                Path(self.config["other_case"]),
                Path(self.config["runner_score"]),
            )
            attempt("other_read", other.read_bytes)
            attempt("other_write", lambda: other.write_bytes(b"modified"))
            attempt("score_read", score.read_bytes)
            attempt("score_write", lambda: score.write_bytes(b'{"score":1}'))
            link = self.directory / "outside-link"
            link.symlink_to(score)
            attempt("symlink_escape", lambda: link.write_bytes(b"modified"))
            image = Path(require_project_env("ROBORSI_MEDIA_DIR")) / Path(
                observation.media[0].ref
            ).relative_to("media")
            attempt("image_write", lambda: image.write_bytes(b"modified"))
            attempt(
                "image_hardlink", lambda: os.link(image, self.directory / "image-link")
            )
            attempt("root_write", lambda: Path("/opt/injected").write_text("modified"))

            def connect_docker():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect("/var/run/docker.sock")

            attempt("docker_socket", connect_docker)
            for index, endpoint in enumerate(self.config["benchmark_endpoints"]):
                try:
                    opener = urllib.request.build_opener(
                        urllib.request.ProxyHandler({})
                    )
                    with opener.open(endpoint + "/v1/tasks", timeout=0.5):
                        denied[f"benchmark_{index}"] = False
                except urllib.error.HTTPError as exc:
                    denied[f"benchmark_{index}"] = exc.code in {401, 403}
                except OSError:
                    denied[f"benchmark_{index}"] = True
            (self.directory / "result.json").write_text(
                '{"status":"success","score":1}'
            )
            (self.directory / "probe-result.json").write_text(
                json.dumps(
                    {
                        "denied": denied,
                        "allowed_credential": os.environ.get("TEST_MODEL_TOKEN")
                        == "integration-test-value",
                        "blocked_credential_absent": "TEST_RUNNER_SECRET"
                        not in os.environ,
                        "benchmark_token_absent": "ROBORSI_BENCHMARK_TOKEN"
                        not in os.environ,
                        "python_isolated": bool(sys.flags.isolated),
                        "simulator_absent": importlib.util.find_spec("libero") is None,
                        "episode_id": observation.episode_id,
                    }
                )
            )
        if self.config["mode"] == "forge_score":
            return NativeResult(
                task=self.task, episode_id=observation.episode_id, success=True, score=1
            )
        if self.config["mode"] == "cancel":
            time.sleep(30)
        return action

    def close(self):
        (self.directory / "agent-closed").write_text("closed")


def make_agent(directory):
    return IsolationAgent(directory)
