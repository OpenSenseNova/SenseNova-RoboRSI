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

"""Serve SIMPLE RPC on its main thread using the shared runtime helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from roborsi.evaluation.messages import Action, Task
from roborsi.evaluation.service_host import load_factory
from roborsi.utils.rpc import RpcFacade
from roborsi.utils.rpc.main_thread_serve import MainThreadServeMixin
from robots.simple.episode import SimpleEpisode


class SimpleWorker(MainThreadServeMixin, RpcFacade):
    """Keep reset, rendering, action conversion and cleanup on one owner thread."""

    def __init__(self, episode: SimpleEpisode) -> None:
        super().__init__()
        self.episode = episode
        self._rpc["request"] = self.request

    def request(self, method: str, payload: dict) -> dict:
        """Dispatch only the four public lifecycle operations."""
        if method == "start":
            return self.episode.start(
                Task.from_dict(payload["task"]), payload["episode_id"]
            ).as_dict()
        if method == "step":
            return self.episode.step(Action.from_dict(payload)).as_dict()
        if method == "finish":
            return self.episode.finish().as_dict()
        if method == "close":
            self.episode.close()
            return {}
        raise ValueError("unsupported SIMPLE worker operation")


def main() -> None:
    """Start an isolated case worker without loading any Agent or model SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--factory", default="robots.simple.native:make_runtime")
    args = parser.parse_args()
    episode = SimpleEpisode(args.directory, load_factory(args.factory)(args.directory))
    worker = SimpleWorker(episode)
    try:
        worker.serve(
            transport="http", host="127.0.0.1", port=args.port, parent_watch=True
        )
    finally:
        episode.close()


if __name__ == "__main__":
    main()
