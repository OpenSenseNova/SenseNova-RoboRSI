# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run all Isaac/OmniGibson calls on one child process main thread."""

from __future__ import annotations

import argparse
import os
import queue
import threading
from concurrent.futures import Future
from pathlib import Path

from roborsi.evaluation.messages import Action, EpisodeStop, Task
from roborsi.evaluation.service_host import load_factory
from roborsi.utils.daemon import watch_parent_death
from roborsi.utils.rpc import RpcFacade
from robots.behavior_1k.episode import BehaviorEpisode


class EpisodeWorker(RpcFacade):
    """Queue RPC work without importing Isaac on an HTTP thread."""

    def __init__(self, episode: BehaviorEpisode) -> None:
        super().__init__()
        self.episode = episode
        self.requests = queue.Queue()
        self._rpc["request"] = self.request

    def request(self, method: str, payload: dict) -> dict:
        future = Future()
        self.requests.put((method, payload, future))
        return future.result()

    def run(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                try:
                    method, payload, future = self.requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if method == "start":
                        result = self.episode.start(
                            Task.from_dict(payload["task"]), payload["episode_id"]
                        )
                    elif method == "inspect":
                        result = self.episode.inspect(payload)
                    elif method == "step":
                        result = self.episode.step(Action.from_dict(payload))
                    elif method == "stop":
                        result = self.episode.stop(EpisodeStop.from_dict(payload))
                    elif method == "finish":
                        result = self.episode.finish()
                    elif method == "close":
                        self.episode.close()
                        future.set_result({})
                        continue
                    else:
                        raise ValueError(f"unknown worker method: {method}")
                    future.set_result(result.as_dict())
                except BaseException as exc:
                    future.set_exception(exc)
        finally:
            self.episode.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--factory", required=True)
    args = parser.parse_args()
    episode = BehaviorEpisode(
        args.directory,
        load_factory(args.factory)(),
        Path(os.environ.get("ROBORSI_BEHAVIOR_NATIVE", "/workspace/BEHAVIOR-1K")),
        Path(os.environ.get("ROBORSI_BEHAVIOR_ASSETS", "/roborsi/assets/behavior_1k")),
    )
    worker = EpisodeWorker(episode)

    def parent_died():
        worker._shutdown_event.set()
        timer = threading.Timer(5, lambda: os._exit(1))
        timer.daemon = True
        timer.start()

    watch_parent_death(parent_died)
    threading.Thread(
        target=worker.serve,
        kwargs={
            "transport": "http",
            "host": "127.0.0.1",
            "port": args.port,
            "parent_watch": False,
        },
        daemon=True,
    ).start()
    worker.run()


if __name__ == "__main__":
    main()
