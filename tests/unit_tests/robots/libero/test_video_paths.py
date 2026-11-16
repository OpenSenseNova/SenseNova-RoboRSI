# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video output follows the caller's run and stays separate across environments."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.libero import env_server


def test_default_video_directories_are_distinct_and_use_system_temp(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(env_server.tempfile, "tempdir", str(tmp_path))
    paths = [
        Path(env_server.build_env_cfg().video_cfg.video_base_dir) for _ in range(2)
    ]
    assert paths[0] != paths[1]
    assert all(path.is_dir() and path.parent == tmp_path for path in paths)


@pytest.mark.parametrize("relative", [True, False])
def test_python_factory_resolves_video_directory(tmp_path, monkeypatch, relative):
    monkeypatch.chdir(tmp_path)
    suite = SimpleNamespace(get_task_init_states=lambda task: [0, 1])
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.libero.libero_env",
        SimpleNamespace(LiberoEnv=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.libero.utils",
        SimpleNamespace(
            benchmark=SimpleNamespace(get_benchmark=lambda name: lambda: suite)
        ),
    )
    path = Path("run/videos") if relative else tmp_path / "run/videos"
    native = env_server.make_env(1, 3, video_dir=path)
    assert native.cfg.video_cfg.save_video is True
    assert native.cfg.video_cfg.video_base_dir == str(tmp_path / "run/videos")
    assert native.cfg.specific_reset_id == 3


def test_python_factory_rejects_empty_initial_state_assets(monkeypatch):
    suite = SimpleNamespace(get_task_init_states=lambda task: [])
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.libero.libero_env",
        SimpleNamespace(LiberoEnv=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.libero.utils",
        SimpleNamespace(
            benchmark=SimpleNamespace(get_benchmark=lambda name: lambda: suite)
        ),
    )

    with pytest.raises(ValueError, match="libero_10_task task 2 has no initial states"):
        env_server.make_env(2, 3, suite_name="libero_10_task")


def test_standalone_server_accepts_relative_video_directory(monkeypatch):
    factory = Mock()
    facade = Mock()
    monkeypatch.setattr(env_server, "make_env", factory)
    monkeypatch.setattr(env_server, "LiberoEnvFacade", Mock(return_value=facade))
    monkeypatch.setattr(sys, "argv", ["env_server", "--video-dir", "run/videos"])
    env_server.main()
    assert factory.call_args.kwargs["video_dir"] == Path("run/videos")
    facade.close.assert_called_once_with()
