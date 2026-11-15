# Copyright 2026 The SenseNova-RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The exported RoboCasa365 service sources cover its lazy runtime imports."""

import subprocess
import sys
from pathlib import Path

from benchmarks.package import prepare_context

# Paths the service reaches only at run time: env_server picks its renderer in
# main(), and the toolkit registers its common tools when it is built.
PROBE = """
import sys
from pathlib import Path
runtime = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(runtime))
import roborsi, robots
assert Path(roborsi.__file__).resolve().is_relative_to(runtime), roborsi.__file__
import robots.robocasa.benchmark_service
import robots.robocasa.roborsi_adapter
import robots.robocasa.env_server
import robots.robocasa.vla_server
from roborsi.utils.egl import configure_egl_device
from roborsi.dashboard.events import NullDashboardEventSink
from roborsi.session import EnvState
from roborsi.tools.toolkit import Toolkit
from roborsi.utils.logging import init_output_dir
init_output_dir(Path(sys.argv[2]))
toolkit = Toolkit(dashboard_events=NullDashboardEventSink(), state=EnvState(Path(sys.argv[2])))
assert "finish" in {spec["name"] for spec in toolkit.get_tools_spec()}
print("ok")
"""


def test_exported_service_sources_import_without_the_checkout(tmp_path):
    context = prepare_context(
        "robocasa365", tmp_path / "context", source_root=Path(__file__).parents[4]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE,
            str(context / "benchmark-runtime"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
