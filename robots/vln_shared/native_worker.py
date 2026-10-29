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

"""Python 3.8 compatible simulator worker, invoked as a file, not a package."""

import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path


def safe(value):
    """Convert native numpy values and unavailable metrics to finite JSON."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    """Keep simulator diagnostics off the RPC stream and release on EOF."""
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    episode = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request["command"]
                if command == "start":
                    if episode is not None:
                        raise ValueError("episode already started")
                    spec = request["spec"]
                    name = "lhpr_vln" if spec["dataset"] == "lhpr_vln" else "vlnce"
                    path = Path(__file__).resolve().parents[1] / name / "native.py"
                    module_spec = importlib.util.spec_from_file_location(
                        "vln_native", path
                    )
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
                    episode = module.Episode(spec)
                    result = episode.observe()
                elif command == "step":
                    feedback = episode.step(request["action"])
                    result = {**episode.observe(), "feedback": feedback}
                elif command == "finish":
                    result = episode.evaluate()
                    result.setdefault("provenance", {})["runtime"] = {
                        "python": sys.version,
                        "executable": sys.executable,
                        "modules": {
                            name: getattr(sys.modules.get(name), "__version__", None)
                            for name in ("habitat", "habitat_sim", "numpy", "torch")
                        },
                    }
                elif command == "close":
                    break
                else:
                    raise ValueError("unknown native operation")
                output.write(
                    json.dumps({"result": safe(result)}, allow_nan=False) + "\n"
                )
            except Exception as exc:
                traceback.print_exc()
                output.write(
                    json.dumps({"error": type(exc).__name__ + ": " + str(exc)}) + "\n"
                )
    finally:
        if episode is not None:
            episode.close()


if __name__ == "__main__":
    main()
