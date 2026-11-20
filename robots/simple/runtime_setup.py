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

"""Select the native device and isolate Kit caches without changing rendering."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
from pathlib import Path
from typing import Any

from robots.simple.contracts import SOURCES


def register_compatibility_task(alias: str, directory: Path) -> dict:
    """Load a hash-pinned original task removed from the selected upstream tree.

    The file is downloaded at image build time, outside the native checkout.
    Its implementation and reward predicate are kept byte-for-byte unchanged.
    """
    spec = SOURCES.get("compatibility", {}).get(alias)
    if spec is None:
        return {}
    path = directory / Path(spec["path"]).name
    if path.is_symlink() or not path.is_file():
        raise ValueError("install the pinned SIMPLE compatibility task first")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != spec["sha256"]:
        raise ValueError("SIMPLE compatibility task SHA256 differs")
    import gymnasium as gym

    env_id = "simple/" + SOURCES["tasks"][alias]["env_id"]
    name = "simple.tasks." + path.stem
    if env_id in gym.registry or name in sys.modules:
        raise ValueError("SIMPLE compatibility task is already registered")
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("cannot load SIMPLE compatibility task")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        # Execute the exact verified bytes, never an adjacent cached .pyc file.
        exec(compile(source, str(path), "exec"), module.__dict__)
        task_class = getattr(module, spec["class_name"])
        if (
            task_class.uid != spec["task_uid"]
            or task_class.metadata["max_episode_steps"]
            != SOURCES["tasks"][alias]["official_steps"]
        ):
            raise ValueError("SIMPLE compatibility task identity differs")
        gym.register(
            id=env_id,
            entry_point=spec["entry_point"],
            kwargs={"task": spec["task_uid"]},
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return dict(spec)


def prepare_kit_cleanup(directory: Path) -> None:
    """Make the owned Kit screenshot directory removable after its worker exits.

    Isaac 4.5 can create this directory with mode 000 under an unmapped container
    UID. Only add owner permissions to that exact path, never follow links or
    change another user's directory. The shared HTTP deleter remains unchanged.
    """
    path = directory
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        return
    parts = ("native", "kit", "data", "documents", "Kit", "shared", "screenshots")
    for part in parts:
        path = path / part
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            return
    path.chmod(stat.S_IMODE(metadata.st_mode) | stat.S_IRWXU)


def kit_config(config: dict, directory: Path, gpu: int) -> dict:
    """Preserve renderer settings while isolating caches and selecting one GPU."""
    if type(gpu) is not int or gpu < 0:
        raise ValueError("native GPU index must be a nonnegative integer")
    result = dict(config)
    extra = list(result.get("extra_args", []))
    if any(a == "--portable-root" or a.startswith("--portable-root=") for a in extra):
        raise ValueError("conflicting Kit portable root")
    result["extra_args"] = extra + [
        "--portable",
        "--portable-root",
        str(directory / "kit"),
        f"--/log/file={directory / 'kit.log'}",
        "--/app/extensions/fsWatcherEnabled=false",
        "--/plugins/carb.tasking.plugin/threadCount=16",
        "--/plugins/omni.tbb.globalcontrol/maxThreadCount=16",
    ]
    result.update(active_gpu=gpu, physics_gpu=gpu, multi_gpu=False)
    return result


def relocate_cuda_constants(module: Any, device: str) -> list[dict]:
    """Relocate traced CUDA allocation constants in memory, retaining weights."""
    import torch

    target = torch.device(device)
    edits = []

    def visit(block: Any, method: str) -> None:
        for node in block.nodes():
            if node.kind() == "prim::Constant" and len(list(node.outputs())) == 1:
                value = node.output().toIValue()
                if (
                    isinstance(value, torch.device)
                    and value.type == "cuda"
                    and value != target
                ):
                    edits.append(
                        {"method": method, "from": str(value), "to": str(target)}
                    )
                    node.s_("value", str(target))
            for child in node.blocks():
                visit(child, method)

    for name, child in module.named_modules():
        for method in child._c._method_names():
            graph = child._c._get_method(method).graph
            visit(graph, name + "." + method)
            graph.lint()
    return edits


def configure_runtime(directory: Path, controller: str) -> dict:
    """Configure the owned native worker before constructing an Isaac environment."""
    import simple.engines.isaac_app as app_helpers
    import torch

    gpu = int(os.environ.get("ROBORSI_SIMPLE_GPU_INDEX", "0"))
    if gpu < 0:
        raise ValueError("ROBORSI_SIMPLE_GPU_INDEX must be nonnegative")
    torch.set_num_threads(8)
    torch.cuda.set_device(gpu)
    facts: dict[str, Any] = {
        "gpu_index": gpu,
        "torch_device": torch.cuda.current_device(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amo_cuda_relocations": [],
    }
    create = app_helpers.create_simulation_app

    def create_owned(app_class: Any, **kwargs: Any) -> Any:
        def factory(config: dict) -> Any:
            config = kit_config(config, directory, gpu)
            facts["kit_config"] = config
            return app_class(config)

        return create(factory, **kwargs)

    app_helpers.create_simulation_app = create_owned
    if controller == "amo":
        import simple.robots.policy.AMO_Policy as amo

        initialize = amo.AMO_Policy.__init__

        def initialize_mapped(
            policy: Any,
            robot_type: str = "g1",
            device: str = "cuda",
            joint_names: list[str] | None = None,
        ) -> None:
            initialize(
                policy,
                robot_type=robot_type,
                device=f"cuda:{gpu}",
                joint_names=joint_names,
            )
            for module in (policy.policy_jit, policy.adapter):
                facts["amo_cuda_relocations"].extend(
                    relocate_cuda_constants(module, f"cuda:{gpu}")
                )

        amo.AMO_Policy.__init__ = initialize_mapped
    return facts
