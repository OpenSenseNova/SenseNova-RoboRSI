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

"""Only the requested Target50 task's verified seed-0 memory reaches the model."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from roborsi.evaluation.messages import Action, Observation, StepResult, Task

NO_MEMORY = [
    "HeatKebabSandwich",
    "PanTransfer",
    "PortionHotDogs",
    "SeparateFreezerRack",
    "WaffleReheat",
    "WashFruitColander",
    "WeighIngredients",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def release(root):
    """Build an independent two-task release with the published file schema."""
    entries = {}

    def write(name, value):
        data = value.encode() if isinstance(value, str) else json.dumps(value).encode()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries[name] = sha(data)
        return entries[name]

    packages = {}
    for task, content in [
        ("OpenDrawer", "Only the drawer demonstration"),
        ("TurnOffStove", "Only the stove demonstration"),
    ]:
        prefix = f"tasks/{task}"
        result_path = f"robocasa/results/{task}_s0.json"
        recipe_path = f"robocasa/results/recipe_{task}_s0.jsonl"
        payload = {
            result_path: write(
                f"{prefix}/payload/{result_path}",
                {"suite": "robocasa365", "task": task, "seed": 0, "notes": content},
            ),
            recipe_path: write(
                f"{prefix}/payload/{recipe_path}", '{"action":"rldx_arm"}\n'
            ),
        }
        manifest_sha = write(
            f"{prefix}/manifest.json",
            {
                "schema_version": 1,
                "benchmark_id": "robocasa365",
                "benchmark": "robocasa365",
                "regime": "target50_same_task_seed0",
                "task": task,
                "split": "atomic",
                "source_repo_id": "RLinf/RPent-memory",
                "source_repo_type": "dataset",
                "source_revision": "551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b",
                "source_paths": list(payload),
                "payload_files": payload,
                "memory_root": "payload",
                "evaluation_trajectories_used": False,
            },
        )
        packages[task] = {
            "path": prefix,
            "split": "atomic",
            "manifest_sha256": manifest_sha,
            "payload_files": payload,
        }
    write(
        "knowledge_packages.json",
        {
            "schema_version": 1,
            "benchmark": "robocasa365",
            "benchmark_id": "robocasa365",
            "version": "v1.0.0",
            "regime": "target50_same_task_seed0",
            "memory_policy": {
                "scope": "same_task_seed_0",
                "use_global_memory": False,
                "use_cross_task_memory": False,
                "tasks_without_memory": NO_MEMORY,
            },
            "case_map": {
                "atomic/OpenDrawer_s1": "OpenDrawer",
                "atomic/TurnOffStove_s1": "TurnOffStove",
            },
            "packages": packages,
        },
    )
    checksums = "".join(f"{value}  {name}\n" for name, value in sorted(entries.items()))
    (root / "MANIFEST.sha256").write_bytes(checksums.encode())
    return sha(checksums.encode())


def task(name="OpenDrawer", seed=1, split="atomic"):
    return Task(
        benchmark="robocasa365",
        task_id=name,
        seed=seed,
        instruction="Current task and live observations are authoritative.",
        max_steps=400,
        native={
            "protocol": "robocasa-harness-vla-v1",
            "environment_split": "target",
            "split": split,
            "timeout_s": 1800 if split == "atomic" else 3600,
        },
    )


@pytest.fixture
def loader():
    module_path = Path(__file__).parents[4] / "robots/robocasa/model_eval_agent.py"
    assert module_path.exists(), "RoboCasa ICL loader must be implemented"
    return importlib.import_module("robots.robocasa.model_eval_agent")


def test_task_memory_is_in_initial_context_without_other_tasks(tmp_path, loader):
    digest = release(tmp_path)
    context, evidence = loader.load_task_context(
        tmp_path, task(), checksums_sha256=digest
    )
    text = "\n".join(context)
    assert "Only the drawer demonstration" in text
    assert "Only the stove demonstration" not in text
    assert "TurnOffStove" not in text
    assert evidence["status"] == "loaded"
    assert evidence["case_key"] == "atomic/OpenDrawer_s1"
    assert evidence["source_seed"] == 0
    assert set(evidence["payload_sha256"]) == {
        "robocasa/results/OpenDrawer_s0.json",
        "robocasa/results/recipe_OpenDrawer_s0.jsonl",
    }


@pytest.mark.parametrize("name", NO_MEMORY)
def test_declared_no_memory_tasks_keep_empty_context(tmp_path, loader, name):
    digest = release(tmp_path)
    context, evidence = loader.load_task_context(
        tmp_path, task(name, split="composite_unseen"), checksums_sha256=digest
    )
    assert context == ()
    assert evidence["status"] == "unavailable_declared"
    assert evidence["task_id"] == name


@pytest.mark.parametrize(
    "selected",
    [task(seed=0), task(seed=2), task(split="composite_seen"), task("OpenCabinet")],
)
def test_missing_or_mismatched_case_is_not_silently_evaluated(
    tmp_path, loader, selected
):
    digest = release(tmp_path)
    with pytest.raises(ValueError):
        loader.load_task_context(tmp_path, selected, checksums_sha256=digest)


@pytest.mark.parametrize(
    "changed",
    [
        "MANIFEST.sha256",
        "knowledge_packages.json",
        "tasks/OpenDrawer/manifest.json",
        "tasks/OpenDrawer/payload/robocasa/results/OpenDrawer_s0.json",
    ],
)
def test_modified_release_fails_before_injecting_memory(tmp_path, loader, changed):
    digest = release(tmp_path)
    with (tmp_path / changed).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="(?i)(checksum|hash)"):
        loader.load_task_context(tmp_path, task(), checksums_sha256=digest)


def test_checksum_manifest_cannot_read_outside_package(tmp_path, loader):
    digest = release(tmp_path / "package")
    del digest
    (tmp_path / "outside.txt").write_text("outside")
    manifest = tmp_path / "package/MANIFEST.sha256"
    data = manifest.read_bytes() + f"{sha(b'outside')}  ../outside.txt\n".encode()
    manifest.write_bytes(data)
    with pytest.raises(ValueError, match="(?i)(path|relative)"):
        loader.load_task_context(
            tmp_path / "package", task(), checksums_sha256=sha(data)
        )


def test_reset_injects_context_and_records_evidence_before_agent_start(
    tmp_path, loader
):
    package = tmp_path / "package"
    digest = release(package)
    selected = task()

    class ModelBoundary:
        evaluation_initial_context = ("Existing user context",)

        def reset(self, actual):
            assert actual == selected
            assert "Only the drawer demonstration" in "\n".join(
                self.evaluation_initial_context
            )
            assert "Only the stove demonstration" not in "\n".join(
                self.evaluation_initial_context
            )
            evidence = json.loads((tmp_path / "case/icl-evidence.json").read_text())
            assert evidence["status"] == "loaded"

    agent = loader.RoboCasaMemoryAgent(
        ModelBoundary(), tmp_path / "case", package, checksums_sha256=digest
    )
    agent.reset(selected)
    assert agent.delegate.evaluation_initial_context[0] == "Existing user context"


def test_factory_uses_the_configured_icl_pin_and_records_its_revision(tmp_path, loader):
    from types import SimpleNamespace

    package = tmp_path / "package"
    digest = release(package)
    delegate = SimpleNamespace(evaluation_initial_context=(), reset=lambda task: None)
    config = SimpleNamespace(
        resources={"icl_package": package},
        parameters={"icl_manifest_sha256": digest, "icl_revision": "a" * 40},
        embodied_agent=lambda case_dir: delegate,
    )
    agent = loader.make_agent(tmp_path / "case", config=config)
    agent.reset(task())
    evidence = json.loads((tmp_path / "case/icl-evidence.json").read_text())
    assert evidence["checksums_sha256"] == digest
    assert evidence["package_revision"] == "a" * 40
    assert "Only the drawer demonstration" in "\n".join(
        delegate.evaluation_initial_context
    )


class Planner:
    """Stand-in EmbodiedAgent recording how the episode ends."""

    def __init__(self, actions):
        self.evaluation_initial_context = ()
        self.planner_timeout_s = 1800
        self.actions = list(actions)
        self.ended, self.closed = [], False

    def reset(self, task):
        pass

    def act(self, observation):
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return Action(
            benchmark=observation.benchmark,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
            data=action,
        )

    def observe_end(self, step):
        self.ended.append(step)

    def close(self):
        self.closed = True


def observation(step):
    return Observation(benchmark="robocasa365", episode_id="e", step_index=step)


def budget_agent(tmp_path, loader, actions, now, **options):
    package = tmp_path / "package"
    digest = release(package)
    planner = Planner(actions)
    agent = loader.RoboCasaMemoryAgent(
        planner,
        tmp_path / "case",
        package,
        checksums_sha256=digest,
        clock=lambda: now[0],
        **options,
    )
    agent.reset(task())
    return agent, planner


def test_time_limit_ends_the_episode_so_the_scene_is_scored(tmp_path, loader):
    now = [0.0]
    agent, planner = budget_agent(
        tmp_path, loader, [{"primitive": "view_env_state"}] * 2, now
    )
    assert planner.planner_timeout_s == 1800 + loader.PLANNER_GRACE_S
    assert agent.act(observation(0)).data == {"primitive": "view_env_state"}
    now[0] = 1800.0
    action = agent.act(observation(1))
    assert action.data["primitive"] == "finish" and action.step_index == 1
    end = StepResult(
        observation=observation(2),
        terminated=True,
        truncated=False,
        termination_reason="finish",
    )
    agent.observe_end(end)
    assert planner.closed and planner.ended == []
    record = json.loads((tmp_path / "case/budget-finish.json").read_text())
    assert record == {"step_index": 1, "reason": "Target50 time limit reached"}


def test_turn_limit_ends_the_episode_so_the_scene_is_scored(tmp_path, loader):
    ended = RuntimeError(loader.PLANNER_ENDED)
    agent, _ = budget_agent(tmp_path, loader, [ended], [0.0])
    action = agent.act(observation(7))
    assert action.data["summary"] == "planner turn limit reached"
    agent, _ = budget_agent(
        tmp_path / "off", loader, [ended], [0.0], finish_on_budget=False
    )
    with pytest.raises(RuntimeError, match="ended before"):
        agent.act(observation(0))


def test_agent_preset_uses_the_target50_prompt(tmp_path):
    from roborsi.agents.config import AgentConfig

    root = Path(__file__).parents[4]
    config = AgentConfig.load(root / "configs/agents/robocasa365/icl.yaml")
    agent = config.embodied_agent(tmp_path)
    assert "RoboCasa365" in agent.system_prompt
    assert "roborsi_action" in agent.system_prompt
    assert agent.invalid_action_retries == 10
    assert agent.max_tokens == 32768
