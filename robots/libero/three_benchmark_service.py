# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Standalone RoboRSI providers for the three extended LIBERO benchmarks."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from roborsi.benchmarks.runtime_source import benchmark_source_digest
from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.messages import BenchmarkRuntimeInfo, Task
from roborsi.utils.config import get_project_env
from robots.libero.benchmark_catalog import ordered_cases, pro_cases
from robots.libero.lilo_task import build_task_contract
from robots.libero.three_benchmark_adapter import ACTION_SCHEMA, ThreeBenchmarkAdapter

_NAMES = frozenset({"libero_pro", "libero_long_plus_plus", "libero_ultra_long"})
_LIBERO_PATH_ENV = {
    "assets": "ROBORSI_LIBERO_ASSETS_ROOT",
    "benchmark_root": "ROBORSI_LIBERO_BENCHMARK_ROOT",
    "datasets": "ROBORSI_LIBERO_DATASETS_ROOT",
    "bddl_files": "ROBORSI_LIBERO_BDDL_ROOT",
    "init_states": "ROBORSI_LIBERO_INIT_ROOT",
}


def _flag(name: str, default: bool = False) -> bool:
    value = get_project_env(name, "1" if default else "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def _platform() -> dict[str, str]:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next(
            (
                line.partition(":")[2].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "system": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
    }


def _ensure_libero_config() -> None:
    """Materialize LIBERO's mutable path file under the container tmpfs."""
    configured = get_project_env("LIBERO_CONFIG_PATH")
    if not configured:
        return
    root = Path(configured).expanduser()
    target = root / "config.yaml"
    if target.is_file():
        return
    paths = {
        key: os.environ[name]
        for key, name in _LIBERO_PATH_ENV.items()
        if get_project_env(name)
    }
    if not paths:
        return
    missing = sorted(set(_LIBERO_PATH_ENV) - set(paths))
    if missing:
        raise ValueError(f"incomplete container LIBERO paths: {missing}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _ordered_contracts(name: str) -> list[dict[str, Any]]:
    _ensure_libero_config()
    import lilo_vla
    import lilo_vla.benchmark  # noqa: F401
    from libero.libero.benchmark import get_benchmark

    suite_name = (
        "libero_long_plus_plus_all"
        if name == "libero_long_plus_plus"
        else "ultra_long_all"
    )
    suite = get_benchmark(suite_name)()
    root = Path(
        get_project_env("ROBORSI_LILO_ROOT", Path(lilo_vla.__file__).parents[1])
    )
    config = json.loads(
        (root / "configs/tasks_and_skills.json").read_text(encoding="utf-8")
    )
    contracts = []
    for index in range(suite.get_num_tasks()):
        descriptor = suite.get_task(index)
        bddl = root / "configs/bddl" / descriptor.problem_folder / descriptor.bddl_file
        contracts.append(build_task_contract(config, suite_name, descriptor, bddl))
    return contracts


class ThreeLiberoBenchmark:
    """Discover one selected evaluation and create private native adapters."""

    def __init__(self) -> None:
        self.name = get_project_env("ROBORSI_LIBERO_BENCHMARK", "libero_pro")
        if self.name not in _NAMES:
            raise ValueError(
                f"ROBORSI_LIBERO_BENCHMARK must be one of {sorted(_NAMES)}"
            )
        self.profile = get_project_env("ROBORSI_LIBERO_PROFILE", "official")
        if self.profile not in {"official", "migration_subset"}:
            raise ValueError(
                "ROBORSI_LIBERO_PROFILE must be official or migration_subset"
            )
        default_bundle = "libero_pro" if self.name == "libero_pro" else "libero_lilo"
        self.source_bundle = get_project_env(
            "ROBORSI_LIBERO_SOURCE_BUNDLE", default_bundle
        )
        allowed = {default_bundle}
        if self.name == "libero_pro":
            allowed.add("libero_pro_image_tools")
        if self.source_bundle not in allowed:
            raise ValueError("source bundle does not match benchmark")
        config_path = get_project_env("ROBORSI_PRO_EVAL_CONFIG")
        if (
            config_path or _flag("ROBORSI_IMAGE_OBSERVATION_TOOLS")
        ) and self.source_bundle != "libero_pro_image_tools":
            raise ValueError(
                "custom protocol/tools require the libero_pro_image_tools source bundle"
            )
        self.pro_config = None
        if config_path:
            if self.name != "libero_pro":
                raise ValueError("PRO config cannot be used with another benchmark")
            from robots.libero.pro_protocol import load_pro_config

            self.pro_config = load_pro_config(config_path)
        self.image_observation_tools = _flag("ROBORSI_IMAGE_OBSERVATION_TOOLS")
        if self.image_observation_tools:
            if self.name != "libero_pro":
                raise ValueError("Image observation tools currently require PRO")
            from robots.libero.image_tools import image_manifest

            self.image_tool_manifest = image_manifest()
        self.tool_urls = {}
        if self.image_observation_tools:
            from robots.libero.image_tools import tool_endpoints

            self.tool_urls = tool_endpoints()
        self.plus9 = _flag("ROBORSI_LIBERO_PLUS9", default=False)

    def _cases(self):
        _ensure_libero_config()
        if self.name == "libero_pro":
            # The RLinf compatibility registry owns the Task/Swap suite names;
            # upstream LIBERO's registry contains only the base suites.
            from rlinf.envs.libero.utils import benchmark

            instructions = {
                suite_name: [
                    benchmark.get_benchmark(suite_name)().get_task(index).language
                    for index in range(10)
                ]
                for suite_name in (
                    f"libero_{family}_{variant}"
                    for family in ("spatial", "object", "goal", "10")
                    for variant in ("task", "swap")
                )
            }
            if self.pro_config is not None:
                from robots.libero.pro_protocol import pro_cases as configured_cases

                return configured_cases(
                    instructions,
                    profile=self.profile,
                    plus9_on_failure=self.plus9,
                    config=self.pro_config,
                )
            return pro_cases(
                instructions, profile=self.profile, plus9_on_failure=self.plus9
            )
        contracts = _ordered_contracts(self.name)
        return ordered_cases(
            self.name,
            [item["instruction"] for item in contracts],
            [len(item["skills"]) for item in contracts],
            profile=self.profile,
            plus9_on_failure=self.plus9,
        )

    def tasks(self) -> list[Task]:
        observation_channels = ["main_images", "wrist_images", "states"]
        if self.name != "libero_pro":
            observation_channels.extend(
                ["robot_state_feedback", "instance_pose_feedback"]
            )
        capabilities = {
            "action_schema": ACTION_SCHEMA,
            "observation_channels": observation_channels,
        }
        if self.image_observation_tools:
            from robots.libero.image_tools import PROFILE, action_schema

            capabilities.update(
                action_schema=action_schema(bool(self.tool_urls)),
                image_observation_tools=PROFILE,
                image_tool_source=self.image_tool_manifest,
                observation_channels=[
                    "agentview",
                    "agentview_high",
                    "wrist_high",
                    "states",
                    "camera_state",
                ],
            )
        return [
            Task(
                benchmark=case.benchmark,
                task_id=case.task_id,
                seed=getattr(case, "seed", 0),
                instruction=case.instruction,
                max_steps=case.max_env_steps,
                native=case.native,
                capabilities=capabilities,
            )
            for case in self._cases()
        ]

    def runtime_info(self) -> BenchmarkRuntimeInfo | None:
        bundle = "libero_pro" if self.name == "libero_pro" else "libero_lilo"
        commit_name = (
            "ROBORSI_LIBERO_PRO_COMMIT"
            if bundle == "libero_pro"
            else "ROBORSI_LILO_COMMIT"
        )
        commit = get_project_env(commit_name)
        if commit is None:
            return None
        score = (
            "native_success" if self.name == "libero_pro" else "ordered_prefix_at_1x"
        )
        aggregation = (
            "task_weighted_success_rate"
            if self.name == "libero_pro"
            else "task_weighted_mean"
        )
        digest = benchmark_source_digest(
            Path(__file__).resolve().parents[2], self.source_bundle
        )
        return BenchmarkRuntimeInfo(
            benchmark=self.name,
            benchmark_version=get_project_env(
                "ROBORSI_LIBERO_VERSION", "public-pinned-v1"
            ),
            benchmark_commit=commit,
            adapter_version=f"roborsi-source-sha256:{digest}",
            evaluation={
                "success_metric": (
                    "final_goal_success_at_1x"
                    if self.name == "libero_pro"
                    else "ordered_success_at_1x"
                ),
                "score_metric": score,
                "aggregation": aggregation,
                "error_denominator": "expected",
            },
            platform=_platform(),
        )

    def assets(self) -> list[dict[str, Any]]:
        from roborsi.benchmarks.registry import directory_sha256

        assets = []
        names = ("libero_pro",) if self.name == "libero_pro" else ("libero",)
        for name in names:
            root = Path("/roborsi/assets") / name
            if root.is_dir():
                assets.append(
                    {
                        "name": name,
                        "kind": "directory",
                        "sha256": directory_sha256(root),
                    }
                )
        return assets

    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> ThreeBenchmarkAdapter:
        if self.image_observation_tools:
            from robots.libero.image_tool_adapter import ObservationAdapter

            return ObservationAdapter(directory, artifacts, tool_urls=self.tool_urls)
        if self.tool_urls:
            raise ValueError("VLA/SAM3 require the image tool profile")
        return ThreeBenchmarkAdapter(directory, artifacts)


def make_three_libero_benchmark() -> ThreeLiberoBenchmark:
    return ThreeLiberoBenchmark()
