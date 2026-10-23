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

"""Resolve benchmark, model and Agent references from user-editable YAML."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from roborsi.utils.config_values import json_value, reject_secrets
from robots.libero.vla_client import OpenPiProfile, parse_profile

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"
BENCHMARKS = (
    "libero",
    "libero_pro",
    "libero_long_plus_plus",
    "libero_ultra_long",
    "robocasa365",
)
SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    benchmark: str
    model: Path
    agent: Path
    environment: dict
    case_timeout_s: float = 4500
    agent_overrides: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    endpoint: str | None = None
    token_env: str | None = None
    protocol: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    base_url: str | None
    model_name: str | None
    api_key: str | None = field(repr=False)
    openai_format: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    vla: OpenPiProfile | None = None
    sources: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)


def _read(path: Path, allowed: set[str], required: set[str]) -> dict:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError(f"Invalid configuration YAML: {path}") from None
    if not isinstance(document, dict) or any(not isinstance(k, str) for k in document):
        raise ValueError(f"Configuration must be a mapping: {path}")
    json_value(document)
    reject_secrets(document)
    if set(document) - allowed or required - set(document):
        raise ValueError(f"Unknown or missing configuration fields: {path}")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError(f"Configuration requires schema_version: 1: {path}")
    return document


def _reference(path: Path, value) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration reference must be a nonempty path: {path}")
    selected = (path.parent / value).resolve()
    if not selected.is_file():
        raise FileNotFoundError(
            f"Referenced configuration/resource is missing: {selected}"
        )
    return selected


def _fields(value, allowed: set[str], required: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f"Unknown or missing fields in {label}")
    return value


def _pin(value, pattern: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"{label} must use a pinned revision/digest")


def _environment(path: Path, name: str, value) -> dict:
    env = dict(
        _fields(
            value,
            {
                "kind",
                "gpus",
                "image",
                "native_bundle",
                "build_args",
                "catalog",
                "assets",
                "icl",
                "startup_timeout_s",
            },
            {"kind", "catalog"},
            "environment",
        )
    )
    kind = env["kind"]
    expected = (
        "libero"
        if name == "libero"
        else "robocasa365"
        if name == "robocasa365"
        else "libero_three"
    )
    if kind != expected:
        raise ValueError(f"environment.kind does not match {name}")
    gpus = env.get("gpus", "all" if kind != "libero" else 0)
    if gpus != "all" and (type(gpus) is not int or gpus < 0):
        raise ValueError("environment.gpus must be all or a nonnegative integer")
    env["gpus"] = gpus
    timeout = env.get("startup_timeout_s", 900)
    if type(timeout) not in {int, float} or timeout <= 0:
        raise ValueError("environment.startup_timeout_s must be positive")
    catalog = env["catalog"]
    key = (
        "suite" if kind == "libero" else "cells" if kind == "robocasa365" else "profile"
    )
    _fields(catalog, {key}, {key}, "environment.catalog")
    allowed = (
        SUITES
        if kind == "libero"
        else {"all", "seed1"}
        if kind == "robocasa365"
        else {"migration_subset"}
    )
    if not isinstance(catalog[key], str) or catalog[key] not in allowed:
        raise ValueError(
            f"Unsupported automatic catalog {key}; sealed ICL supports migration_subset only"
        )
    if kind == "libero_three":
        _pin(env.get("image"), r"[^\s]+@sha256:[0-9a-f]{64}", "environment.image")
        bundle = "libero_pro" if name == "libero_pro" else "libero_lilo"
        allowed_bundles = {bundle}
        if name == "libero_pro":
            allowed_bundles.add("libero_pro_image_tools")
        if env.get("native_bundle") not in allowed_bundles:
            raise ValueError("environment.native_bundle does not match benchmark")
        icl = dict(
            _fields(
                env.get("icl"),
                {"group", "root_env", "asset_lock"},
                {"group", "root_env", "asset_lock"},
                "environment.icl",
            )
        )
        group = {
            "libero_pro": "pro",
            "libero_long_plus_plus": "longpp",
            "libero_ultra_long": "ultra",
        }[name]
        if (
            icl["group"] != group
            or not isinstance(icl["root_env"], str)
            or not re.fullmatch(r"[A-Za-z_]\w*", icl["root_env"])
        ):
            raise ValueError("Invalid sealed ICL group/environment binding")
        icl["asset_lock"] = _reference(path, icl["asset_lock"])
        env["icl"] = icl
    else:
        build_args = env.get("build_args", {})
        argument = (
            "LIBERO_RUNTIME_IMAGE" if kind == "libero" else "ROBOCASA365_BASE_IMAGE"
        )
        _fields(build_args, {argument}, set(), "environment.build_args")
        for image in build_args.values():
            _pin(image, r"[^\s]+@sha256:[0-9a-f]{64}", "build image")
        env["build_args"] = build_args
        field_name = "assets" if kind == "libero" else "icl"
        keys = (
            {"repo_id", "revision"}
            if kind == "libero"
            else {"repo_id", "revision", "prefix", "manifest_sha256"}
        )
        download = _fields(env.get(field_name), keys, keys, f"environment.{field_name}")
        if not isinstance(download["repo_id"], str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", download["repo_id"]
        ):
            raise ValueError("Invalid Hugging Face dataset repository")
        _pin(download["revision"], r"[0-9a-f]{40}", "dataset revision")
        if kind == "robocasa365":
            _pin(download["manifest_sha256"], r"[0-9a-f]{64}", "ICL manifest SHA256")
            prefix = download["prefix"]
            if (
                not isinstance(prefix, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", prefix)
                or any(p in {".", ".."} for p in prefix.split("/"))
            ):
                raise ValueError("Invalid ICL dataset prefix")
    return env


def _model_document(path: Path) -> dict:
    document = _read(
        path,
        {
            "schema_version",
            "provider",
            "base_url",
            "model_name",
            "base_url_env",
            "model_name_env",
            "api_key_env",
            "openai_format",
            "vla",
            "sam3",
            "tool_runtime",
        },
        {"schema_version", "provider"},
    )
    if document["provider"] != "openai":
        raise ValueError("Model profiles require an OpenAI-compatible provider")
    if document.get("openai_format") is not None and (
        not isinstance(document["openai_format"], str)
        or document["openai_format"] not in {"chat", "responses"}
    ):
        raise ValueError("Model openai_format must be chat, responses or null")
    for key in ("base_url", "model_name"):
        if document.get(key) is not None and (
            not isinstance(document[key], str) or not document[key].strip()
        ):
            raise ValueError(f"Model {key} must be nonempty text or null")
    for key in ("base_url_env", "model_name_env", "api_key_env"):
        if key in document and (
            not isinstance(document[key], str)
            or not re.fullmatch(r"[A-Za-z_]\w*", document[key])
        ):
            raise ValueError(f"Model {key} must name an environment variable")
    return document


def load_benchmark(
    name: str, *, config_root: Path | None = None, path: Path | None = None
) -> BenchmarkConfig:
    """Load one benchmark and validate references without running any services."""
    if name not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {name}")
    custom = path is not None
    path = (
        Path(path).expanduser()
        if custom
        else (config_root or CONFIG_ROOT) / "benchmarks" / f"{name}.yaml"
    ).resolve()
    document = _read(
        path,
        {
            "schema_version",
            "benchmark",
            "model",
            "agent",
            "environment",
            "case_timeout_s",
            "agent_overrides",
            "evaluation",
            "endpoint",
            "token_env",
            "protocol",
        },
        {"schema_version", "benchmark"}
        if custom
        else {"schema_version", "benchmark", "model", "agent", "environment"},
    )
    if document["benchmark"] != name:
        raise ValueError(f"Benchmark name differs from configuration file: {path}")
    base = load_benchmark(name, config_root=config_root) if custom else None
    model = _reference(path, document["model"]) if "model" in document else base.model
    agent = _reference(path, document["agent"]) if "agent" in document else base.agent
    _model_document(model)
    timeout = document.get("case_timeout_s", base.case_timeout_s if base else 4500)
    if type(timeout) not in {int, float} or timeout <= 0:
        raise ValueError("case_timeout_s must be positive")
    overrides = _fields(
        document.get("agent_overrides", base.agent_overrides if base else {}),
        {"agent", "runtime", "parameters"},
        set(),
        "agent_overrides",
    )
    if any(not isinstance(v, dict) for v in overrides.values()):
        raise ValueError("agent_overrides sections must be mappings")
    endpoint = document.get("endpoint", base.endpoint if base else None)
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.strip()):
        raise ValueError("Benchmark endpoint must be an HTTP origin or null")
    token_env = document.get("token_env", base.token_env if base else None)
    if token_env is not None and (
        not isinstance(token_env, str) or not re.fullmatch(r"[A-Za-z_]\w*", token_env)
    ):
        raise ValueError("Benchmark token_env must name an environment variable")
    protocol = base.protocol if base else None
    if "protocol" in document:
        if name != "libero_pro":
            raise ValueError("protocol is supported only for libero_pro")
        from robots.libero.pro_protocol import (
            load_pro_config,
            validate_pro_80_config,
        )

        protocol = _reference(path, document["protocol"])
        validate_pro_80_config(load_pro_config(protocol))
    return BenchmarkConfig(
        path,
        name,
        model,
        agent,
        _environment(path, name, document["environment"])
        if "environment" in document
        else base.environment,
        timeout,
        overrides,
        _evaluation(path, document.get("evaluation", base.evaluation if base else {})),
        endpoint,
        token_env,
        protocol,
    )


def _evaluation(path: Path, value: dict) -> dict:
    selected = dict(
        _fields(
            value, {"tasks", "seeds", "selector", "parameters"}, set(), "evaluation"
        )
    )
    for key in ("tasks", "seeds"):
        if key not in selected:
            continue
        values = selected[key]
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(item, str) or not item.strip()
                if key == "tasks"
                else type(item) is not int or item < 0
                for item in values
            )
            or len(values) != len(set(values))
        ):
            raise ValueError(
                f"evaluation.{key} must be a nonempty unique list of {'task IDs' if key == 'tasks' else 'nonnegative integers'}"
            )
    if "parameters" in selected and not isinstance(selected["parameters"], dict):
        raise ValueError("evaluation.parameters must be a mapping")
    if "selector" in selected:
        reference = selected["selector"]
        if not isinstance(reference, str):
            raise ValueError("evaluation.selector must be path.py:function")
        filename, separator, function = reference.rpartition(":")
        if not separator or not re.fullmatch(r"[A-Za-z_]\w*", function):
            raise ValueError("evaluation.selector must be path.py:function")
        source = _reference(path, filename)
        if source.suffix != ".py":
            raise ValueError("evaluation.selector must reference a Python file")
        selected["selector"] = f"{source}:{function}"
    return selected


def catalog_setting(config: BenchmarkConfig, key: str) -> str:
    """Apply the existing environment overrides to a managed service catalog."""
    variable, choices = {
        "suite": ("ROBORSI_LIBERO_SUITE", SUITES),
        "cells": ("ROBORSI_ROBOCASA_CELLS", {"all", "seed1"}),
        "profile": (None, {"migration_subset"}),
    }[key]
    value = (
        os.environ.get(variable, config.environment["catalog"][key])
        if variable
        else config.environment["catalog"][key]
    )
    if value not in choices:
        raise ValueError(
            f"{variable or key} must be one of {', '.join(sorted(choices))}"
        )
    return value


def resolve_model(
    path: Path, *, base_url=None, model_name=None, api_key=None, openai_format=None
) -> ModelConfig:
    """Apply CLI > named environment > model YAML without persisting a key."""
    document = _model_document(path)
    sources = {}

    def choose(explicit, field_name, environment):
        if explicit is not None:
            sources[field_name] = "CLI"
            return explicit
        variable = document.get(field_name + "_env", environment)
        sources[field_name] = f"env:{variable}" if variable in os.environ else str(path)
        return os.environ.get(variable, document.get(field_name))

    tools = {}
    if (
        isinstance(document.get("vla"), dict)
        and document["vla"].get("backend") == "rlinf"
    ):
        from scripts.deploy.libero_tools import model_settings

        tools = model_settings(document)
    elif "sam3" in document or "tool_runtime" in document:
        raise ValueError("SAM3/tool_runtime requires the RLinf VLA profile")
    return ModelConfig(
        choose(base_url, "base_url", "OPENAI_BASE_URL"),
        choose(model_name, "model_name", "MODEL_NAME"),
        choose(api_key, "api_key", "OPENAI_API_KEY"),
        openai_format if openai_format is not None else document.get("openai_format"),
        document.get("api_key_env", "OPENAI_API_KEY"),
        parse_profile(document["vla"], base_dir=Path(path).parent)
        if document.get("vla") is not None and not tools
        else None,
        sources,
        tools,
    )


def pro_tool_mode(path: Path) -> str:
    """Validate the selected PRO Agent and return its tool mode."""
    from roborsi.agents.config import AgentConfig

    config = AgentConfig.from_file(path)
    document = config.document
    reject_secrets(document)
    llm = document.get("runtime", {}).get("llm", {})
    if llm.get("provider") != "openai":
        raise ValueError("--agent-config must declare an OpenAI-compatible model")
    # Validate only fields supported by the original runtime; do not silently
    # accept settings that require the removed shared-runtime patches.
    config.runtime()
    if config.factory != "robots.libero.roborsi_27b_agent:make_agent":
        raise ValueError("explicit PRO tool profiles require the 27B Agent config")
    parameters = config.parameters
    flags = [parameters.get(name) for name in ("enable_vla", "enable_sam3")]
    if any(type(flag) is not bool for flag in flags) or flags[0] != flags[1]:
        raise ValueError("--agent-config must enable or disable VLA/SAM3 together")
    if parameters.get("tool_transport") != "openpi_sam3":
        raise ValueError("--agent-config requires tool_transport: openpi_sam3")
    return "vla_sam3" if flags[0] else "no_vla_no_sam3"
