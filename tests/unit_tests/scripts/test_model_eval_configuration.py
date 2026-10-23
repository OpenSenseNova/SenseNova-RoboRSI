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

"""User-editable configuration composes without Docker, network or model calls."""

import importlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def configuration():
    path = ROOT / "scripts/evaluation/configuration.py"
    assert path.exists(), "the unified configuration loader is missing"
    return importlib.import_module("scripts.evaluation.configuration")


@pytest.mark.parametrize(
    "name,agent",
    [
        ("libero", "libero/standard.yaml"),
        ("libero_pro", "libero/pro.yaml"),
        ("libero_long_plus_plus", "libero/long_plus_plus.yaml"),
        ("libero_ultra_long", "libero/ultra_long.yaml"),
        ("robocasa365", "robocasa365/icl.yaml"),
    ],
)
def test_canonical_benchmark_references_compose(configuration, name, agent):
    from roborsi.agents.config import AgentConfig

    config = configuration.load_benchmark(name)
    assert config.agent == ROOT / "configs/agents" / agent
    assert config.model == ROOT / "configs/models/default.yaml"
    assert config.path == ROOT / "configs/benchmarks" / f"{name}.yaml"
    preset = AgentConfig.from_file(config.agent)
    if name.startswith("libero_"):
        assert preset.resources["context_map"].is_file()
        assert preset.document["agent"]["reasoning_effort"] == "max"
        assert ROOT / "robots/libero/three_benchmark_agent.py" in preset.input_files


@pytest.fixture
def config_tree(tmp_path):
    root = tmp_path / "configs"
    (root / "benchmarks").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "agents").mkdir()
    (root / "agents/test.yaml").write_text("schema_version: 1\n")
    model = {
        "schema_version": 1,
        "provider": "openai",
        "base_url": "http://yaml.example/v1",
        "model_name": "yaml-model",
        "base_url_env": "TEST_MODEL_URL",
        "model_name_env": "TEST_MODEL_NAME",
        "api_key_env": "TEST_MODEL_KEY",
        "openai_format": "chat",
    }
    benchmark = {
        "schema_version": 1,
        "benchmark": "libero",
        "model": "../models/test.yaml",
        "agent": "../agents/test.yaml",
        "environment": {
            "kind": "libero",
            "gpus": 0,
            "build_args": {},
            "catalog": {"suite": "libero_goal"},
            "assets": {"repo_id": "RLinf/LIBERO-assets", "revision": "a" * 40},
        },
    }

    def write(document=benchmark):
        (root / "benchmarks/libero.yaml").write_text(yaml.safe_dump(document))
        (root / "models/test.yaml").write_text(yaml.safe_dump(model))

    write()
    return root, benchmark, model, write


def test_references_resolve_from_yaml_even_from_another_cwd(
    configuration, config_tree, tmp_path, monkeypatch
):
    root, _, _, _ = config_tree
    monkeypatch.chdir(tmp_path.parent)
    config = configuration.load_benchmark("libero", config_root=root)
    assert config.agent == root / "agents/test.yaml"
    assert config.model == root / "models/test.yaml"
    assert configuration.catalog_setting(config, "suite") == "libero_goal"


def test_model_cli_environment_yaml_precedence_and_secret_reference(
    configuration, config_tree, monkeypatch
):
    root, _, _, _ = config_tree
    config = configuration.load_benchmark("libero", config_root=root)
    for name in ("TEST_MODEL_URL", "TEST_MODEL_NAME", "TEST_MODEL_KEY"):
        monkeypatch.delenv(name, raising=False)
    model = configuration.resolve_model(config.model)
    assert model.base_url == "http://yaml.example/v1"
    assert model.model_name == "yaml-model"
    assert model.api_key is None
    monkeypatch.setenv("TEST_MODEL_URL", "http://env.example/v1")
    monkeypatch.setenv("TEST_MODEL_NAME", "env-model")
    monkeypatch.setenv("TEST_MODEL_KEY", "secret-env-key")
    model = configuration.resolve_model(config.model)
    assert model.base_url == "http://env.example/v1"
    assert model.model_name == "env-model"
    assert model.api_key == "secret-env-key"
    assert "secret-env-key" not in repr(model)
    cli = configuration.resolve_model(
        config.model,
        base_url="http://cli.example/v1",
        model_name="cli-model",
        api_key="cli-key",
        openai_format="responses",
    )
    assert (cli.base_url, cli.model_name, cli.api_key, cli.openai_format) == (
        "http://cli.example/v1",
        "cli-model",
        "cli-key",
        "responses",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"benchmark": "libero_pro"},
        {"agent": "../agents/missing.yaml"},
        {"unknown": "ignored?"},
        {"environment": {"kind": "unsupported"}},
    ],
)
def test_invalid_benchmark_config_is_rejected(configuration, config_tree, change):
    root, benchmark, _, write = config_tree
    write({**benchmark, **change})
    with pytest.raises((ValueError, FileNotFoundError)):
        configuration.load_benchmark("libero", config_root=root)


@pytest.mark.parametrize(
    "patch",
    [
        {"api_key": "secret-in-yaml"},
        {"provider": "unsupported"},
        {"openai_format": "invalid"},
        {"base_url_env": "BAD NAME"},
        {"unknown": True},
    ],
)
def test_invalid_model_config_is_rejected(configuration, config_tree, patch):
    root, _, model, write = config_tree
    model.update(patch)
    write()
    with pytest.raises(ValueError):
        configuration.resolve_model(root / "models/test.yaml")


def test_catalog_environment_overrides_yaml(configuration, config_tree, monkeypatch):
    root, _, _, _ = config_tree
    config = configuration.load_benchmark("libero", config_root=root)
    monkeypatch.setenv("ROBORSI_LIBERO_SUITE", "libero_90")
    assert configuration.catalog_setting(config, "suite") == "libero_90"
    monkeypatch.setenv("ROBORSI_LIBERO_SUITE", "typo")
    with pytest.raises(ValueError, match="ROBORSI_LIBERO_SUITE"):
        configuration.catalog_setting(config, "suite")


def test_pins_live_in_benchmark_yaml_instead_of_bootstrap(configuration):
    pro = configuration.load_benchmark("libero_pro")
    casa = configuration.load_benchmark("robocasa365")
    assert "@sha256:" in pro.environment["image"]
    assert (
        casa.environment["icl"]["revision"]
        == "e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c"
    )
    assert pro.environment["icl"]["asset_lock"].is_file()
    source = (ROOT / "scripts/evaluation/bootstrap.py").read_text()
    assert pro.environment["image"] not in source
    assert casa.environment["icl"]["revision"] not in source
    assert "secret" not in json.dumps(yaml.safe_load(casa.path.read_text()))


def test_pro_80_uses_separate_protocol_without_changing_original_pro(configuration):
    from robots.libero.pro_protocol import load_pro_config

    original = configuration.load_benchmark("libero_pro")
    selected = configuration.load_benchmark(
        "libero_pro", path=ROOT / "configs/benchmarks/libero_pro_80.yaml"
    )
    assert original.protocol is None
    assert original.agent.name == "pro.yaml"
    assert original.agent_overrides["agent"]["reasoning_effort"] == "none"
    assert selected.benchmark == "libero_pro"
    assert selected.agent.name == "sensenova_roborsi_27b_no_tools.yaml"
    assert original.environment["native_bundle"] == "libero_pro"
    assert selected.environment == {
        **original.environment,
        "native_bundle": "libero_pro_image_tools",
    }
    assert selected.agent_overrides == {}
    protocol = load_pro_config(selected.protocol)
    assert protocol["init_state_ids"] == [0]
    assert protocol["seed"] == 0 and protocol["max_env_steps"] == 2000


def test_rlinf_model_settings_are_separate_from_agent(configuration):
    selected = configuration.resolve_model(ROOT / "configs/models/roborsi_27b.yaml")
    assert selected.vla is None
    assert selected.tools["deployment"]["vla"]["repo_id"] == (
        "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT"
    )
    assert selected.tools["deployment"]["sam3"]["repo_id"] == "facebook/sam3"
    assert "libero_pro_80" not in configuration.BENCHMARKS
