# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Real batch provenance/resume and HTTP probes behind the evaluation front end."""

import io
import json
import os
import subprocess
import threading
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.unit_tests.scripts.test_evaluate_model import (
    arguments,
    catalog_server,
    evaluation_report,
)
from tests.unit_tests.scripts.test_evaluate_model import registry as registry
from tests.unit_tests.scripts.test_evaluate_model import runner as runner


@pytest.fixture
def native_batch(runner, monkeypatch):
    from roborsi.evaluation import cli
    from tests.unit_tests.roborsi.evaluation.test_batch import _PublishedCases

    monkeypatch.setattr(cli, "DockerExecutor", lambda: object())
    monkeypatch.setattr(
        cli,
        "CaseOrchestrator",
        lambda registry, store, launcher, executor: _PublishedCases(registry, store),
    )

    def execute(command, **kwargs):
        output, errors = io.StringIO(), io.StringIO()
        with monkeypatch.context() as context:
            for key, value in kwargs["env"].items():
                context.setenv(key, value)
            with redirect_stdout(output), redirect_stderr(errors):
                try:
                    code = cli.main(command[command.index("eval") :])
                except SystemExit as exc:
                    code = exc.code
        return subprocess.CompletedProcess(
            command, code, output.getvalue(), errors.getvalue()
        )

    monkeypatch.setattr(runner, "run_child", execute)


def test_invocation_resume_reuses_native_results_and_saves_redacted_summary(
    runner, registry, tmp_path, native_batch
):
    from roborsi.evaluation.store import ResultStore

    args = [
        *arguments(registry, tmp_path),
        "--benchmark",
        "libero",
        "--api-key",
        "private-fixture-credential",
    ]
    assert runner.main(args) == 0
    assert runner.main(args + ["--run-id", "retry", "--resume-from", "trial"]) == 0
    store = ResultStore(tmp_path / "results")
    assert store.list_cases("retry-libero") == {}
    snapshot = (tmp_path / "results/retry/resolved_config.json").read_text()
    assert "private-fixture-credential" not in snapshot
    assert "--resume-from" in snapshot and "trial-libero" in snapshot
    summary = json.loads((tmp_path / "results/retry/summary.json").read_text())
    assert summary["total"] == 1 and summary["success_rate"] == 1
    assert summary["complete"] is True
    assert "libero" in summary["groups"][0]["benchmarks"]


def test_native_resume_rejects_changed_model_before_cases_run(
    runner, registry, tmp_path, native_batch
):
    args = [
        *arguments(registry, tmp_path),
        "--benchmark",
        "libero",
        "--api-key",
        "EMPTY",
    ]
    assert runner.main(args) == 0
    assert (
        runner.main(
            args
            + [
                "--run-id",
                "retry",
                "--resume-from",
                "trial",
                "--model-name",
                "changed-model",
            ]
        )
        != 0
    )
    assert not (tmp_path / "results/retry-libero").exists()


@pytest.mark.parametrize("api_format", ["chat", "responses"])
def test_model_probe_checks_image_tool_schema_and_authentication(api_format):
    from scripts.evaluation.configuration import ModelConfig
    from scripts.evaluation.preflight import probe_model

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{}], "output": []}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        probe_model(
            ModelConfig(
                f"http://127.0.0.1:{server.server_port}/v1",
                "fixture-model",
                "fixture-key",
            ),
            api_format,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    path, auth, body = requests[0]
    assert path == ("/v1/chat/completions" if api_format == "chat" else "/v1/responses")
    assert auth == "Bearer fixture-key"
    assert body["tools"] and "data:image/png;base64," in json.dumps(body)


def test_managed_vla_port_changes_do_not_change_agent_identity(tmp_path, monkeypatch):
    from roborsi.agents.config import AgentConfig
    from scripts.evaluation import vla
    from scripts.evaluation.configuration import ROOT, resolve_model

    ports = iter(["127.0.0.1:21000", "127.0.0.1:21001"])
    monkeypatch.setattr(
        vla,
        "_run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 0, next(ports) if command[1] == "port" else "", ""
        ),
    )
    monkeypatch.setattr(vla, "_wait", lambda *a: None)
    deployment = vla.Deployment(
        resolve_model(ROOT / "configs/models/default.yaml").vla, "image", tmp_path
    )
    source = ROOT / "configs/agents/libero/pro_vla.yaml"
    fingerprints, urls = [], []
    for _ in range(2):
        with vla.services(
            {"libero_pro": deployment}, {"libero_pro": source}, tmp_path
        ) as bound:
            config = AgentConfig.from_file(bound["libero_pro"])
            fingerprints.append(config.fingerprint)
            urls.append(os.environ[config.parameters["vla_endpoint_env"]])
    assert urls[0] != urls[1]
    assert fingerprints[0] == fingerprints[1]
    assert "ROBORSI_LIBERO_PRO_VLA_URL" not in os.environ


def test_http_catalogs_and_children_use_only_their_own_tokens(
    runner, registry, tmp_path, monkeypatch, capsys
):
    tokens = {"libero": "private-libero-token", "robocasa365": "private-robocasa-token"}
    calls = []

    def execute(command, **kwargs):
        name = command[command.index("--benchmark") + 1]
        variable = f"{name.upper()}_TOKEN"
        assert kwargs["env"][variable] == tokens[name]
        assert not any(
            f"{other.upper()}_TOKEN" in kwargs["env"]
            for other in tokens
            if other != name
        )
        calls.append(name)
        return subprocess.CompletedProcess(
            command, 0, json.dumps(evaluation_report(registry, name, "success")), ""
        )

    monkeypatch.setattr(runner, "run_child", execute)
    args = [*arguments(registry, tmp_path)[2:], "--api-key", "EMPTY"]
    with ExitStack() as stack:
        for name, token in tokens.items():
            monkeypatch.setenv(f"{name.upper()}_TOKEN", token)
            endpoint = stack.enter_context(
                catalog_server(registry, name, tmp_path, token=token)
            )
            config = tmp_path / f"{name}.yaml"
            config.write_text(
                f"schema_version: 1\nbenchmark: {name}\nendpoint: {endpoint}\ntoken_env: {name.upper()}_TOKEN\n"
            )
            args += ["--benchmark", name, "--benchmark-config", f"{name}={config}"]
        assert runner.main(args) == 0
    assert set(calls) == set(tokens)
    snapshot = (tmp_path / "results/trial/resolved_config.json").read_text()
    output = capsys.readouterr()
    assert all(
        token not in snapshot + output.out + output.err for token in tokens.values()
    )
