# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest

from robots.libero import three_benchmark_serve


def test_serve_freezes_manifest_visible_environment(monkeypatch):
    received = []
    environment = {}

    def fake_serve(arguments):
        received.extend(arguments)

    import roborsi.evaluation.service_host

    monkeypatch.setattr(roborsi.evaluation.service_host, "main", fake_serve)
    monkeypatch.setattr(three_benchmark_serve.os, "environ", environment)
    three_benchmark_serve.main(
        [
            "--benchmark",
            "libero_ultra_long",
            "--profile",
            "migration_subset",
            "--plus9-on-failure",
        ]
    )
    assert environment["ROBORSI_LIBERO_BENCHMARK"] == "libero_ultra_long"
    assert environment["ROBORSI_LIBERO_PROFILE"] == "migration_subset"
    assert environment["ROBORSI_LIBERO_PLUS9"] == "1"
    assert received[-2:] == ["--port", "8000"]


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_image_metadata_requires_explicit_deployment_flag(monkeypatch, enabled):
    import roborsi.evaluation.service_host

    environment = {
        "HEART_LIBERO_ASSETS_ROOT": "/legacy/assets",
        "HEART_LIBERO_INIT_ROOT": "/legacy/init",
        "ROBORSI_LIBERO_INIT_ROOT": "/explicit/init",
    }
    monkeypatch.setattr(three_benchmark_serve.os, "environ", environment)
    monkeypatch.setattr(roborsi.evaluation.service_host, "main", lambda _: None)
    arguments = ["--benchmark", "libero_pro", "--profile", "official"]
    if enabled:
        arguments.append("--legacy-image-env")
    three_benchmark_serve.main(arguments)
    assert environment.get("ROBORSI_LIBERO_ASSETS_ROOT") == (
        "/legacy/assets" if enabled else None
    )
    assert environment["ROBORSI_LIBERO_INIT_ROOT"] == "/explicit/init"


@pytest.mark.parametrize(
    "option", [["--image-observation-tools"], ["--eval-config", "protocol.yaml"]]
)
def test_extensions_require_explicit_bundle(monkeypatch, option):
    import roborsi.evaluation.service_host

    monkeypatch.setattr(three_benchmark_serve.os, "environ", {})
    monkeypatch.setattr(roborsi.evaluation.service_host, "main", lambda _: None)
    args = ["--benchmark", "libero_pro", "--profile", "official", *option]
    with pytest.raises(SystemExit):
        three_benchmark_serve.main(args)
    three_benchmark_serve.main([*args, "--source-bundle", "libero_pro_image_tools"])
    assert (
        three_benchmark_serve.os.environ["ROBORSI_LIBERO_SOURCE_BUNDLE"]
        == "libero_pro_image_tools"
    )


def test_pro_bundle_rejected_for_long_benchmark(monkeypatch):
    monkeypatch.setattr(three_benchmark_serve.os, "environ", {})
    with pytest.raises(SystemExit):
        three_benchmark_serve.main(
            [
                "--benchmark",
                "libero_long_plus_plus",
                "--profile",
                "official",
                "--source-bundle",
                "libero_pro_image_tools",
            ]
        )
