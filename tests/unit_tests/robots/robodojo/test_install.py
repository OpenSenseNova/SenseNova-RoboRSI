# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Installation options reach the original independent model installer."""


def test_simate_custom_checkpoint_is_used_for_install_and_verification(
    tmp_path, monkeypatch
):
    from benchmarks.robodojo import install_agent
    from robots.robodojo.agents.submission import simate_runtime

    calls = []
    verified = []
    monkeypatch.setattr(
        install_agent.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    monkeypatch.setattr(
        simate_runtime,
        "check_simate_runtime",
        lambda policy, checkpoint: verified.append(checkpoint) or {"ok": True},
    )
    checkpoint = tmp_path / "weights"
    result = install_agent.install_simate(checkpoint)
    assert calls[0][1]["env"]["ROBORSI_SIM_CHECKPOINT"] == str(checkpoint.resolve())
    assert verified == [checkpoint.resolve()]
    assert result == {"ok": True}
