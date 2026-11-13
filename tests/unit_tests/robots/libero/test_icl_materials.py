# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Original stable alias contract for frozen LIBERO ICL inputs."""

from types import SimpleNamespace

from scripts.libero_three.prepare_context import _destination_name


def test_pro_context_export_uses_a_public_stable_alias() -> None:
    package = SimpleNamespace(package_id="private-source-name")

    assert (
        _destination_name("libero_pro", "libero_spatial_swap-t07-i03", package)
        == "libero-pro-context-spatial-t07"
    )
    assert _destination_name("libero_ultra_long", "t00-i01", package) == (
        "private-source-name"
    )
