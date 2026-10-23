# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Freeze three-LIBERO provider settings in a manifest-visible command."""

from __future__ import annotations

import argparse
import os


def _import_legacy_image_environment() -> None:
    """Translate pinned legacy-image metadata only for an explicit deployment."""
    fields = (
        "LIBERO_ASSETS_ROOT",
        "LIBERO_BENCHMARK_ROOT",
        "LIBERO_DATASETS_ROOT",
        "LIBERO_BDDL_ROOT",
        "LIBERO_INIT_ROOT",
        "LIBERO_PRO_COMMIT",
        "LIBERO_VERSION",
        "LILO_COMMIT",
        "LILO_ROOT",
        "LIBERO_EVALUATION_ASSET_SHA256",
    )
    for field in fields:
        old, new = f"HEART_{field}", f"ROBORSI_{field}"
        if old in os.environ and new not in os.environ:
            os.environ[new] = os.environ[old]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("libero_pro", "libero_long_plus_plus", "libero_ultra_long"),
        required=True,
    )
    parser.add_argument(
        "--profile", choices=("official", "migration_subset"), required=True
    )
    parser.add_argument(
        "--framework", choices=("no_vla_no_sam3", "vla_sam3"), default="no_vla_no_sam3"
    )
    parser.add_argument(
        "--source-bundle",
        choices=("libero_pro", "libero_pro_image_tools", "libero_lilo"),
        help="explicit service source bundle (defaults to the benchmark's original bundle)",
    )
    parser.add_argument("--image-observation-tools", action="store_true")
    parser.add_argument("--eval-config", help="explicit PRO YAML or JSON config")
    parser.add_argument(
        "--legacy-image-env",
        action="store_true",
        help="translate HEART_* metadata for an explicitly selected legacy image",
    )
    parser.add_argument("--plus9-on-failure", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    default_bundle = "libero_pro" if args.benchmark == "libero_pro" else "libero_lilo"
    bundle = args.source_bundle or os.environ.get(
        "ROBORSI_LIBERO_SOURCE_BUNDLE", default_bundle
    )
    allowed = {default_bundle}
    if args.benchmark == "libero_pro":
        allowed.add("libero_pro_image_tools")
    if bundle not in allowed:
        parser.error("--source-bundle does not match --benchmark")
    if (
        args.eval_config or args.image_observation_tools or args.framework == "vla_sam3"
    ) and bundle != "libero_pro_image_tools":
        parser.error(
            "custom protocol/tools require --source-bundle libero_pro_image_tools"
        )
    os.environ["ROBORSI_LIBERO_SOURCE_BUNDLE"] = bundle
    if args.image_observation_tools and args.benchmark != "libero_pro":
        parser.error("--image-observation-tools currently requires PRO")
    os.environ["ROBORSI_IMAGE_OBSERVATION_TOOLS"] = (
        "1" if args.image_observation_tools else "0"
    )
    if args.legacy_image_env:
        _import_legacy_image_environment()
    if args.framework == "vla_sam3" and args.benchmark != "libero_pro":
        parser.error("VLA/SAM3 framework currently requires libero_pro")
    os.environ["ROBORSI_LIBERO_TOOL_MODE"] = args.framework
    if args.eval_config:
        if args.benchmark != "libero_pro" or args.plus9_on_failure:
            parser.error("--eval-config requires PRO without extra-budget recovery")
        os.environ["ROBORSI_PRO_EVAL_CONFIG"] = args.eval_config
    else:
        os.environ.pop("ROBORSI_PRO_EVAL_CONFIG", None)
    os.environ["ROBORSI_LIBERO_BENCHMARK"] = args.benchmark
    os.environ["ROBORSI_LIBERO_PROFILE"] = args.profile
    os.environ["ROBORSI_LIBERO_PLUS9"] = "1" if args.plus9_on_failure else "0"
    from roborsi.evaluation.service_host import main as serve

    serve(
        [
            "--kind",
            "benchmark",
            "--factory",
            "robots.libero.three_benchmark_service:make_three_libero_benchmark",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )


if __name__ == "__main__":
    main()
