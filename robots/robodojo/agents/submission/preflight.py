"""Check the pinned CLI and provider configuration without calling a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

POLICY_DIR = Path(__file__).resolve().parent



def relative_artifact_path(value: Any, field: str) -> Path:
    """Keep installation artifacts within the policy directory."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"submission.json: {field} is required")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(
            f"submission.json: {field} must stay within the policy directory"
        )
    return relative


def sha256_file(path: Path) -> str:
    """Return a file digest without loading the artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_submission(policy_dir: Path = POLICY_DIR) -> dict[str, Any]:
    """Validate nonsecret submission metadata using only the standard library."""
    path = policy_dir / "submission.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("submission.json: unsupported schema_version")
    validate_adaptive_profile(metadata, policy_dir)
    if not metadata.get("agent_label"):
        raise ValueError("submission.json: agent_label is required")
    cli = metadata.get("cli", {})
    if not isinstance(cli, dict):
        raise ValueError("submission.json: cli must be a mapping")
    if cli.get("distribution") != "standalone":
        raise ValueError(
            "submission.json: the confirmed CLI distribution is standalone"
        )
    version = cli.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(
            "submission.json: an exact CLI version is required; never use latest"
        )
    if cli.get("expected_version") != f"codex-cli {version}":
        raise ValueError(
            "submission.json: expected_version must match the CLI --version output"
        )
    if cli.get("target") != "x86_64-unknown-linux-musl":
        raise ValueError("submission.json: unsupported confirmed standalone target")
    if not re.fullmatch(r"[0-9a-f]{64}", str(cli.get("sha256", ""))):
        raise ValueError(
            "submission.json: the confirmed CLI executable SHA-256 is required"
        )
    companions = cli.get("companion_sha256")
    if not isinstance(companions, dict) or set(companions) != {
        "bin/codex-code-mode-host",
        "codex-resources/bwrap",
    }:
        raise ValueError(
            "submission.json: confirmed Codex companion identities are required"
        )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in companions.values()
    ):
        raise ValueError(
            "submission.json: companion identities must be SHA-256 digests"
        )
    return metadata


def validate_adaptive_profile(metadata: dict[str, Any], policy_dir: Path) -> None:
    """Verify the separately pinned adaptive snapshot, including public interfaces."""
    hybrid = metadata.get("hybrid", {})
    if hybrid.get("variant") != "adaptive_2_50":
        raise ValueError("The submission requires its explicit adaptive hybrid profile")
    root = policy_dir / relative_artifact_path(
        hybrid.get("profile_path"), "hybrid.profile_path"
    )
    manifest_path = policy_dir / relative_artifact_path(
        hybrid.get("manifest_path"), "hybrid.manifest_path"
    )
    expected = hybrid.get("manifest_sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(expected))
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != expected
    ):
        raise ValueError("The adaptive source provenance is missing or changed")
    document = json.loads(manifest_path.read_text())
    if (
        document.get("schema_version") != 1
        or document.get("policy_variant") != hybrid["variant"]
    ):
        raise ValueError("Adaptive snapshot has a different baseline or policy")
    names = set()
    for row in document["files"]:
        relative = relative_artifact_path(row["path"], "adaptive source file")
        name = relative.as_posix()
        path = root / relative
        if (
            name in names
            or not path.resolve().is_relative_to(root.resolve())
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError("Adaptive profile is missing or changed: " + name)
        names.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Adaptive source cannot contain symlinks")
        if path.is_file() and not (
            "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
        ):
            actual.add(path.relative_to(root).as_posix())
    if actual != names | {manifest_path.relative_to(root).as_posix()}:
        raise ValueError("Adaptive snapshot contains unexpected files")


def validate_archive(metadata: dict[str, Any], policy_dir: Path = POLICY_DIR) -> Path:
    """Require an authenticated installation artifact or a pinned download URL."""
    cli = metadata["cli"]
    relative = relative_artifact_path(cli.get("archive_path"), "cli.archive_path")
    archive = policy_dir / relative
    digest = cli.get("archive_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(
            "The fixed Codex native release is declared, but its install archive "
            "SHA-256 is not recorded. Supply the verified release archive and "
            "cli.archive_sha256; a Python SDK or npm version is not a substitute."
        )
    if archive.is_file():
        if sha256_file(archive) != digest:
            raise ValueError("Codex standalone archive SHA-256 mismatch")
        return archive
    download = cli.get("download", {})
    if not isinstance(download, dict):
        raise ValueError("submission.json: cli.download must be a mapping")
    url = download.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(
            f"Codex installation archive is missing: {archive}. Record its confirmed "
            "HTTPS distribution URL and digest in cli.download or supply that archive."
        )
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "submission.json: cli.download.url must be HTTPS without credentials"
        )
    digest = download.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(
            "submission.json: cli.download.sha256 is required for the official archive"
        )
    relative = relative_artifact_path(download.get("path"), "cli.download.path")
    archive = policy_dir / relative
    if archive.is_file() and sha256_file(archive) != digest:
        raise ValueError("Cached official Codex archive SHA-256 mismatch")
    return archive


def validate_companions(metadata: dict[str, Any], codex_root: Path) -> None:
    """Check the host and sandbox shipped with the confirmed standalone CLI."""
    manifest_path = codex_root / "codex-package.json"
    if not manifest_path.is_file():
        raise ValueError("Codex standalone package manifest is missing")
    package = json.loads(manifest_path.read_text())
    if any(
        package.get(key) != value
        for key, value in {
            "version": metadata["cli"]["version"],
            "target": metadata["cli"]["target"],
            "entrypoint": "bin/codex",
        }.items()
    ):
        raise ValueError("Codex standalone package version/target/entrypoint mismatch")
    for relative, expected in metadata["cli"]["companion_sha256"].items():
        companion = codex_root / relative
        if not companion.is_file() or sha256_file(companion) != expected:
            raise ValueError(f"Codex companion identity mismatch: {relative}")


def validate_skillpack(policy_dir: Path = POLICY_DIR, task_name: str | None = None) -> dict[str, Any]:
    """Verify shared skills and the public task-name catalog."""
    metadata = json.loads((policy_dir/"submission.json").read_text())
    pins = metadata["skillpack"]
    for key in ("catalog", "frozen_common"):
        path = policy_dir/relative_artifact_path(pins[key+"_path"], key)
        if path.is_symlink() or sha256_file(path) != pins[key+"_sha256"]:
            raise ValueError("Shared skillpack manifest differs: "+key)
    root = policy_dir/"skillpack"
    frozen = json.loads((root/"frozen_common.json").read_text())
    try:
        from .runtime_config import PI05_BASE_PROMPT_SHA256, load_pi05_base_prompt
    except ImportError:
        from runtime_config import PI05_BASE_PROMPT_SHA256, load_pi05_base_prompt
    if frozen.get("common/pi05_base_prompt.txt") != PI05_BASE_PROMPT_SHA256:
        raise ValueError("Base prompt must be registered in the shared manifest")
    load_pi05_base_prompt(policy_dir)
    for name, expected in frozen.items():
        path = root/relative_artifact_path(name, "skillpack resource")
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError("Shared skillpack resource differs: "+name)
    tasks = json.loads((root/"task_catalog.json").read_text())["tasks"]
    if task_name is not None and task_name not in tasks:
        raise ValueError("Unknown public RoboDojo task: "+task_name)
    return {"shared_files_verified": len(frozen), "task_count": len(tasks),
            "exposed_skills": ["robo-general", "manipulation"]}


def check_vla_runtime(model_cfg: dict) -> dict:
    """Resolve the selected backend; disabled mode imports no VLA runtime."""
    try:
        from .runtime_config import resolve_vla_config
    except ImportError:
        from runtime_config import resolve_vla_config
    selected = resolve_vla_config(model_cfg)
    if selected["backend"] == "disabled":
        return {"enabled": False, "backend": "disabled"}
    if os.environ.get("ROBORSI_VLA_POOL_URL"):
        if selected["backend"] != "pi05":raise ValueError("Pi pool requires pi05 backend")
        try:
            from .vla_pool import PoolClient
        except ImportError:
            import importlib
            PoolClient=importlib.import_module("XPolicyLab.policy.SenseNova-RoboRSI.vla_pool").PoolClient
        client=PoolClient(POLICY_DIR)
        return {"backend":"pi05","pool_endpoint":client.endpoint,"pool_id":client.metadata["pool_id"],"identity":client.metadata["roborsi_vla_identity"],"checkpoint_files_verified_by_service_owner":True}
    try:
        from .pi05_source import load_openpi_client
    except ImportError:
        from pi05_source import load_openpi_client
    try:
        from .checkpoint_resolver import resolve_checkpoint_root
    except ImportError:
        from checkpoint_resolver import resolve_checkpoint_root

    load_openpi_client(POLICY_DIR)
    effective = {**model_cfg, "checkpoint_path": selected["checkpoint_path"]}
    checkpoint = resolve_checkpoint_root(effective, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
    if selected["backend"] == "simate_beta":
        try:
            from .simate_runtime import check_simate_runtime
        except ImportError:
            from simate_runtime import check_simate_runtime
        return check_simate_runtime(POLICY_DIR, checkpoint)
    shared_port = os.environ.get("ROBORSI_SHARED_VLA_PORT")
    if shared_port:
        # The one model owner verifies all weights and runtime distributions
        # before publishing this receipt. Clients authenticate that live service
        # instead of each rereading the same multi-GB checkpoint over AFS.
        try:
            from .vla_identity import check_shared_vla
        except ImportError:
            from vla_identity import check_shared_vla
        port = int(shared_port)
        if not 1 <= port <= 65535:
            raise ValueError("Shared Pi0.5 port is outside the TCP port range")
        receipt_name = os.environ.get("ROBORSI_SHARED_VLA_RECEIPT")
        receipt_path = Path(receipt_name) if receipt_name else None
        check_shared_vla(port, backend="pi05", receipt_path=receipt_path, policy_dir=POLICY_DIR)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("checkpoint_dir") != str(checkpoint.resolve()):
            raise ValueError("Shared Pi0.5 checkpoint differs from deployment")
        return {"backend": "pi05", "shared_service_verified": True,
                "port": port, "identity": receipt["identity"],
                "checkpoint": receipt["checkpoint_dir"],
                "checkpoint_files_verified_by_service_owner": True}
    try:
        from .pi05_runtime import check_pi05_runtime
    except ImportError:
        from pi05_runtime import check_pi05_runtime
    return check_pi05_runtime(POLICY_DIR, checkpoint)


def check_runtime(
    config_path: Path,
    *,
    require_key: bool,
    installation_only: bool = False,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Use the same AgentConfig and executable preflight as the production Model."""
    import yaml
    from roborsi.planner.codex_cli import preflight_codex_cli
    from roborsi.planner.codex_cli_config import CodexCliConfig

    try:
        from .runtime_config import load_agent_config
    except ImportError:
        from runtime_config import load_agent_config

    model_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(model_cfg, dict):
        raise ValueError("deploy config must contain a mapping")
    agent_config = load_agent_config(
        model_cfg, require_key=require_key, installation_only=installation_only
    )
    cli = CodexCliConfig.from_dict(agent_config.document["agent"]["codex_cli"])
    environment = None
    if not require_key:
        # Keep CODEX_API_KEY in the production config while allowing an offline
        # version check. Never mutate os.environ or synthesize credentials.
        environment = dict(os.environ) if cli.inherit_env else {}
        environment.pop("CODEX_API_KEY", None)
        for name in cli.env_vars:
            if name == "CODEX_API_KEY":
                continue
            if name not in os.environ:
                raise ValueError(
                    f"required Codex environment variable is missing: {name}"
                )
            environment[name] = os.environ[name]
        environment.update(cli.env)
    identity = preflight_codex_cli(cli, repo_root=POLICY_DIR, env=environment)
    metadata = load_submission()
    try:
        from .custom_cli import selected_cli
    except ImportError:
        from custom_cli import selected_cli
    selected = selected_cli(POLICY_DIR, metadata["cli"])
    if identity["sha256"] != selected["sha256"]:
        raise ValueError(
            "Installed CLI SHA-256 does not match the confirmed standalone release"
        )
    validate_companions({"cli": selected}, POLICY_DIR / "runtime/codex")
    if selected["selection"] == "operator_custom_build":
        validate_companions({"cli": selected}, Path(selected["executable"]).parent.parent)
    identity["selection"] = selected["selection"]
    if "manifest" in selected:
        identity["selection_manifest"] = selected["manifest"]
    try:
        from .runtime_capabilities import (
            check_python_capabilities,
            check_sandbox_capabilities,
        )
    except ImportError:
        from runtime_capabilities import (
            check_python_capabilities,
            check_sandbox_capabilities,
        )

    capabilities = {
        "python": check_python_capabilities(),
        "sandbox": check_sandbox_capabilities(POLICY_DIR),
    }
    selected_task = task_name or model_cfg.get("task_name")
    skillpack = validate_skillpack(POLICY_DIR, selected_task)
    try:
        from .agent_track import validate_mode
        from .runtime_config import resolve_eef_execution_backend
    except ImportError:
        from agent_track import validate_mode
        from runtime_config import resolve_eef_execution_backend
    validate_mode(model_cfg.get("skill_library_mode", "general_only"))
    resolve_eef_execution_backend(model_cfg, selected_task or "")
    vla = check_vla_runtime(model_cfg)
    provider = cli.config_overrides
    # Deliberately print only an allowlist; never serialize AgentConfig or the environment.
    return {

        "agent_label": metadata["agent_label"],
        "model": None if installation_only else agent_config.document["agent"]["model"],
        "base_url": None if installation_only else provider["model_providers.runtime.base_url"],
        "deployment_configured": not installation_only,
        "wire_api": provider["model_providers.runtime.wire_api"],
        "executable": identity["executable"],
        "version": identity["version"],
        "sha256": identity["sha256"],
        "source_files": "verified against the authenticated source artifact manifest",
        "check": ("offline installation checks; deployment settings and model access were not tested"
                  if installation_only else "offline CLI identity and configuration; model access was not tested"),
        "key_required": require_key,
        "runtime_capabilities": capabilities,
        "skillpack": skillpack,
        **({"pi05": vla} if model_cfg.get("vla_backend") not in {"simate_beta", "disabled"} else {"vla": vla}),
    }


def main() -> int:
    """Run an offline preflight and report actionable errors without traceback noise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=POLICY_DIR / "deploy.yml")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="check source/CLI installation metadata without importing policy dependencies",
    )
    parser.add_argument(
        "--allow-missing-key",
        action="store_true",
        help="permit only this offline configuration/version check without CODEX_API_KEY",
    )
    parser.add_argument(
        "--installation-only", action="store_true",
        help="validate installed artifacts and CLI offline without deployment settings or credentials",
    )
    parser.add_argument(
        "--task-name",
        help="check the public task identity",
    )
    args = parser.parse_args()
    try:
        metadata = load_submission()
        skillpack = validate_skillpack(POLICY_DIR, args.task_name)
        try:
            from .pi05_assets import load_pi05_metadata
            from .pi05_source import validate_pi05_source
            from .pi05_tokenizer import load_pi05_tokenizer_metadata
        except ImportError:
            from pi05_assets import load_pi05_metadata
            from pi05_source import validate_pi05_source
            from pi05_tokenizer import load_pi05_tokenizer_metadata

        load_pi05_metadata(POLICY_DIR)
        validate_pi05_source(POLICY_DIR)
        load_pi05_tokenizer_metadata(POLICY_DIR)
        try:
            from .tool_runtime import load_tool_metadata
        except ImportError:
            from tool_runtime import load_tool_metadata

        tools = load_tool_metadata(POLICY_DIR)
        validate_archive(metadata)
        if args.metadata_only:
            result = {

                "agent_label": metadata["agent_label"],
                "version": metadata["cli"]["expected_version"],
                "check": "installation metadata only; CLI/model were not executed",
                "tool_python": tools["python_version"],
                "skillpack": skillpack,
            }
        else:
            result = check_runtime(
                args.config.resolve(),
                require_key=not (args.allow_missing_key or args.installation_only),
                installation_only=args.installation_only,
                task_name=args.task_name,
            )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"RoboRSI preflight failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
