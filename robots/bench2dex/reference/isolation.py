# Copyright 2026 The Bench2Dex Evaluation Contributors.
# Licensing of this copied reference code is pending; see LICENSE_NOTICE.md.

"""A minimal mount namespace around Codex, in addition to its tool sandbox."""

import shutil
import subprocess
import sysconfig
from pathlib import Path

from .core import MODEL

DISABLED = (
    "multi_agent",
    "multi_agent_v2",
    "apps",
    "plugins",
    "remote_plugin",
    "hooks",
    "memories",
    "external_agent_memory_import",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "guardian_approval",
    "guardian_ext",
    "guardianv2",
    "workspace_dependencies",
    "skill_search",
    "skill_mcp_dependency_install",
    "shell_snapshot",
    "goals",
)


def mounts(
    workspace: Path,
    codex_home: Path,
    interface: Path,
    binary: Path,
    numerical_runtime: bool = False,
    input_source: Path | None = None,
    prior_inputs: Path | None = None,
) -> list[str]:
    if not shutil.which("bwrap"):
        raise RuntimeError(
            "bubblewrap is required; refusing host-visible policy execution"
        )
    command = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/home/policy",
        "--dir",
        "/etc",
        "--bind",
        str(workspace),
        "/workspace",
        "--ro-bind",
        str(input_source or workspace / "input"),
        "/workspace/input",
        "--bind",
        str(codex_home),
        "/codex-home",
        "--ro-bind",
        str(interface),
        "/interface",
        "--ro-bind",
        str(binary.resolve()),
        "/opt/codex",
        "--setenv",
        "HOME",
        "/home/policy",
        "--setenv",
        "CODEX_HOME",
        "/codex-home",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
        "--setenv",
        "TZ",
        "UTC",
        "--chdir",
        "/workspace",
    ]
    if prior_inputs is not None:
        prior_inputs = Path(prior_inputs)
        if (
            prior_inputs.name != "input"
            or prior_inputs.parent.name != "policy"
            or not prior_inputs.is_dir()
        ):
            raise ValueError(
                "Continuation can mount only its prior public policy input"
            )
        if prior_inputs.resolve(strict=True) != prior_inputs:
            raise ValueError("Prior policy input may not be a symlink")
        command.extend(["--ro-bind", str(prior_inputs), "/prior_input"])
    transport = interface.parent / "action-transport"
    if transport.is_dir():
        # A mount cannot be replaced by renaming model scratch directories.
        # Expose only transport, never its private parent or evaluator data.
        command.extend(["--bind", str(transport), "/workspace/.action_transport"])
    # Preparation-only mailbox: requests are writable; replies/images are not.
    # Neither mount contains raw HDF5 files, server code or private provenance.
    requests = interface.parent / "demo-reader-requests"
    responses = interface.parent / "demo-reader-public"
    if requests.is_dir() and responses.is_dir():
        command.extend(
            [
                "--bind",
                str(requests),
                "/workspace/.demo_requests",
                "--ro-bind",
                str(responses),
                "/workspace/.demo_responses",
            ]
        )
    if numerical_runtime:
        # Debian/Ubuntu BLAS alternatives point through /etc/alternatives,
        # outside this mount. Select only existing system library directories;
        # do not inherit a user's LD_LIBRARY_PATH or expose their home.
        triplet = sysconfig.get_config_var("MULTIARCH")
        paths = [Path("/usr/lib") / str(triplet) / name for name in ("blas", "lapack")]
        paths = [str(p) for p in paths if p.is_dir()]
        if paths:
            command.extend(["--setenv", "LD_LIBRARY_PATH", ":".join(paths)])
    for path in (
        "/etc/ssl/certs",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/ld.so.cache",
    ):
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])
    code_host = binary.resolve().with_name("codex-code-mode-host")
    if not code_host.is_file():
        raise RuntimeError("Pinned Codex installation requires codex-code-mode-host")
    command.extend(["--ro-bind", str(code_host), "/opt/codex-code-mode-host"])
    return command


def command(
    workspace,
    codex_home,
    interface,
    binary,
    config,
    *,
    preparation=False,
    input_source=None,
    resume_session=None,
    prior_inputs=None,
):
    result = mounts(
        workspace,
        codex_home,
        interface,
        binary,
        numerical_runtime=config.get("deployment_profile")
        == "calibrated_deployment_v2",
        input_source=input_source,
        prior_inputs=prior_inputs,
    )
    result += [
        "--",
        "/opt/codex",
        "exec",
        "--model",
        MODEL,
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--json",
        "--color",
        "never",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="live"' if preparation else 'web_search="disabled"',
        "-c",
        "sandbox_workspace_write.network_access=false",
        "-c",
        'model_reasoning_effort="' + config["reasoning_effort"] + '"',
        "-c",
        "allow_login_shell=false",
        "-c",
        "project_doc_max_bytes=0",
    ]
    for feature in DISABLED:
        result.extend(["--disable", feature])
    if resume_session is not None:
        import uuid

        if (
            preparation
            or str(uuid.UUID(resume_session)) != resume_session
            or prior_inputs is None
        ):
            raise ValueError(
                "Resume requires an explicit same-episode session and prior input"
            )
        result.extend(["resume", resume_session, "-"])
    else:
        result.append("-")
    return result


def prepare_auth(codex_home: Path, auth: Path):
    # Explicit single auth file only. Never copy user's config, sessions or plugins.
    codex_home.mkdir(parents=True, exist_ok=True)
    target = codex_home / "auth.json"
    shutil.copyfile(auth, target)
    target.chmod(0o600)


def probe(workspace, codex_home, interface, binary, private_path: Path):
    code = (
        "import os,socket; "
        "assert not os.path.exists(" + repr(str(private_path)) + "); "
        "assert not os.path.exists('/data'); "
        "assert os.path.exists('/workspace'); print('mount isolation passed')"
    )
    return subprocess.run(
        mounts(workspace, codex_home, interface, binary)
        + ["--", "/usr/bin/python3", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
