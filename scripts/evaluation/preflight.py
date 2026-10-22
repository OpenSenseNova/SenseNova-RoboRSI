# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small API probes before expensive downloads, model loading or episodes."""

import json
import os
import urllib.error
import urllib.request

from roborsi.agents.config import AgentConfig

IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aP1sAAAAASUVORK5CYII="


def model_format(args, name):
    configured = args.models[name].openai_format
    if configured:
        return configured
    config = args.configurations[name]
    override = (
        config.agent_overrides.get("runtime", {}).get("llm", {}).get("openai_format")
    )
    if override:
        return override
    llm = AgentConfig.from_file(config.agent).runtime().llm
    return llm.openai_format if llm and llm.openai_format else "responses"


def probe_model(model, api_format, *, timeout_s=15):
    tool = {
        "name": "roborsi_connectivity_check",
        "description": "Connectivity check only.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }
    if api_format == "chat":
        path = "/chat/completions"
        body = {
            "model": model.model_name,
            "max_tokens": 16,
            "tool_choice": "none",
            "tools": [{"type": "function", "function": tool}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "API connectivity check. Reply OK."},
                        {"type": "image_url", "image_url": {"url": IMAGE}},
                    ],
                }
            ],
        }
        field = "choices"
    else:
        path = "/responses"
        body = {
            "model": model.model_name,
            "max_output_tokens": 16,
            "tool_choice": "none",
            "tools": [{"type": "function", **tool}],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "API connectivity check. Reply OK.",
                        },
                        {"type": "input_image", "image_url": IMAGE},
                    ],
                }
            ],
        }
        field = "output"
    request = urllib.request.Request(
        model.base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model.api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read(1024 * 1024))
        if not isinstance(result, dict) or not isinstance(result.get(field), list):
            raise ValueError("unexpected response format")
    except urllib.error.HTTPError as exc:
        hint = (
            "check api_key_env/authentication"
            if exc.code in {401, 403}
            else "check base_url, model_name, openai_format and image/tool support"
        )
        raise RuntimeError(
            f"Main model API check failed (HTTP {exc.code}); {hint}. No environments downloaded."
        ) from None
    except (OSError, ValueError):
        raise RuntimeError(
            "Main model API check failed; check connectivity, base_url and openai_format. No environments downloaded."
        ) from None


def check(args):
    from roborsi.evaluation.http_benchmark import HttpBenchmarkRegistry
    from scripts.evaluation import vla

    presets = {name: str(config.agent) for name, config in args.configurations.items()}
    vla_plan = vla.plan(
        presets,
        models=args.models,
        parameter_overrides={
            name: config.agent_overrides.get("parameters", {})
            for name, config in args.configurations.items()
        },
    )
    seen = set()
    for name, model in args.models.items():
        fmt = model_format(args, name)
        identity = (model.base_url, model.model_name, model.api_key, fmt)
        if identity not in seen:
            print(
                f"Checking main model API: {name} ({fmt}, image/tool schema, 16-token probe)",
                flush=True,
            )
            probe_model(model, fmt)
            seen.add(identity)
    for name, url in args.endpoints.items():
        print(f"Checking Benchmark HTTP catalog: {name}", flush=True)
        variable = args.benchmark_tokens.get(name)
        try:
            HttpBenchmarkRegistry(
                {name: url},
                token=os.environ.get(variable) if variable else None,
                timeout_s=10,
            )
        except Exception:
            raise RuntimeError(
                f"{name}: Benchmark HTTP check failed; check endpoint, token_env and the service protocol."
            ) from None
    for name, profile in vla_plan.items():
        if profile["mode"] == "external":
            try:
                with urllib.request.urlopen(
                    profile["endpoint"] + "/v1/health", timeout=10
                ) as response:
                    status = json.loads(response.read(4096))
                if (
                    status.get("protocol") != vla.PROTOCOL
                    or status.get("policy_config") != profile["policy_config"]
                    or status.get("ready") is not True
                ):
                    raise ValueError("VLA health mismatch")
            except (OSError, ValueError, AttributeError):
                raise RuntimeError(
                    f"{name}: VLA endpoint is unavailable or incompatible; check model.vla.endpoint."
                ) from None
