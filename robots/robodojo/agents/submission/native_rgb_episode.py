"""Run one configured native episode with lossless RGB IPC and JPEG-only archives.

This launcher creates no PNG observation files. Existing frozen experiments
continue to use their original launchers and capture identities.
"""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import time
import traceback
import uuid


ROOT = WORKSPACE = POLICY = None


def configure(argv=None):
    global ROOT, WORKSPACE, POLICY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Owned batch directory containing cases.json")
    parser.add_argument("--workspace", type=Path, required=True, help="Parent directory containing the frozen XPolicyLab checkout")
    parser.add_argument("--native-python", type=Path, required=True, help="Native simulator Python or its existing wrapper")
    parser.add_argument("--native-root", type=Path, required=True, help="Existing native RoboDojo installation")
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    ROOT, WORKSPACE = args.root.resolve(strict=True), args.workspace.resolve(strict=True)
    POLICY = WORKSPACE/"XPolicyLab/policy/SenseNova-RoboRSI"
    if not (ROOT/"cases.json").is_file() or not (POLICY/"native_rgb_adapter.py").is_file():
        parser.error("Missing cases.json or the frozen raw-RGB policy source")
    if not args.native_python.is_file() or not args.native_root.is_dir():
        parser.error("Native Python/root must already exist")
    sys.path[:0] = [str(WORKSPACE), str(POLICY/"runtime/RoboRSI")]
    os.environ["ROBORSI_ROBODOJO_NATIVE_ROOT"] = str(args.native_root.resolve())
    return args


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".new")
    temp.write_text(json.dumps(data, indent=2)+"\n")
    temp.replace(path)


def provider_credential(provider_kind, stream):
    """Only CCI is unauthenticated; native Messages keeps the paid credential."""
    if provider_kind == 'cci':
        return 'local-cci-noauth'
    if provider_kind not in ('responses', 'anthropic_messages'):
        raise ValueError('Unsupported provider kind')
    value = json.loads(stream.readline())['key']
    if not isinstance(value, str) or not value:
        raise ValueError('Provider credential missing')
    return value

def configure_vla(spec: dict) -> dict[str, str | None]:
    """Bind only an explicitly selected shared service; otherwise own the VLA."""
    try:
        from .runtime_config import resolve_vla_config
    except ImportError:
        from runtime_config import resolve_vla_config
    vla = resolve_vla_config(spec)
    if vla["backend"] == "disabled":
        os.environ.pop("ROBORSI_SHARED_VLA_PORT", None)
        os.environ.pop("ROBORSI_SHARED_VLA_RECEIPT", None)
        return {"vla_backend": "disabled", "checkpoint_path": None}
    port = spec.get("shared_vla_port")
    if port is not None and (type(port) is not int or not 1 <= port <= 65535):
        raise ValueError("shared_vla_port must be an integer in 1..65535")
    receipt = spec.get("shared_vla_receipt")
    if receipt is not None and (not isinstance(receipt, str) or not Path(receipt).is_absolute()):
        raise ValueError("shared_vla_receipt must be an absolute path")
    if port is None:
        os.environ.pop("ROBORSI_SHARED_VLA_PORT", None)
        os.environ.pop("ROBORSI_SHARED_VLA_RECEIPT", None)
    else:
        os.environ["ROBORSI_SHARED_VLA_PORT"] = str(port)
        if receipt is None:
            os.environ.pop("ROBORSI_SHARED_VLA_RECEIPT", None)
        else:
            os.environ["ROBORSI_SHARED_VLA_RECEIPT"] = receipt
    return {"vla_backend": vla["backend"], "checkpoint_path": vla["checkpoint_path"]}


def main():
    args = configure()
    label = args.label
    spec = json.loads((ROOT/"cases.json").read_text())[label]
    save_video=spec.get('save_video',False)
    if type(save_video) is not bool:raise ValueError('save_video must be boolean')
    route = spec["model"]
    task = spec["task"]
    step_limit = spec["native_step_limit"]
    # Only a single JSON stdin line carries the credential; never persist it.
    credential = provider_credential(spec["provider_kind"], sys.stdin)
    os.environ.update(CODEX_API_KEY=credential, OPENAI_API_KEY=credential,
                      ROBORSI_CODEX_MODEL=route, ROBORSI_CODEX_BASE_URL=spec["endpoint"],
                      ROBORSI_NATIVE_LAYOUT_ID="0")
    del credential
    case = ROOT/"cases"/label
    case.mkdir(parents=True, exist_ok=False)
    run_id = f"xpolicy-{label}-{task}-s0-layout0-"+uuid.uuid4().hex[:12]
    os.environ["ROBODOJO_RUN_ID"] = run_id
    os.environ["ROBORSI_RUNTIME_STATE_DIR"] = str(case/"state")
    model_module = importlib.import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")
    rgb_module = importlib.import_module("XPolicyLab.policy.SenseNova-RoboRSI.native_rgb")
    cfg = json.loads((POLICY/"vendor/robodojo_adaptive/resources/fresh-configs"/(task+"-s0.json")).read_text())
    source = POLICY/"vendor/robodojo_adaptive/source"
    adapter = source/"code/configurations/robodojo/20261005_standard_1x_10x/adapters/adapter.py"
    cfg.update(adapter=str(adapter)+":Adapter", benchmark_python=str(args.native_python.resolve()),
               environment_dir=os.environ["ROBORSI_ROBODOJO_NATIVE_ROOT"], request_seconds=180,
               episode_seconds=None, explore_seconds=0, training_set=str(case/"empty-training"))
    cfg["options"].update(action_mode="joint", review_window_steps=step_limit, review_max_windows=1,
                           stop_min_steps=step_limit, max_repeat=1, gpu=0, physical_gpu=0, reset_request_seconds=180,
                           save_video=save_video)
    native = case/"native"
    native.mkdir()
    (case/"empty-training").mkdir()
    observations = native/"observations"
    observations.mkdir()
    records = native/"episode"
    records.mkdir()
    model_cfg = dict(bench_name="RoboDojo", task_name=task, env_cfg_type="arx_x5",
                     action_type="joint", eval_batch=False, seed=0, **configure_vla(spec),
                     ckpt_name="roborsi-codex-v1", trajectory_dir=str(case/"trajectories"),
                     provider_kind=spec["provider_kind"], reasoning_effort=spec["reasoning_effort"], episode_seconds=None)
    if "spatial_localization" in spec:model_cfg["spatial_localization"] = spec["spatial_localization"]
    if "sampling" in spec:model_cfg["sampling"] = spec["sampling"]
    if "agent_exit_policy" in spec:model_cfg["agent_exit_policy"] = spec["agent_exit_policy"]
    model_cfg['storage_mode']=spec.get('storage_mode','training')
    model_cfg['save_video']=save_video
    if "enable_thinking" in spec:model_cfg["enable_thinking"] = spec["enable_thinking"]
    if "eef_execution_backend" in spec:model_cfg["eef_execution_backend"] = spec["eef_execution_backend"]
    prior=ROOT/"prior_usage_files.json"
    if prior.exists():model_cfg["prior_usage_files"]=json.loads(prior.read_text())
    model_cfg["episode_seconds"] = cfg["episode_seconds"] = None
    model_cfg["native_step_limit"] = step_limit
    write(case/"status.json", {"phase": "preflight", "at": time.time(), "model": route,
          "policy_source": str(POLICY), "native_step_limit": step_limit, "score": None,
          "native_rgb_transport": rgb_module.TRANSPORT, "archive_image_format": "JPEG95_444"})
    backend = model = None
    prune_native = False
    started = time.time()
    control_steps = 0
    try:
        model = model_module.Model(model_cfg)
        model.reset()
        backend = rgb_module.create_backend(cfg, native)
        backend.request(op="initialize")
        episode = cfg["test_episodes"][0]
        reply = backend.request(op="reset", episode=episode, record_dir=str(records), observation_dir=str(observations))
        native_identity = json.loads((records/"native_episode.json").read_text())
        scene_sha = native_identity["layout_sha256"]
        expected_sha = "8d45b23010c950c35d5618c8701c6b8203036abaaf20a228db067ec3b1e30617"
        if task == "arrange_largest_number" and scene_sha != expected_sha:
            raise ValueError("Native layout differs from the authorized seed0/layout0")
        write(case/"native-identity.json", native_identity)

        def current_observation(result):
            obs = result["observation"]
            value = backend.model_observation(result)
            value["roborsi_native_identity"] = {"schema": 1, "task": task,
                "eval_seed": 0, "layout": 0, "run_id": run_id, "scene_sha256": scene_sha,
                "control_step": obs["control_step"], "step_limit": step_limit}
            return value

        model.update_obs(current_observation(reply))
        write(case/"status.json", {"phase": "agent_running", "model": route, "native_steps": 0, "at": time.time(), "score": None})
        while not reply["done"]:
            actions = model.get_action()
            if actions == []:
                continue
            if len(actions) != 1:
                raise ValueError("Expected exactly one native action per Model chunk")
            action = actions[0]
            if "_roborsi_control" in action:
                if action == {"_roborsi_control": "finish"}:
                    reply = backend.request(op="finish")
                    break
                if action.get("_roborsi_control") != "bounded_curobo":
                    raise ValueError("Unknown native finish envelope")
            command = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in action.items()}
            reply = backend.request(op="step", action=command)
            control_steps += 1
            if reply["observation"]["control_step"] != control_steps or control_steps > step_limit:
                raise ValueError("Native step ACK/budget mismatch")
            with (case/"joint_actions.jsonl").open("a") as stream:
                actual = reply.get("roborsi_execution", {}).get("applied_action", {}).get("joint_command", command)
                stream.write(json.dumps({"step": control_steps, "action": actual, "done": reply["done"]})+"\n")
            write(case/"status.json", {"phase": "running", "model": route, "native_steps": control_steps,
                  "at": time.time(), "policy_source": str(POLICY), "score": None})
            if not reply["done"]:
                model.update_obs(current_observation(reply))
        # Persist the native terminal reply before controller/capture cleanup.
        write(case/"native-terminal.json", reply)
        receipt = model.end_episode(current_observation(reply))
        if model_cfg['storage_mode']=='training':
            try:
                training_module=importlib.import_module('XPolicyLab.policy.SenseNova-RoboRSI.training_store')
                finalized=training_module.finalize_case(case,case/'native-terminal.json')
                stores=list(case.glob('trajectories/*/seed-*/*/runtime-*/agent/training'))
                prune_native=bool(finalized and all((root.parent.parent/'storage-receipt.json').is_file() for root in stores))
            except Exception as error:
                # Storage is infrastructure accounting; the completed native
                # score remains authoritative even if export needs repair.
                write(case/'storage-review.json',{'stage':'finalize_training_source',
                    'error_type':type(error).__name__,'error':str(error),'at':time.time()})
        write(case/"status.json", {"phase": "completed", "model": route, "native_steps": control_steps,
              "at": time.time(), "wall_seconds": time.time()-started, "cleanup_receipt": receipt,
              "policy_source": str(POLICY), "score": reply.get("metrics", {}).get("score"), "native_score_at_stop": reply.get("metrics", {}).get("score"), "full_episode_evaluation": True, "episode_seconds": None, "smoke_act_limit": None,
              "native_metrics": reply.get("metrics"), "native_step_limit": step_limit})
    except BaseException as error:
        write(case/"status.json", {"phase": "technical_error", "model": route, "native_steps": control_steps,
              "at": time.time(), "wall_seconds": time.time()-started, "error_type": type(error).__name__,
              "error": str(error), "traceback": traceback.format_exc(), "score": None})
        raise
    finally:
        if model is not None:
            model.shutdown()
        if backend is not None:
            backend.close()
        # Writers have exited; no late native flush can recreate a stream.
        try:
            import shutil
            if prune_native and (native/'observations').is_dir():
                shutil.rmtree(native/'observations')
            if backend is not None and not save_video:
                for stream in native.rglob('_stream'):
                    if stream.is_dir() and not stream.is_symlink():shutil.rmtree(stream)
        except OSError as error:
            write(case/'storage-review.json',{'stage':'prune_native_cache',
                'error_type':type(error).__name__,'error':str(error),'at':time.time()})


if __name__ == "__main__":
    main()
