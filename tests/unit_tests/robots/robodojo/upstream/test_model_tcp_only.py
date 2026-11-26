"""A disabled VLA must never attach a shared service or resolve its checkpoint."""
from importlib import import_module
from types import SimpleNamespace
import threading
import json

import pytest


def test_disabled_model_ignores_shared_vla_and_checkpoint(monkeypatch):
    module = import_module("robots.robodojo.agents.submission.model")
    model = module.Model.__new__(module.Model)
    model.model_cfg = {"vla_backend": "disabled"}
    model._vla = None
    monkeypatch.setenv("ROBORSI_SHARED_VLA_PORT", "not-a-valid-port")

    def forbidden(*args, **kwargs):
        raise AssertionError("TCP-only attempted to start or attach a VLA service")

    monkeypatch.setattr(module, "_SharedVLA", forbidden)
    monkeypatch.setattr(module, "resolve_checkpoint_root", forbidden)
    assert model._ensure_vla() is None
    assert model._vla is None


def test_experiment_rejects_vla_before_native_execution_dispatch(monkeypatch, tmp_path):
    module = import_module("robots.robodojo.agents.submission.skillpack_agent")
    profile = import_module("robots.robodojo.agents.submission.execution_profile")
    calls = []

    class Base:
        def __init__(self, *args): pass
        def request(self, value):
            calls.append(value)
            return {"accepted": True}

    runtime = SimpleNamespace(native_request=SimpleNamespace(make_experiment_class=lambda *a, **k: Base))
    monkeypatch.setattr(profile, "runtime_modules", lambda path: runtime)
    reference = SimpleNamespace(runner=SimpleNamespace(InvalidAction=ValueError))
    context = SimpleNamespace(reference=reference, backend=None, model_cfg={"vla_backend": "disabled"})
    agent = module.SkillpackEmbodiedAgent.__new__(module.SkillpackEmbodiedAgent)
    agent._skillpack_toolkit = SimpleNamespace(operation_thread=threading.get_ident())
    experiment = agent.make_experiment(context, {}, tmp_path)
    for request in [{"op": "vla_propose", "subgoal": "pick"},
                    {"op": "vla_review", "proposal_id": "cached", "decision": "accept"}]:
        with pytest.raises(ValueError, match="VLA is disabled"):
            experiment.request(request)
    assert not calls
    assert experiment.request({"op": "act_eef"}) == {"accepted": True}
    assert calls == [{"op": "act_eef"}]


@pytest.mark.parametrize('code, reason, policy, native_done, terminal, expect_finish', [
    (0,'agent_exited','native_finish',False,'turn.completed',True),
    (0,'agent_exited','native_finish',False,'turn.failed',False),
    (1,'agent_exited','native_finish',False,'turn.completed',False),
    (0,'budget_exhausted','native_finish',False,'turn.completed',False),
    (0,'adapter_error','native_finish',False,'turn.completed',False),
    (0,'agent_exited','protocol_error',False,'turn.completed',False),
    (0,'agent_exited','native_finish',True,'turn.completed',False),
])
def test_clean_exit_finalizes_incomplete_once_without_motion(tmp_path, monkeypatch, code, reason, policy, native_done, terminal, expect_finish):
    module = import_module('robots.robodojo.agents.submission.skillpack_agent')
    calls = []
    class Base:
        def __init__(self, cfg, run, backend):
            self.run, self.backend = run, backend
            self.phase, self.fault = 'evaluation', None
            self.latest = dict(done=native_done, observation={'control_step':17})
            self.active, self.results = {'recorded':False}, []
        def finalize(self, reason):
            if not self.active['recorded']:
                self.results.append(reason)
                self.active['recorded']=True
        def accept(self, reply, reason):
            self.latest=reply
            self.finalize(reason)
    def finish(**request):
        calls.append(request)
        return dict(done=True, observation={'control_step':17})
    profile = import_module('robots.robodojo.agents.submission.execution_profile')
    monkeypatch.setattr(profile, 'runtime_modules', lambda _: SimpleNamespace(native_request=SimpleNamespace(make_experiment_class=lambda *a, **k: Base)))
    agent = object.__new__(module.SkillpackEmbodiedAgent)
    segment=tmp_path/'segment';segment.mkdir()
    (segment/'model.jsonl').write_text(json.dumps({'type':terminal})+'\n')
    agent._hybrid_planner=SimpleNamespace(_last_agent_process=SimpleNamespace(returncode=code),
                                        _last_agent_segment=segment)
    reference=SimpleNamespace(runner=SimpleNamespace(Experiment=Base,InvalidAction=ValueError),
        hybrid_host=SimpleNamespace(request=lambda *a:None, ORIGINAL=lambda *a:None))
    context=SimpleNamespace(reference=reference,backend=SimpleNamespace(request=finish),model_cfg={})
    experiment=agent.make_experiment(context, {'options':{'agent_exit_policy':policy}}, tmp_path)
    experiment.finalize(reason)
    experiment.finalize(reason)
    assert calls==([{'op':'finish'}] if expect_finish else [])
    assert experiment.latest['observation']['control_step']==17
    assert experiment.results==(['agent_exit'] if expect_finish else [reason])
    receipt=tmp_path/'agent-exit.json'
    assert receipt.is_file() is expect_finish
    if expect_finish:
        assert json.loads(receipt.read_text())['policy_status']=='incomplete'



def test_resume_cannot_change_implicit_exit_policy():
    from copy import deepcopy
    resume = import_module("robots.robodojo.agents.submission.hybrid_resume")
    old = dict(episode_seconds=None, options={"review_window_steps": 100})
    fresh = deepcopy(old)
    fresh["options"]["agent_exit_policy"] = "protocol_error"
    resume.validate_policy(old, fresh)
    fresh["options"]["agent_exit_policy"] = "native_finish"
    with pytest.raises(ValueError, match="original action/review/context options"):
        resume.validate_policy(old, fresh)


def test_disabled_preflight_never_imports_vla_runtime(monkeypatch):
    import builtins
    module = import_module("robots.robodojo.agents.submission.preflight")
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.endswith(("pi05_runtime", "simate_runtime", "pi05_source")):
            raise AssertionError("TCP-only preflight attempted to load a VLA runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert module.check_vla_runtime({"vla_backend": "disabled"}) == {"enabled": False, "backend": "disabled"}
