"""Optional real Codex code-mode/MCP test with a synthetic, local-only provider."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

P = 'robots.robodojo.agents.submission.'
profile = import_module(P+'execution_profile')
protocol = import_module(P+'cci_protocol')
POLICY = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


@pytest.mark.parametrize('vla_backend', ['simate_beta', 'disabled'])
def test_real_code_mode_robot_bindings_publish_registered_constraints(tmp_path, reference, vla_backend):
    binary = os.environ.get('ROBORSI_OFFLINE_CODEX_BIN')
    if not binary:
        pytest.skip('Set ROBORSI_OFFLINE_CODEX_BIN to the pinned 0.160.1 CLI for this offline integration test')
    assert subprocess.check_output([binary, '--version'], text=True).strip() == 'codex-cli 0.160.1'
    from roborsi.planner.utils.http_mcp_server import HttpMcpServer
    from roborsi.tools import ToolResult

    frozen = profile.load_skillpack_profile(POLICY, 'align_blocks', tmp_path, vla_backend=vla_backend)
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'preparation').mkdir()
    core = workspace/'core.md'
    core.write_text(frozen.core)
    catalog = profile.stage_cci_catalog(workspace, 'fixture', reasoning_effort='medium')
    names = sorted(frozen.expected_tools)
    calls = []
    run = tmp_path/'toolkit'
    for name in ('workspace', 'observations'): (run/name).mkdir(parents=True)
    def no_native_call(request):
        raise AssertionError('The protocol fixture must not execute a Native/model request')
    registered = profile.start_toolkit(SimpleNamespace(request=no_native_call), run, tmp_path,
                                       dict(vla_backend=vla_backend))
    specs = registered.get_tools_spec()
    arm = dict(position=[0,0,0], pitch_deg=0, roll_deg=0, yaw_deg=0, gripper_opening=1)
    arguments = dict(observe={}, hold_current={}, vla_propose=dict(subgoal='Synthetic hold.'),
        act=dict(targets=[dict(left=arm, right=arm)], reason='Synthetic protocol fixture.',
                 assessment=dict(current_subgoal='Hold.')),
        close_in_place=dict(arm='left'), move_relative=dict(arm='left', delta_m=[0,0,0]),
        spatial_localize_pixel=dict(camera_id='cam_head',
            observation_id='fixture:00000_spatial_case', pixel=[1,1],
            coordinate_frame='environment_origin', reference_point='visible_contact',
            known_height_m=0))

    class Toolkit:
        def get_tools_spec(self):
            return deepcopy(specs)

        def execute_tool(self, name, arguments):
            calls.append(dict(name=name, arguments=deepcopy(arguments)))
            return ToolResult(name=name, result=dict(synthetic=True, tool=name, arguments=arguments))

    robot = HttpMcpServer(Toolkit())
    robot.start()
    captured = []
    capture_errors = []
    program = ('text({binding: typeof tools.mcp__robot__observe});\n'
               + '\n'.join('text(await tools.mcp__robot__'+name+'('+json.dumps(arguments[name])+'));' for name in names)
               + '\ntext({vla: typeof tools.mcp__robot__vla_propose});')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            raw = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            # The same frozen model-facing transform used in production.
            (tmp_path/('request-'+str(len(runs))+'-'+str(len(captured))+'.json')).write_text(json.dumps(raw))
            try:
                selected = profile.RobotCodeModeProfile(frozen, Toolkit().get_tools_spec()) if runs else frozen
                normalized, _ = selected.transform(raw)
            except ValueError as error:
                capture_errors.append(str(error))
                body = json.dumps(dict(error=dict(message=str(error)))).encode()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            captured.append(dict(raw=raw, forwarded=normalized))
            if len(captured) == 1:
                output = [dict(type='custom_tool_call', id='fc1', call_id='c1',
                               name='exec', namespace='functions', input=program, status='completed')]
            elif len(captured) == 2:
                output = [dict(type='function_call', id='fc2', call_id='c2', name='observe',
                               namespace='mcp__robot', arguments='{}', status='completed')]
            else:
                output = [dict(type='message', id='m1', role='assistant', status='completed',
                               content=[dict(type='output_text', text='offline complete', annotations=[])])]
            response = dict(id='r'+str(len(captured)), status='completed', output=output,
                            usage=dict(input_tokens=1, output_tokens=1, total_tokens=2))
            body = protocol.sse(response)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runs = []
    try:
        os_tmp = tmp_path/'os-tmp'
        os_tmp.mkdir()
        for omitted in (['code_mode'], []):
            calls.clear()
            captured.clear()
            home = tmp_path/('old-home' if omitted else 'new-home')
            home.mkdir()
            env = {k:v for k,v in os.environ.items() if k not in (
                'OPENAI_API_KEY', 'CODEX_API_KEY', 'CODEX_THREAD_ID', 'CODEX_HOME', 'CODEX_SESSION_ID')}
            env.update(CODEX_HOME=str(home), TMPDIR=str(os_tmp), OPENAI_API_KEY='offline-fixture')
            options = {
                'model_provider': 'offline',
                'model_providers.offline': dict(name='Offline fixture', base_url=f'http://127.0.0.1:{server.server_port}/v1',
                                                wire_api='responses', requires_openai_auth=False),
                'model_catalog_json': str(catalog), 'model_instructions_file': str(core),
                'features.code_mode': True, 'features.code_mode_only': False,
                'mcp_servers.robot': dict(url=robot.url, enabled=True, required=True,
                    omit_tools_from=omitted, startup_timeout_sec=10, tool_timeout_sec=10),
                'model_context_window': 200000, 'model_auto_compact_token_limit': 200000,
            }
            command = [binary, 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral',
                       '--skip-git-repo-check', '-C', str(workspace), '-s', 'danger-full-access',
                       '-m', 'fixture', '--json']
            # Each -c value is parsed as TOML; JSON objects use '=' in TOML.
            def toml(value):
                if isinstance(value, dict): return '{'+','.join(k+'='+toml(v) for k,v in value.items())+'}'
                if isinstance(value, list): return '['+','.join(toml(v) for v in value)+']'
                return json.dumps(value)
            for key,value in options.items(): command.extend(['-c', key+'='+toml(value)])
            command.append('Synthetic offline protocol test.')
            result = subprocess.run(command, input='', text=True, capture_output=True, timeout=45, env=env)
            (tmp_path/('old-cli.jsonl' if omitted else 'new-cli.jsonl')).write_text(result.stdout)
            (tmp_path/('old-cli.stderr' if omitted else 'new-cli.stderr')).write_text(result.stderr)
            assert result.returncode == 0, result.stdout+result.stderr+str(capture_errors)
            assert len(captured) == 3, result.stdout+result.stderr
            runs.append(dict(requests=deepcopy(captured), calls=deepcopy(calls)))
    finally:
        server.shutdown()
        server.server_close()
        robot.stop()
    old, new = runs
    assert old['calls'] == [dict(name='observe', arguments={})]
    assert new['calls'] == [dict(name=name, arguments=arguments[name]) for name in names]+[dict(name='observe', arguments={})]
    # Only registered parameter schemas change; names, prose and instructions
    # remain pinned. The real CLI loses these bounds even on direct MCP tools.
    expected = deepcopy(old['requests'][0]['forwarded'])
    registered = {s['name']:s['input_schema'] for s in specs}
    for item in expected['input']:
        if item.get('type') == 'additional_tools':
            for group in item['tools']:
                if group.get('name') == 'mcp__robot':
                    for spec in group['tools']:
                        assert 'maxItems' not in spec['parameters'].get('properties', {}).get('targets', {})
                        spec['parameters'] = deepcopy(registered[spec['name']])
    for key in ('tools', 'instructions'):
        assert expected.get(key) == new['requests'][0]['forwarded'].get(key)
    def visible_input(request):
        return [{k:v for k,v in item.items() if k!='id'} for item in request['input']]
    # Separate CLI invocations create different opaque message IDs. All tool
    # declarations, descriptions, instructions and history content must match.
    assert visible_input(expected) == visible_input(new['requests'][0]['forwarded'])
    old_cci = protocol.request_to_cci(expected)
    new_cci = protocol.request_to_cci(new['requests'][0]['forwarded'])
    old_cci['input'], new_cci['input'] = visible_input(old_cci), visible_input(new_cci)
    assert old_cci == new_cci
    assert json.dumps(old_cci, ensure_ascii=False) == json.dumps(new_cci, ensure_ascii=False)
    before = json.dumps(old['requests'][1]['forwarded']['input'], ensure_ascii=False)
    after = json.dumps(new['requests'][1]['forwarded']['input'], ensure_ascii=False)
    assert 'not a function' in before
    assert 'synthetic' in after
    def text_values(value):
        if isinstance(value, str): return value
        if isinstance(value, list): return '\n'.join(text_values(v) for v in value)
        if isinstance(value, dict): return '\n'.join(text_values(v) for v in value.values())
        return ''
    expected_vla = 'undefined' if vla_backend == 'disabled' else 'function'
    assert json.dumps(dict(vla=expected_vla), separators=(',', ':')) in text_values(new['requests'][1]['forwarded']['input'])
    assert program in [i.get('input') for i in new['requests'][1]['forwarded']['input']]
    (tmp_path/'compatibility.actual.json').write_text(json.dumps(dict(
        cli_version='0.160.1', cli_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        vla_backend=vla_backend, old_calls=old['calls'], new_calls=new['calls'],
        visible_prefix_sha256=hashlib.sha256(json.dumps(new_cci, ensure_ascii=False).encode()).hexdigest(),
        visible_prefix_unchanged=False, registered_parameter_schemas_restored=True,
        descriptions_instructions_and_history_unchanged=True,
        JS_source_unchanged=True, Model_calls=0, Native_calls=0), indent=2))
