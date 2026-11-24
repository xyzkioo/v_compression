"""Executed by a relocated independent source project: exercise its real HTTP/UI task boundary."""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
import webview  # noqa: F401 - bundled desktop dependency
import PySide6  # noqa: F401

from vtools_ui.webapp import core
from vtools_ui.webapp.catalog import APP, TOOLS
from vtools_ui.webapp.server import WorkbenchServer
from vtools_ui.webapp.tasks import TaskManager


def main():
    fixture = json.loads(Path(os.environ['VTOOLS_PROBE_FIXTURE']).read_text())
    component = APP['component']
    model_python = os.environ['VTOOLS_PROBE_MODEL_PYTHON']
    case = fixture[component]
    manager = TaskManager()
    server = WorkbenchServer(manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'

    def api(path, body=None):
        request = Request(base + '/api/' + path,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={'X-Vtools-Token': server.token, 'Content-Type': 'application/json'})
        with urlopen(request, timeout=20) as response:
            return json.load(response)

    try:
        try:
            urlopen(base + '/api/bootstrap', timeout=10)
        except HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError('API accepted request without session token')
        bootstrap = api('bootstrap')
        assert bootstrap['app']['component'] == component and not bootstrap['app']['updates_enabled']
        manifest = json.loads((core.ROOT / 'component.json').read_text(encoding='utf-8'))
        assert bootstrap['version'] == manifest['version']
        assert set(bootstrap['catalog']) == set(TOOLS)
        for other in ('model_diagnostics', 'speed_test', 'model_visualization', 'model_compression'):
            assert (core.ROOT / other).exists() == (other == component), other
        assert core.STATE_DIR == core.ROOT / 'user-data'
        html = urlopen(base + '/', timeout=10).read().decode()
        assert 'assets/index-' in html
        assert api('update/check')['status'] == 'unsupported'
        try:
            api('update/install', {})
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError('portable app installed deb update')
        from urllib.parse import quote
        document = api('config?path=' + quote(TOOLS[APP['primary_tool']]['config']))
        assert Path(document['path']).is_relative_to(core.STATE_DIR)
        template = yaml.safe_load(document['text'])
        assert not Path(template['project']['root']).is_absolute()
        assert (Path(document['path']).parent / template['project']['root']).resolve() == core.ROOT / 'inputs'
        api('config/save', {'path': document['path'], 'text': yaml.safe_dump(case['config'])})
        settings = api('settings', {'python_executable': model_python, 'results_root': 'runs'})['settings']
        assert settings['results_root'] == str(core.STATE_DIR / 'runs')
        assert json.loads(core.SETTINGS_PATH.read_text())['results_root'] == 'runs'
        request = {'tool_id': APP['primary_tool'], 'config_path': document['path'],
                   'python_executable': model_python, 'values': case['values'], 'module_edits': {}}
        api('runs', request)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = api('runs/current')['state']
            if state and state['status'] not in {'starting', 'running', 'stopping'}:
                break
            time.sleep(.15)
        else:
            raise AssertionError('UI task timed out')
        log = api('runs/log?id=' + state['id'])['text']
        assert state['status'] == 'succeeded', log
        run_dir = Path(state['result_dir'])
        result = json.loads((run_dir / 'result.json').read_text())
        assert result['status'] == 'succeeded' and result['component'] == component
        assert result['artifacts'] and all((run_dir / item['path']).is_file() for item in result['artifacts'])
        assert (run_dir / 'stdout.log').read_text().strip() in log or (run_dir / 'stdout.log').read_text().splitlines()[0] in log
        if component == 'model_visualization':
            assert any(item['path'].endswith('.png') for item in result['artifacts'])
        if component == 'model_compression':
            assert any(item['kind'] == 'model' for item in result['artifacts'])
        assert api('history')['history'][0]['status'] == '成功'
        assert api('results?root=' + quote(str(run_dir)))['files']
        # A real model/input error must reach both UI and protocol failure states.
        failed_request = dict(request, values=dict(case['values'], resource=str(core.STATE_DIR / 'missing.pt')))
        api('runs', failed_request)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = api('runs/current')['state']
            if state and state['status'] not in {'starting', 'running', 'stopping'}:
                break
            time.sleep(.15)
        assert state['status'] == 'failed', state
        failure = json.loads((Path(state['result_dir']) / 'result.json').read_text())
        assert failure['status'] == 'failed'
        report = {'component': component, 'catalog_isolated': True, 'relocated_folder': True,
                  'frontend_http': True, 'authenticated_api': True, 'editable_config': True,
                  'task_status': result['status'], 'artifacts': len(result['artifacts']),
                  'streamed_logs': True, 'visible_failure': True, 'result': result, 'request': json.loads((run_dir / 'request.json').read_text())}
        print(json.dumps(report, ensure_ascii=False))
    finally:
        manager.close()
        server.shutdown()
        server.server_close()


main()
