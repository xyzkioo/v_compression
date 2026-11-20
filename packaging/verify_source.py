#!/usr/bin/env python3
"""Copy only this project to a folder with spaces and run real CPU UI tasks."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
OWNER = 'model_compression'
FIXTURES = "from pathlib import Path\nimport json\nimport yaml\nroot = Path.cwd()\nimport torch\ntorch.manual_seed(0)\ntorch.save(torch.nn.Sequential(torch.nn.Conv2d(3,2,3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(2,2)), root / 'fixture.pth')\n"
CASE = {'config': {'project': {'root': '/fixture-root'}, 'model': {'adapter': 'torch', 'task': 'classify'}, 'export': {'path': '/fixture-root/fixture.pth'}, 'modules': {'model.parameters': True, 'compression.quantize.dynamic_int8': True, 'artifact.export': True}}, 'values': {'resource': '/fixture-root/fixture.pth', 'task': 'classify', 'device': 'cpu'}}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-python', required=True, type=Path)
    parser.add_argument('--report', type=Path, default=ROOT/'verification/independence.json')
    args = parser.parse_args()
    python = str(args.model_python.expanduser().absolute())
    with tempfile.TemporaryDirectory(prefix='source-isolation-') as directory:
        scratch = Path(directory)
        moved = scratch / 'Only this project with spaces'
        shutil.copytree(ROOT, moved, ignore=shutil.ignore_patterns('__pycache__', 'node_modules', '.venv', 'build', 'dist', 'user-data', 'verification', '.git'))
        # Keep the frontend's dist while excluding any desktop build directory.
        shutil.copytree(ROOT/'vtools_ui/webapp/frontend/dist', moved/'vtools_ui/webapp/frontend/dist')
        (moved/'user-data').mkdir()
        fixtures = scratch/'fixtures'
        fixtures.mkdir()
        env = dict(os.environ, PYTHONPATH=str(moved), QT_QPA_PLATFORM='offscreen')
        env.pop('VTOOLS_COMPONENT', None)
        env.pop('VTOOLS_USER_DATA', None)
        generated = subprocess.run([python, '-c', FIXTURES], cwd=fixtures, env=env, capture_output=True, text=True)
        if generated.returncode:
            raise RuntimeError(generated.stdout+generated.stderr)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in fixtures.iterdir() if p.is_file()}
        case = json.loads(json.dumps(CASE).replace('/fixture-root', str(fixtures)))
        file = scratch/'case.json'
        file.write_text(json.dumps({OWNER:case}))
        env.update(VTOOLS_PROBE_FIXTURE=str(file), VTOOLS_PROBE_MODEL_PYTHON=python)
        checked = subprocess.run([str(moved/'启动.sh'), '--check'], cwd=scratch, env=dict(env,VTOOLS_PYTHON=python),capture_output=True,text=True)
        if checked.returncode:
            raise RuntimeError(checked.stdout+checked.stderr)
        paths = json.loads(checked.stdout)
        assert Path(paths['ui']).is_relative_to(moved)
        assert Path(paths['runtime']).is_relative_to(moved)
        command = [python, str(moved/'packaging/smoke_probe.py')]
        result = subprocess.run(command, cwd=scratch, env=env, capture_output=True, text=True, timeout=240)
        if result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        import jsonschema
        for key in ('request', 'result'):
            schema = json.loads((ROOT/f'vtools_runtime/schemas/{key}-v1.schema.json').read_text())
            jsonschema.validate(report[key], schema)
        assert all(hashlib.sha256((fixtures/name).read_bytes()).hexdigest()==value for name,value in hashes.items())
        report.pop('request')
        report.pop('result')
        report.update(passed=True, verified_on=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(), version='1.0', input_hashes_unchanged=True, schemas_valid=True, private_import_paths=True,
                      limitations=['This probe covers CPU HTTP/API tasks; GPU/TensorRT and native desktop windows are not exercised by this probe'])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    return 0
if __name__=='__main__':
    raise SystemExit(main())
