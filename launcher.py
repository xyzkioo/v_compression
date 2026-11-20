#!/usr/bin/env python3
"""Start this project's UI; native double-click startup keeps a diagnostic log."""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def show_error(message, env):
    print(message, file=sys.stderr)
    dialog = shutil.which('zenity')
    if dialog and (env.get('DISPLAY') or env.get('WAYLAND_DISPLAY')):
        subprocess.run([dialog, '--error', '--title=v_compression 启动失败', '--width=520', '--text', message], env=env, check=False)

def main():
    options = sys.argv[1:]
    desktop = '--desktop-launch' in options
    options = [value for value in options if value != '--desktop-launch']
    candidates = [os.environ.get('VTOOLS_PYTHON'), str(ROOT / '.venv/bin/python'),
                  str(Path.home() / 'python-envs/vtools/.venv/bin/python'), sys.executable]
    executable = next((str(Path(p).expanduser().absolute()) for p in candidates if p and Path(p).expanduser().is_file()), None)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
    env.pop('VTOOLS_COMPONENT', None)
    env.setdefault('VTOOLS_ULTRALYTICS_REPO', str(Path.home() / 'Projects/libraries/ultralytics-cn'))
    env.setdefault('VTOOLS_USER_DATA', str(ROOT / 'user-data'))
    if not executable:
        show_error('找不到 Python；请用 VTOOLS_PYTHON 指定已安装依赖的解释器。', env)
        return 127
    if options == ['--check']:
        probe = "from vtools_ui.standalone import configure,identity;configure();from vtools_ui.webapp.catalog import public_catalog;import vtools_ui,vtools_runtime;import json;print(json.dumps(dict(app=identity(),tools=list(public_catalog()),ui=vtools_ui.__file__,runtime=vtools_runtime.__file__),ensure_ascii=False))"
        return subprocess.call([executable, '-c', probe], cwd=ROOT, env=env)
    command = [executable, '-m', 'vtools_ui', *options]
    if not desktop:
        return subprocess.call(command, cwd=ROOT, env=env)
    log_path = Path(env['VTOOLS_USER_DATA']).expanduser() / 'launcher.log'
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\n{datetime.now().isoformat(timespec="seconds")} · 启动 · Python: {executable}\n')
            log.flush()
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    except OSError as error:
        show_error(f'无法启动：{error}\n项目目录：{ROOT}', env)
        return 127
    if code != 0:
        show_error(f'程序启动或运行失败（退出码 {code}）。\n请查看日志：{log_path}\n可通过 VTOOLS_PYTHON 指定已安装依赖的 Python 环境。', env)
    return code

if __name__ == '__main__':
    raise SystemExit(main())
