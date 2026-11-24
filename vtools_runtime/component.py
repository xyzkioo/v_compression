"""Private versioned process boundary for this independent project.

The worker uses each tool's existing CLI and configuration loader. This module
only validates the request, owns the job directory and records process results.
It never imports a model backend into the caller's interpreter.
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from . import __version__

SCHEMA_VERSION = 1
COMPONENTS: dict[str, dict[str, Any]] = {'model_compression': {'worker': 'model_compression.run_model_compression', 'inputs': ('weights', 'data', 'train', 'val', 'teacher_weights', 'student_weights', 'structured_initial_weights'), 'options': ('device', 'task', 'structured_scale', 'structured_method', 'structured_methods', 'structured_epochs', 'quantization_precision')}}
SELECTION = ('only', 'enable', 'disable')

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{name} 必须是 JSON 对象')
    return dict(value)

def _path(value: Any, base: Path, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} 必须是非空路径字符串')
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())

def _normalize_method_selection(options: dict[str, Any]) -> None:
    from model_compression.core.module_selection import compression_mode_modules
    from model_compression.core.pruning_methods import normalize_structured_method

    methods = options['structured_methods']
    if not isinstance(methods, list) or not methods or any(not isinstance(method, str) for method in methods):
        raise ValueError('structured_methods 必须是非空的方法 ID 数组')
    if 'structured_method' in options:
        raise ValueError('structured_methods 与 structured_method 只能使用其中一个')
    methods = [normalize_structured_method(method) for method in methods]
    if len(methods) != len(set(methods)):
        raise ValueError('structured_methods 中有重复的方法')
    structured_modules = compression_mode_modules('structured', options.get('task', 'detect'))
    if options.get('enable') or options.get('disable'):
        raise ValueError('structured_methods 不能与 enable/disable 同时使用')
    selected = options.setdefault('only', structured_modules)
    if 'compression.prune.structured' not in selected or set(selected) - set(structured_modules):
        raise ValueError('structured_methods 要求 only 包含结构化剪枝，且仅可搭配参数统计、基线评估')
    if len(methods) == 1:
        del options['structured_methods']
        options['structured_method'] = methods[0]
    else:
        options['structured_methods'] = methods

def normalize_request(component: str, request: Mapping[str, Any], base: Path) -> dict[str, Any]:
    """Resolve envelope paths; paths inside the YAML remain project.root-relative."""
    if component not in COMPONENTS:
        raise ValueError(f'未知组件：{component}')
    spec = COMPONENTS[component]
    raw = _object(request, 'request')
    unknown = set(raw) - {'schema_version', 'component', 'config', 'run_dir', 'inputs', 'options'}
    if unknown:
        raise ValueError(f"未知请求字段：{', '.join(sorted(unknown))}")
    if type(raw.get('schema_version')) is not int or raw['schema_version'] != SCHEMA_VERSION:
        raise ValueError('schema_version 必须是整数 1')
    if raw.get('component') != component:
        raise ValueError(f'component 必须是 {component}')
    inputs = _object(raw.get('inputs', {}), 'inputs')
    options = _object(raw.get('options', {}), 'options')
    for (values, allowed, label) in ((inputs, spec['inputs'], 'inputs'), (options, (*spec['options'], *SELECTION), 'options')):
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"{component} 不支持 {label}：{', '.join(sorted(unknown))}")
    for (key, value) in inputs.items():
        if key == 'adapter' and isinstance(value, str) and value and (not any((c in value for c in ('/', '\\')))) and (not value.endswith('.py')):
            continue
        inputs[key] = _path(value, base, f'inputs.{key}')
    for (key, value) in options.items():
        if key in SELECTION:
            if not isinstance(value, list) or any((not isinstance(x, str) or not x.strip() for x in value)):
                raise ValueError(f'options.{key} 必须是模块 ID 字符串数组；[] 表示空选择')
        elif key == 'structured_epochs':
            if type(value) is not int or value < 0:
                raise ValueError('structured_epochs 必须是非负整数')
        elif key != 'structured_methods' and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'options.{key} 必须是非空字符串')
    if 'structured_methods' in options:
        _normalize_method_selection(options)
    if 'quantization_precision' in options:
        from model_compression.modules.onnx_quantization import ONNX_PRECISIONS

        if options['quantization_precision'] not in ONNX_PRECISIONS:
            raise ValueError('quantization_precision 须为 fp32、fp16 或 int8')
    if 'only' in options and (options.get('enable') or options.get('disable')):
        raise ValueError('only 不能与 enable/disable 同时使用')
    config = _path(raw.get('config'), base, 'config')
    run_dir = _path(raw.get('run_dir'), base, 'run_dir')
    output = Path(run_dir)
    for value in (config, *inputs.values()):
        candidate = Path(value)
        if candidate.is_absolute() and (candidate == output or output in candidate.parents):
            raise ValueError('输入路径不能位于本次新建的 run_dir 内')
    return {'schema_version': 1, 'component': component, 'config': config, 'run_dir': run_dir, 'inputs': inputs, 'options': options}

def _write_text(path: Path, payload: str) -> None:
    """An interrupted writer never leaves half a manifest or configuration."""
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def _snapshot_config(request: dict[str, Any], root: Path) -> Path:
    import yaml
    source = Path(request['config'])
    config = _object(yaml.safe_load(source.read_text(encoding='utf-8')), 'config')
    project = _object({} if config.get('project') is None else config['project'], 'project')
    root_value = project.get('root')
    project['root'] = _path('.' if root_value in (None, '') else root_value, source.parent, 'project.root')
    config['project'] = project
    if config.get('export') is not None:
        export = _object(config['export'], 'export')
        if export.get('path'):
            name = Path(_path(export['path'], source.parent, 'export.path')).name
            export['path'] = str(root / 'artifacts' / 'exports' / name)
        config['export'] = export
    target = root / 'config.snapshot.yaml'
    _write_text(target, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return target

def _arguments(request: dict[str, Any], config: Path, artifacts: Path) -> list[str]:
    args = ['--config', str(config), '--run-dir', str(artifacts)]
    for (key, value) in request['inputs'].items():
        flag = key.replace('_', '-')
        args.extend([f'--{flag}', str(value)])
    for (key, value) in request['options'].items():
        flag = f"--{key.replace('_', '-')}"
        if key in SELECTION:
            for item in value or ([''] if key == 'only' else []):
                args.extend([flag, item])
        else:
            args.extend([flag, str(value)])
    return args

def _artifacts(root: Path) -> list[dict[str, Any]]:
    items = []
    for path in sorted((root / 'artifacts').rglob('*')):
        relative = path.relative_to(root)
        if '_input' in relative.parts or path.is_symlink() or (not path.is_file()):
            continue
        if root not in path.resolve().parents:
            continue
        items.append({'path': relative.as_posix(), 'size_bytes': path.stat().st_size, 'kind': 'model' if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.onnx', '.engine', '.kmodel', '.tflite', '.torchscript'} else 'report'})
    return items

def _run_structured_batch(
    request: dict[str, Any], snapshot: Path, root: Path, result: dict[str, Any],
    stdout: BinaryIO, stream_logs: bool,
) -> None:
    options = request['options'].copy()
    methods = options.pop('structured_methods')
    artifacts = root / 'artifacts'
    artifacts.mkdir()
    summary = artifacts / 'structured-methods.json'
    entries: list[dict[str, Any]] = [
        {'method': method, 'status': 'pending', 'result': f'{method}/result.json', 'error': None}
        for method in methods
    ]
    write_json(summary, {'methods': entries})
    for entry in entries:
        method = entry['method']
        entry['status'] = 'running'
        write_json(summary, {'methods': entries})
        message = f'开始运行结构化剪枝方法：{method}\n'
        stdout.write(message.encode('utf-8'))
        stdout.flush()
        if stream_logs:
            print(message, end='', file=sys.stderr, flush=True)
        child = execute_request(
            request['component'],
            {
                **request, 'config': str(snapshot), 'run_dir': str(artifacts / method),
                'options': {**options, 'structured_method': method},
            },
            stream_logs=stream_logs,
        )
        entry.update(status=child['status'], error=child['error'])
        write_json(summary, {'methods': entries})
        if child['status'] == 'cancelled':
            raise KeyboardInterrupt
    failed = [entry['method'] for entry in entries if entry['status'] != 'succeeded']
    if failed:
        message = f'方法执行失败：{"、".join(failed)}，详见 artifacts/structured-methods.json'
        result.update(status='failed', exit_code=2, error={'type': 'BatchFailed', 'message': message})
    else:
        result.update(status='succeeded', exit_code=0)

def _run_worker(
    component: str, request: dict[str, Any], snapshot: Path, root: Path,
    result: dict[str, Any], stdout: BinaryIO, stderr: BinaryIO, stream_logs: bool,
) -> int:
    worker = COMPONENTS[component]['worker']
    command = [sys.executable, '-m', worker, *_arguments(request, snapshot, root / 'artifacts')]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE if stream_logs else stdout, stderr=subprocess.PIPE if stream_logs else stderr, start_new_session=os.name != 'nt')
    pumps = []
    pump_errors = []
    try:
        if stream_logs:
            def pump(source, destination):
                try:
                    for chunk in iter(source.readline, b''):
                        destination.write(chunk)
                        destination.flush()
                        sys.stderr.write(chunk.decode('utf-8', errors='replace'))
                        sys.stderr.flush()
                except (OSError, ValueError) as error:
                    pump_errors.append(error)
                finally:
                    source.close()

            for source, destination in ((process.stdout, stdout), (process.stderr, stderr)):
                thread = threading.Thread(target=pump, args=(source, destination), daemon=True)
                thread.start()
                pumps.append(thread)
        backend_code = process.wait()
    finally:
        if process.poll() is None:
            if os.name == 'nt':
                process.terminate()
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        result['backend_exit_code'] = process.returncode
        for thread in pumps:
            thread.join(timeout=5)
    if pump_errors:
        raise OSError(f'日志转发失败：{pump_errors[0]}')
    return backend_code

def execute_request(component: str, request: Mapping[str, Any], *, base: Path | None=None, stream_logs: bool=False) -> dict[str, Any]:
    """Synchronous Python API. The returned object is identical to result.json.

    Invalid envelopes and existing run directories raise ValueError/OSError
    without modifying that directory. Failures after allocation return a failed
    result and preserve the logs. A successful process is not a quality claim;
    inspect the component's domain reports for metrics and acceptance results.
    """
    normalized = normalize_request(component, request, (base or Path.cwd()).resolve())
    root = Path(normalized['run_dir'])
    root.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {'schema_version': 1, 'component': component, 'component_version': __version__, 'status': 'running', 'run_dir': str(root), 'started_at': _now(), 'finished_at': None, 'exit_code': None, 'backend_exit_code': None, 'error': None, 'artifacts': [], 'request': 'request.json', 'config': 'config.snapshot.yaml', 'logs': {'stdout': 'stdout.log', 'stderr': 'stderr.log'}}
    try:
        write_json(root / 'request.json', normalized)
        write_json(root / 'result.json', result)
        with (root / 'stdout.log').open('wb') as stdout, (root / 'stderr.log').open('wb') as stderr:
            snapshot = _snapshot_config(normalized, root)
            if 'structured_methods' in normalized['options']:
                _run_structured_batch(normalized, snapshot, root, result, stdout, stream_logs)
            else:
                backend_code = _run_worker(component, normalized, snapshot, root, result, stdout, stderr, stream_logs)
                skipped = backend_code == 0 and normalized['options'].get('only') == []
                result.update(backend_exit_code=backend_code, exit_code=0 if backend_code == 0 else 2, status='skipped' if skipped else 'succeeded' if backend_code == 0 else 'failed')
                if backend_code != 0:
                    result['error'] = {'type': 'BackendFailed', 'message': f'后端退出码 {backend_code}，详见 stdout.log、stderr.log 与业务报告'}
                    from model_compression.core.interrupted_run import write_interrupted_summary

                    try:
                        stdout.flush()
                        write_interrupted_summary(root / 'artifacts', root / 'stdout.log', backend_code, result['started_at'])
                    except (OSError, ValueError) as error:
                        stderr.write(f'无法补写失败摘要：{error}\n'.encode('utf-8'))
    except KeyboardInterrupt:
        result.update(status='cancelled', exit_code=130, error={'type': 'Cancelled', 'message': '用户中断运行'})
    except Exception as error:
        result.update(status='failed', exit_code=2, error={'type': type(error).__name__, 'message': str(error)})
        with (root / 'stderr.log').open('a', encoding='utf-8') as stream:
            stream.write(f'{type(error).__name__}: {error}\n')
    try:
        result['artifacts'] = _artifacts(root)
    except OSError as error:
        result.update(status='failed', exit_code=2, error={'type': type(error).__name__, 'message': str(error)})
    result['finished_at'] = _now()
    if not (root / 'config.snapshot.yaml').is_file():
        result['config'] = None
    write_json(root / 'result.json', result)
    return result

def component_main(component: str, argv: list[str] | None=None) -> int:
    spec = COMPONENTS[component]
    parser = argparse.ArgumentParser(description=f'{component}：JSON 请求和统一运行结果')
    parser.add_argument('--request', type=Path, help='版本 1 JSON 请求文件；相对路径以该文件目录为基准')
    parser.add_argument('--config', help='原生 YAML/JSON 配置；相对路径以当前工作目录为基准')
    parser.add_argument('--run-dir', help='本次新建的运行目录，不允许覆盖或复用')
    parser.add_argument('--list-modules', action='store_true', help='列出原生模块 ID')
    parser.add_argument('--stream-logs', action='store_true', help='向 stderr 同步后端日志；stdout 仍只有结果 JSON')
    for key in spec['inputs']:
        parser.add_argument('--' + key.replace('_', '-'))
    for key in spec['options']:
        if key == 'structured_methods':
            parser.add_argument('--structured-methods', nargs='+', metavar='METHOD')
        else:
            parser.add_argument('--' + key.replace('_', '-'), type=int if key == 'structured_epochs' else str)
    for key in SELECTION:
        parser.add_argument('--' + key, action='append')
    args = parser.parse_args(argv)
    if args.list_modules:
        return subprocess.call([sys.executable, '-m', spec['worker'], '--list-modules'])
    try:
        if args.request:
            if any((getattr(args, key) is not None for key in ('config', 'run_dir', *spec['inputs'], *spec['options'], *SELECTION))):
                raise ValueError('--request 不可与配置、输入或选项覆盖混用')
            path = args.request.expanduser().resolve()
            request = json.loads(path.read_text(encoding='utf-8'))
            base = path.parent
        else:
            request = {'schema_version': 1, 'component': component, 'config': args.config, 'run_dir': args.run_dir, 'inputs': {key: getattr(args, key) for key in spec['inputs'] if getattr(args, key) is not None}, 'options': {key: getattr(args, key) for key in spec['options'] if getattr(args, key) is not None}}
            for key in SELECTION:
                if getattr(args, key) is not None:
                    request['options'][key] = [item.strip() for value in getattr(args, key) for item in value.split(',') if item.strip()]
            base = Path.cwd()
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            result = execute_request(component, request, base=base, stream_logs=args.stream_logs)
        finally:
            signal.signal(signal.SIGTERM, previous)
    except (ValueError, TypeError, OSError) as error:
        result = {'schema_version': 1, 'component': component, 'status': 'failed', 'exit_code': 2, 'run_dir': None, 'error': {'type': type(error).__name__, 'message': str(error)}}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return int(result['exit_code'])
