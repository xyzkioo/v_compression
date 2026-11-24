"""Convert the existing graphical form to the stable component request v1."""
from __future__ import annotations
import json
import sys
import uuid
from pathlib import Path
from .catalog import APP, ROOT, TOOLS

def build_component_command(request: dict) -> tuple[str, list[str], str]:
    from .core import STATE_DIR, _atomic_text, _field_path, _option, _visible_run_name, load_config, python_executable_path, read_settings
    component = APP['component']
    tool_id = str(request.get('tool_id', ''))
    if not component or tool_id != APP['primary_tool'] or tool_id not in TOOLS:
        raise ValueError('此目录只支持自身组件')
    values = request.get('values') or {}
    if not isinstance(values, dict):
        raise ValueError('参数必须是对象')
    loaded = load_config(_field_path(request.get('config_path') or TOOLS[tool_id]['config']))
    settings = read_settings()
    executable = python_executable_path(str(request.get('python_executable') or settings['python_executable']))
    if not executable.is_file():
        raise ValueError(f'Python 解释器不存在：{executable}')
    if getattr(sys, 'frozen', False) and executable.resolve() == Path(sys.executable).resolve():
        raise ValueError('请选择外部 Python 模型环境')
    (inputs, options) = ({}, {})
    from model_compression.core.module_selection import COMPRESSION_MODES, compression_mode_modules
    task = _option(values.get('task'), ['detect', 'classify'], '模型任务')
    mode = values.get('processing_mode')
    mode_modules = None
    if mode is not None:
        mode_modules = compression_mode_modules(str(mode), task or loaded['data'].get('model', {}).get('task', 'detect'))
    resource = values.get('resource')
    if resource and (mode is None or COMPRESSION_MODES[str(mode)]['public_weights']):
        path = _field_path(resource)
        key = 'weights'
        inputs[key] = path
    keys = ('data', 'train', 'val', 'teacher_weights', 'student_weights')
    for key in keys:
        if task == 'detect' and key in {'train', 'val'} or (task == 'classify' and key in {'data', 'teacher_weights', 'student_weights'}):
            continue
        if mode is not None and key in {'teacher_weights', 'student_weights'} and mode != 'detection_distillation':
            continue
        if values.get(key):
            inputs[key] = _field_path(values[key])
    if task:
        options['task'] = task
    if values.get('device') and values['device'] != 'auto':
        options['device'] = _option(values['device'], ['auto', 'cpu', 'cuda:0'], '设备')
    edits = request.get('module_edits') or {}
    if not isinstance(edits, dict):
        raise ValueError('模块修改必须是对象')
    if mode_modules is not None and edits:
        raise ValueError('选择处理模式时不能同时指定模块开关。')
    if mode_modules is not None:
        options['only'] = mode_modules
    if mode == 'detection_quantization':
        weights = resource or loaded['data'].get('model', {}).get('weights')
        if weights and Path(weights).suffix.lower() != '.pt':
            raise ValueError('模型量化需要 YOLO .pt 检测权重。')
        options['device'] = 'cpu'
        if values.get('quantization_precision'):
            options['quantization_precision'] = values['quantization_precision']
    if mode == 'structured':
        if 'structured_methods' in values:
            options['structured_methods'] = values['structured_methods']
        if values.get('structured_method'):
            from model_compression.core.pruning_methods import normalize_structured_method

            options['structured_method'] = normalize_structured_method(values['structured_method'])
    available = {item[0] for item in TOOLS[tool_id].get('modules', [])}
    configured = loaded['data'].get('modules', {})
    if isinstance(configured, dict):
        available.update(configured)
    for (module, enabled) in edits.items():
        if module not in available or not isinstance(enabled, bool):
            raise ValueError(f'无效模块：{module}')
        options.setdefault('enable' if enabled else 'disable', []).append(module)
    output = Path(settings['results_root']) / tool_id
    output.mkdir(parents=True, exist_ok=True)
    envelope = {'schema_version': 1, 'component': component, 'config': loaded['path'], 'run_dir': str(output / _visible_run_name(output)), 'inputs': inputs, 'options': options}
    from vtools_runtime.component import normalize_request
    envelope = normalize_request(component, envelope, ROOT)
    path = STATE_DIR / 'requests' / (uuid.uuid4().hex + '.json')
    _atomic_text(path, json.dumps(envelope, ensure_ascii=False, indent=2))
    return (str(executable), ['-m', component, '--request', str(path), '--stream-logs'], loaded['path'])
