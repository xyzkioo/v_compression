"""Detection evaluation and checkpoint pruning (multiple boxes per image)."""
from __future__ import annotations

from pathlib import Path
import copy
import math
import os
import shutil
import statistics
import time
from typing import Any, Callable, Mapping, TypeVar

from .detection_adapter import DetectionAdapter
from .model_runtime import parameter_metrics, resolve_device
from .provenance import artifact_info
from .pruning_methods import normalize_structured_method
from .run_manager import write_json
from ..modules.pruning import prune_unstructured
from ..modules.dependency_pruning import prune_detector_dependency_aware
from ..modules.structured import build_structured_detector, finetune_structured_detector

_T = TypeVar('_T')


def _cuda_device(torch: Any, value: Any) -> Any:
    text = str(value or 'auto').strip().lower()
    if text == 'auto':
        return torch.device('cuda', torch.cuda.current_device())
    if text.isdigit():
        text = f'cuda:{text}'
    return torch.device(text)


def _cuda_memory_note(*, include_peak: bool = False, device_value: Any = 'auto') -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return ''
        device = _cuda_device(torch, device_value)
        if device.type != 'cuda':
            return ''
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        gib = 1024 ** 3
        note = (f'；CUDA 空闲 {free / gib:.2f}/{total / gib:.2f} GiB，'
                f'{device} 本进程已分配 {allocated / gib:.2f} GiB、已保留 {reserved / gib:.2f} GiB')
        if include_peak:
            note += (f'，本阶段分配峰值 {torch.cuda.max_memory_allocated(device) / gib:.2f} GiB、'
                     f'保留峰值 {torch.cuda.max_memory_reserved(device) / gib:.2f} GiB')
        return note
    except (ImportError, RuntimeError, AssertionError):
        return ''


def _structured_stage(name: str, action: Callable[[], _T], *, device_value: Any = 'auto') -> _T:
    try:
        import torch
        device = _cuda_device(torch, device_value)
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
    except (ImportError, RuntimeError, AssertionError):
        pass
    print(f'[结构化剪枝] 开始：{name}{_cuda_memory_note(device_value=device_value)}', flush=True)
    try:
        result = action()
    except Exception as exc:
        detail = str(exc)
        if 'out of memory' in detail.lower() or 'cuda error: out of memory' in detail.lower():
            detail += '；可降低评估 batch_size、输入尺寸、微调 batch_size 或改用 CPU/更大显存设备'
        raise RuntimeError(f'结构化剪枝阶段「{name}」失败{_cuda_memory_note(include_peak=True, device_value=device_value)}：{detail}') from exc
    print(f'[结构化剪枝] 完成：{name}{_cuda_memory_note(include_peak=True, device_value=device_value)}', flush=True)
    return result


def _warn_structured_memory(config: Mapping[str, Any]) -> None:
    if str(config.get('model', {}).get('device', 'auto')).strip().lower() == 'cpu':
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        device = _cuda_device(torch, config.get('model', {}).get('device', 'auto'))
        if device.type != 'cuda':
            return
        free, total = torch.cuda.mem_get_info(device)
        gib = 1024 ** 3
        if total < 8 * gib or free < 6 * gib:
            print(f'[结构化剪枝] 显存风险：{device} 当前空闲 {free / gib:.2f} GiB，'
                  f'总量 {total / gib:.2f} GiB。评估、测速、依赖图追踪和微调均可能出现显存峰值；'
                  '该提示不是精确容量预测。', flush=True)
    except (ImportError, RuntimeError, AssertionError):
        return


def _resolved_dataset_root(data_path: Path, raw_root: Any) -> Path:
    root = Path(str(raw_root or ".")).expanduser()
    return (root if root.is_absolute() else data_path.parent / root).resolve()


def _link_or_copy(source: Path, target: Path, *, copy_files: bool = False) -> None:
    """Mirror a dataset directory into a run directory without editing input."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            # Existing Ultralytics caches must not be hardlinked into a run:
            # rebuilding them could otherwise truncate a user's cache inode.
            if child.is_file() and child.suffix.lower() in {'.cache', '.npy'}:
                continue
            _link_or_copy(child, target / child.name, copy_files=copy_files)
        return
    if copy_files:
        shutil.copy2(source, target)
        return
    try:
        # Hardlinks keep the mirror cheap while preserving the shadow path;
        # Ultralytics then writes its cache beside the hardlinked labels.
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _isolated_validation_data(data_path: Path, run_dir: Path, split: str) -> Path:
    """Point standard YOLO validation paths at run-local label-cache locations.

    Ultralytics stores ``labels.cache`` beside the first label directory. A
    run-local mirror keeps that generated file out of the user's dataset while
    retaining the original bytes through hardlinks (or a safe copy fallback).
    Non-standard layouts are left untouched because their label mapping cannot
    be inferred safely here.
    """

    try:
        import yaml
    except ImportError:
        return data_path
    try:
        raw = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return data_path
    if not isinstance(raw, dict):
        return data_path
    source = raw.get(split)
    if not isinstance(source, str) or not source.strip() or any(char in source for char in "[]"):
        return data_path
    dataset_root = _resolved_dataset_root(data_path, raw.get("path"))
    source_path = Path(source).expanduser()
    source_path = (source_path if source_path.is_absolute() else dataset_root / source_path).resolve()
    if not source_path.is_dir():
        return data_path
    try:
        relative_source = source_path.relative_to(dataset_root)
    except ValueError:
        return data_path
    relative_parts = relative_source.parts
    if not relative_parts:
        return data_path
    if relative_parts[0].lower() == "images":
        suffix = Path(*relative_parts[1:])
        original_labels = dataset_root / "labels" / suffix
    elif relative_parts[-1].lower() == "images":
        suffix = Path(*relative_parts[:-1])
        original_labels = source_path.parent / "labels"
    else:
        return data_path
    if not original_labels.is_dir():
        return data_path
    mirror_root = run_dir / "_input" / "dataset"
    mirror_images = mirror_root / "images" / suffix
    mirror_labels = mirror_root / "labels" / suffix
    if not mirror_images.exists():
        _link_or_copy(source_path, mirror_images, copy_files=True)
    if not mirror_labels.exists():
        _link_or_copy(original_labels, mirror_labels)
    raw["path"] = str(mirror_root)
    raw[split] = str(Path("images") / suffix)
    isolated = run_dir / "_input" / "data.yaml"
    isolated.parent.mkdir(parents=True, exist_ok=True)
    isolated.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return isolated


def detection_data(config: Mapping[str, Any]) -> Path | None:
    dataset = config.get('dataset', {})
    explicit = dataset.get('data')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'检测数据集 YAML 不存在：{path}')
        return path
    # Compatibility with the former train/val directory fields; never invent class names.
    candidates = set()
    for key in ('train', 'val'):
        value = dataset.get(key)
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path.suffix.lower() in {'.yaml', '.yml'} and path.is_file():
            candidates.add(path)
        else:
            for parent in (path, path.parent):
                for name in ('data.yaml', 'data.yml'):
                    candidate = parent / name
                    if candidate.is_file():
                        candidates.add(candidate)
    if len(candidates) > 1:
        raise ValueError('发现多份 data.yaml，请明确选择 dataset.data，避免使用错误的数据集。')
    return next(iter(candidates), None)


def _isolated_distillation_data(data_path: Path, run_dir: Path, *, copy_files: bool = False) -> Path:
    """Mirror train and val label trees so Fork cache writes stay run-local."""
    import yaml

    raw = yaml.safe_load(data_path.read_text(encoding='utf-8')) or {}
    if not isinstance(raw, dict):
        raise ValueError('检测蒸馏数据集 YAML 必须是对象。')
    root = _resolved_dataset_root(data_path, raw.get('path'))
    target_root = run_dir / '_input' / 'distillation_dataset'
    for split in ('train', 'val'):
        entry = raw.get(split)
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f'检测蒸馏 data.yaml 必须指定 {split} 图片目录。')
        images = Path(entry).expanduser()
        images = (images if images.is_absolute() else root / images).resolve()
        if not images.is_dir():
            raise ValueError(f'检测蒸馏 {split} 必须是现有图片目录：{images}')
        if images.parent.name.lower() == 'images':
            labels = images.parent.parent / 'labels' / images.name
        elif images.name.lower() == 'images':
            labels = images.parent / 'labels'
        else:
            raise ValueError(f'检测蒸馏仅支持标准 images/labels 数据集布局：{images}')
        if not labels.is_dir():
            raise FileNotFoundError(f'检测蒸馏找不到对应标签目录：{labels}')
        _link_or_copy(images, target_root / 'images' / split, copy_files=copy_files)
        _link_or_copy(labels, target_root / 'labels' / split)
        raw[split] = f'images/{split}'
    raw['path'] = str(target_root)
    target = run_dir / '_input' / 'distillation_data.yaml'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return target


def is_detection(config: Mapping[str, Any]) -> bool:
    model = config.get('model') or {}
    task = str(model.get('task') or '').strip().lower()
    if task in {'detect', 'classify'}:
        return task == 'detect'
    if model.get('adapter') == 'ultralytics':
        return True
    dataset = config.get('dataset') or {}
    if dataset.get('data'):
        return True
    for key in ('train', 'val'):
        value = dataset.get(key)
        if value:
            path = Path(value)
            if (path / 'images').is_dir() and (path / 'labels').is_dir():
                return True
    return False


def load_detector(weights: str | Path):
    from ultralytics import YOLO
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到检测模型权重：{path}')
    detector = YOLO(str(path))
    if detector.task != 'detect':
        raise ValueError(f'需要目标检测模型，当前权重任务为 {detector.task}')
    return detector


def _clean_detector_checkpoint(detector) -> None:
    """Remove stale EMA/model snapshots before serializing a transformed model."""
    checkpoint = getattr(detector, "ckpt", None)
    if isinstance(checkpoint, Mapping):
        detector.ckpt = {key: value for key, value in checkpoint.items()
                         if key not in {"ema", "optimizer", "model"}}


def _parameter_signature(model) -> tuple[int, tuple[tuple[str, tuple[int, ...]], ...]]:
    params = list(model.named_parameters())
    return sum(int(param.numel()) for _, param in params), tuple(
        (str(name), tuple(int(dim) for dim in param.shape)) for name, param in params
    )


def _distillation_feature_sizes(detector: Any, indices: list[int], imgsz: int) -> list[tuple[int, int]]:
    """Check actual neck feature sizes at a small stride-aligned resolution."""
    import torch

    network = detector.model
    captured: dict[int, tuple[int, int]] = {}
    handles = []
    original_training = network.training

    def capture(index: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError(f'检测蒸馏特征层 {index} 未输出四维特征张量。')
            captured[index] = tuple(int(size) for size in output.shape[-2:])
        return hook

    try:
        for index in indices[:-1]:
            handles.append(network.model[index].register_forward_hook(capture(index)))
        network.eval()
        size = max(32, min(imgsz, 256) // 32 * 32)
        device = next(network.parameters()).device
        channels = int(network.yaml.get('channels', 3))
        with torch.no_grad():
            network(torch.zeros(1, channels, size, size, device=device))
        return [captured[index] for index in indices[:-1]]
    finally:
        for handle in handles:
            handle.remove()
        network.train(original_training)


def _check_distillation_architecture(teacher: Any, student: Any, distillation_model: Any, imgsz: int) -> None:
    teacher_head = teacher.model.model[-1]
    student_head = student.model.model[-1]
    teacher_nc = int(teacher_head.nc)
    student_nc = int(student_head.nc)
    if teacher_nc != student_nc:
        raise ValueError(f'检测蒸馏教师/学生类别数不一致：{teacher_nc} != {student_nc}')
    teacher_layers = distillation_model.get_distill_layers(teacher.model)
    student_layers = distillation_model.get_distill_layers(student.model)
    if teacher_layers != student_layers:
        raise ValueError(f'当前 Fork 要求教师与学生的蒸馏特征层索引一致：教师={teacher_layers}，学生={student_layers}；请选用兼容架构。')
    teacher_sizes = _distillation_feature_sizes(teacher, teacher_layers, imgsz)
    student_sizes = _distillation_feature_sizes(student, student_layers, imgsz)
    if teacher_sizes != student_sizes:
        raise ValueError(f'检测蒸馏教师/学生特征空间尺寸不一致：教师={teacher_sizes}，学生={student_sizes}')


def _distillation_class_names(value: Any, owner: str) -> dict[int, str]:
    if isinstance(value, list):
        names = dict(enumerate(value))
    elif isinstance(value, Mapping):
        names = {}
        for key, name in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)) or not str(key).isdigit():
                raise ValueError(f'{owner} names 的类别 ID 必须从 0 连续编号。')
            index = int(key)
            if index in names:
                raise ValueError(f'{owner} names 包含重复类别 ID。')
            names[index] = name
    else:
        raise ValueError(f'{owner} 必须提供 names 类别名称及顺序。')
    if not names or set(names) != set(range(len(names))):
        raise ValueError(f'{owner} names 的类别 ID 必须从 0 连续编号。')
    if any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError(f'{owner} names 必须包含非空类别名称。')
    names = {index: names[index].strip() for index in range(len(names))}
    if len(set(names.values())) != len(names):
        raise ValueError(f'{owner} names 包含重复类别名称。')
    return names


def _prepare_distillation_student(teacher: Any, student: Any, data: Path, seed: int) -> dict[str, Any]:
    """Validate teacher semantics, then adapt a generic student's actual output layers."""
    import torch
    import yaml

    raw = yaml.safe_load(data.read_text(encoding='utf-8'))
    if not isinstance(raw, Mapping):
        raise ValueError('检测蒸馏数据集 YAML 必须是对象。')
    names = _distillation_class_names(raw.get('names'), '检测蒸馏数据集')
    nc = len(names)
    if 'nc' in raw and (isinstance(raw['nc'], bool) or raw['nc'] != nc):
        raise ValueError('检测蒸馏数据集 nc 与 names 类别数不一致。')
    teacher_nc = int(teacher.model.model[-1].nc)
    if teacher_nc != nc:
        raise ValueError(f'检测蒸馏教师与数据集类别数不一致：教师={teacher_nc}，数据集={nc}；请选择与教师匹配的 dataset.data。')
    teacher_names = _distillation_class_names(getattr(teacher.model, 'names', None), '检测蒸馏教师')
    if teacher_names != names:
        raise ValueError('检测蒸馏教师与数据集类别名称或顺序不一致；请核对 dataset.data 的 names，不能仅修改类别数量。')

    source = student.model
    source_nc = int(source.model[-1].nc)
    source_names = _distillation_class_names(getattr(source, 'names', None), '检测蒸馏学生')
    if len(source_names) != source_nc:
        raise ValueError('检测蒸馏学生 names 与实际检测头类别数不一致，无法安全迁移分类输出。')
    metadata = {'adapted': source_nc != nc or source_names != names,
                'source_nc': source_nc, 'target_nc': nc, 'names': names}
    if not metadata['adapted']:
        return metadata
    source_yaml = getattr(source, 'yaml', None)
    if not isinstance(source_yaml, Mapping):
        raise ValueError('学生模型缺少可重建的 model.yaml，无法自动适配检测头。')
    # Match the native trainer's model constructor. Preserve the caller's RNG
    # and seed the new head so the saved initialization is reproducible.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        target = type(source)(copy.deepcopy(dict(source_yaml)),
                              ch=int(raw.get('channels', 3)), nc=nc, verbose=False)
    target.names = dict(names)
    source_state = source.state_dict()
    target_state = target.state_dict()
    # Classification logits encode class order. Transfer matching class rows
    # explicitly, and leave new classes initialized even when nc is unchanged.
    class_keys = set()
    head_index = len(target.model) - 1
    for attr in ('cv3', 'one2one_cv3'):
        for index, branch in enumerate(getattr(target.model[-1], attr, ())):
            if getattr(branch[-1], 'out_channels', None) == nc:
                class_keys.update(f'model.{head_index}.{attr}.{index}.{len(branch) - 1}.{p}'
                                  for p in ('weight', 'bias'))
    transfer = {key: value for key, value in source_state.items()
                if key in target_state and key not in class_keys and value.shape == target_state[key].shape}
    # Reject pruned/custom backbones that cannot be reconstructed faithfully.
    head_prefix = f'model.{head_index}.'
    if any(key not in transfer for key in target_state if not key.startswith(head_prefix)):
        raise ValueError('学生主干结构与 model.yaml 不一致，无法安全重建；请使用架构完整的学生初始化权重。')
    target.load_state_dict(transfer, strict=False)
    source_lookup = {name: index for index, name in source_names.items()}
    with torch.no_grad():
        for key in class_keys & source_state.keys():
            src, dst = source_state[key], target_state[key]
            if src.shape[1:] == dst.shape[1:]:
                for index, name in names.items():
                    if name in source_lookup:
                        dst[index].copy_(src[source_lookup[name]])
    if hasattr(source, 'args'):
        target.args = copy.deepcopy(source.args)
    student.model = target
    _clean_detector_checkpoint(student)
    metadata['transferred_tensor_count'] = len(transfer)
    print(f'[检测蒸馏] 已适配学生检测头：{source_nc} → {nc} 类；复用兼容权重，无需预先训练学生。', flush=True)
    return metadata


def _unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法为检测模型产物生成唯一文件名：{path}")


def evaluate_detector(detector, config: Mapping[str, Any], run_dir: Path, name: str) -> dict[str, Any]:
    data = detection_data(config)
    if data is None:
        raise ValueError('检测压缩评估需要 dataset.data（YOLO data.yaml，包含 names 和 val）。')
    evaluation = config.get('evaluation', {})
    device = config.get('model', {}).get('device', 'auto')
    device = None if device == 'auto' else str(device).replace('cuda:', '')
    # ``val`` may fuse the model in place. Callers that need to mutate the
    # checkpoint after validation reload it before doing so.
    split = str(evaluation.get('split', 'val'))
    validation_data = _isolated_validation_data(data, run_dir, split)
    metrics = detector.val(
        data=str(validation_data), split=split,
        imgsz=config.get('dataset', {}).get('input_size', 640),
        batch=int(evaluation.get('batch_size', 1)), device=device,
        workers=0, project=str(run_dir), name=name, exist_ok=True,
        plots=True, save_json=False, half=False,
        conf=float(evaluation.get('conf', 0.001)), iou=float(evaluation.get('iou', 0.7)),
        fraction=float(evaluation.get('fraction', 1.0)),
    )
    box = getattr(metrics, 'box', None)
    if box is None:
        raise RuntimeError('Ultralytics 验证没有返回检测框指标。')

    def metric(name: str) -> float:
        value = getattr(box, name, None)
        if value is None:
            raise RuntimeError(f'Ultralytics 验证缺少检测指标：{name}')
        return float(value)

    return {'evaluation_map50': metric('map50'),
            'evaluation_map50_95': metric('map'),
            'evaluation_precision': metric('mp'),
            'evaluation_recall': metric('mr')}


def _image_shape(size: Any) -> tuple[int, int]:
    if isinstance(size, (list, tuple)):
        return int(size[0]), int(size[-1])
    return int(size), int(size)


def _output_signature(forward, model):
    import torch
    from torch.utils._pytree import tree_flatten

    with torch.inference_mode():
        leaves, structure = tree_flatten(forward(model.eval()))
    return structure, tuple(tuple(leaf.shape) if isinstance(leaf, torch.Tensor) else type(leaf) for leaf in leaves)


def benchmark_detector(detector, config: Mapping[str, Any], *, adapter: DetectionAdapter | None = None) -> dict[str, Any]:
    """Measure model forward latency with fixed input and warmup settings."""

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("检测测速需要 PyTorch") from exc
    compression = config.get('compression', {})
    section = compression.get('structured', {}) if isinstance(compression, Mapping) else {}
    section = section if isinstance(section, Mapping) else {}
    device_value = config.get('model', {}).get('device', 'auto')
    device = resolve_device(device_value)
    height, width = _image_shape(config.get('dataset', {}).get('input_size', 640))
    warmup = int(section.get('benchmark_warmup', 5))
    repeats = int(section.get('benchmark_repeats', 20))
    if warmup < 0 or repeats < 1:
        raise ValueError('结构化测速参数必须满足 warmup >= 0 且 repeats >= 1')
    model = detector.model.to(device).eval()
    if adapter is None:
        inputs = torch.zeros(1, 3, height, width, device=device)
        forward = lambda current: current(inputs)
    else:
        forward = adapter.make_forward((height, width), model)

    def sync() -> None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            forward(model)
        sync()
        durations: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            forward(model)
            sync()
            durations.append((time.perf_counter() - start) * 1000.0)
    p95_index = max(0, min(len(durations) - 1, math.ceil(len(durations) * 0.95) - 1))
    return {
        'benchmark_device': str(device),
        'benchmark_input_size': [height, width],
        'benchmark_warmup': warmup,
        'benchmark_repeats': repeats,
        'benchmark_inference_ms_median': statistics.median(durations),
        'benchmark_inference_ms_p95': sorted(durations)[p95_index],
    }


def _paired_detector_benchmark(original, candidate, config: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    section = config["compression"]["structured"]
    device = resolve_device(config["model"].get("device", "auto"))
    size = config["dataset"].get("input_size", 640)
    shape = (int(size[0]), int(size[-1])) if isinstance(size, (list, tuple)) else (int(size), int(size))
    warmup = int(section.get("benchmark_warmup", 5))
    repeats = int(section.get("benchmark_repeats", 20))
    if warmup < 0 or repeats < 1:
        raise ValueError("速度搜索需要非负预热次数和正数计时次数。")
    inputs = torch.zeros((1, 3, *shape), device=device)
    models = [copy.deepcopy(item.model).float().eval().fuse(verbose=False).to(device) for item in (original, candidate)]
    durations = [[], []]

    def run(index):
        models[index](inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            run(0)
            run(1)
        for repetition in range(repeats):
            for index in (repetition % 2, 1 - repetition % 2):
                started = time.perf_counter()
                run(index)
                durations[index].append((time.perf_counter() - started) * 1000)
    baseline, reduced = durations
    p95_index = math.ceil(repeats * 0.95) - 1
    return {"paired_gain": statistics.median((before - after) / before for before, after in zip(baseline, reduced)),
            "baseline_p50": statistics.median(baseline), "candidate_p50": statistics.median(reduced),
            "baseline_p95": sorted(baseline)[p95_index], "candidate_p95": sorted(reduced)[p95_index],
            "repeats": repeats, "device": str(device)}


def run_detection_operation(module_id, config, state, run_dir, run_id):
    """Use the same configured detector backend before and after each transform."""
    adapter = DetectionAdapter.from_config(config)
    if module_id == 'distillation.detection':
        if adapter is not None:
            raise ValueError('检测蒸馏使用原生 Ultralytics Fork，请将 model.adapter 设为 ultralytics。')
        return _run_detection_distillation(config, state, run_dir, run_id)
    if module_id in {'compression.quantize.dynamic_int8', 'distillation.classification'}:
        raise ValueError('该模块为分类流程；YOLO 检测当前支持基线评估、非结构化剪枝、结构化缩放和导出。')
    custom = adapter if adapter is not None and not adapter.yolo else None
    prefix = 'detector' if custom is not None else 'yolo'
    load = adapter.load if adapter is not None else load_detector

    def evaluate(detector, name):
        if custom is not None:
            return custom.evaluate(detector.model, run_dir, name)
        return evaluate_detector(detector, config, run_dir, name)

    def save(detector, target):
        if adapter is not None:
            adapter.save(detector, target)
        else:
            detector.save(str(target))

    explicit = config.get('model', {}).get('weights')
    head, resolved_weights = state.current(explicit)
    if not resolved_weights:
        raise ValueError('请先选择检测模型权重。' if adapter is not None else '请先选择 YOLO 模型权重。')
    weights = Path(resolved_weights)
    input_weights = weights
    if module_id == 'compression.quantize.onnx' and (custom is not None or weights.suffix.lower() != '.pt'):
        raise ValueError('ONNX 量化需要原生或通过 load_detector 接入的 YOLO .pt 检测模型。')
    exporting_onnx = module_id == 'artifact.export' and weights.suffix.lower() == '.onnx'
    if exporting_onnx:
        metadata = {}
    else:
        detector = load(weights)
        metadata = parameter_metrics(detector.model)
    if module_id == 'baseline.evaluate':
        metadata.update(evaluate(detector, 'baseline_validation'))
    elif module_id == 'compression.prune.unstructured':
        # Always compare on the same validation split, even if baseline was not selected.
        before = evaluate(detector, 'before_unstructured_pruning')
        # Validation may fuse layers; always prune a fresh model loaded from
        # the original checkpoint and evaluate the serialized result later.
        detector = load(weights)
        detector.model, pruning = prune_unstructured(detector.model.float(), config.get('compression', {}))
        metadata.update(pruning)
        target = _unique_target(run_dir / 'artifacts' / 'unstructured_pruning' / f'{prefix}-unstructured-l1.pt')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Avoid exporting an old EMA that would hide the pruned weights on reload.
        _clean_detector_checkpoint(detector)
        save(detector, target)
        reloaded = load(target)
        metadata.update(evaluate(reloaded, 'after_unstructured_pruning'))
        metadata['before_unstructured_pruning'] = before
        metadata['map50_95_delta'] = metadata['evaluation_map50_95'] - before['evaluation_map50_95']
        metadata['note'] = '非结构化剪枝减少非零权重，不保证 PT 文件更小或普通推理更快。'
        weights = target
    elif module_id == 'compression.prune.structured':
        device_value = config.get('model', {}).get('device', 'auto')
        def stage(name: str, action: Callable[[], _T]) -> _T:
            return _structured_stage(name, action, device_value=device_value)

        structured_section = config.get('compression', {}).get('structured', {})
        structured_section = structured_section if isinstance(structured_section, Mapping) else {}
        structured_method = normalize_structured_method(structured_section.get('method', 'scale'))
        epochs = int(structured_section.get('finetune_epochs', 80))
        if custom is not None and structured_method not in custom.GENERIC_STRUCTURED_METHODS:
            raise ValueError(f'结构化剪枝方法 {structured_method} 需要 YOLO 兼容 adapter（load_detector）。')
        if epochs < 0:
            raise ValueError('结构化微调轮数不能小于 0。')
        before = stage('剪枝前评估', lambda: evaluate(detector, 'before_structured_pruning'))
        before_speed = stage('剪枝前测速', lambda: benchmark_detector(detector, config, adapter=custom))
        source_file_size = artifact_info(weights)['size_bytes']
        del detector
        detector = stage('重新加载待剪枝模型', lambda: load(weights))
        data = detection_data(config) if custom is None else None
        if custom is None and data is None:
            raise ValueError('结构化检测压缩需要 dataset.data（YOLO data.yaml）。')
        structured_dir = run_dir / 'artifacts' / 'structured_pruning'
        structured_dir.mkdir(parents=True, exist_ok=True)
        calibrate = None
        if custom is not None and structured_method in {'taylor', 'fisher'}:
            calibrate = custom.calibrate
        if custom is None and structured_method in {'taylor', 'fisher', 'latency_search', 'bn_gamma'}:
            data = _isolated_distillation_data(data, run_dir, copy_files=True)
        if custom is None and structured_method in {'taylor', 'fisher', 'latency_search'}:
            from ..modules.gradient_calibration import detection_calibrator

            calibrate, samples = detection_calibrator(
                data, int(structured_section.get('pruning_input_size', 640)),
                int(structured_section.get('calibration_images', 24)),
            )
            metadata['calibration_images'] = samples
        if structured_method == 'bn_gamma':
            from ..modules.dependency_pruning import _protected_stages, _stage_candidates

            epochs = int(structured_section.get('bn_sparse_epochs', 4))
            coefficient = float(structured_section.get('bn_sparse_coefficient', 0.01))
            protected = _protected_stages(structured_section)
            stages = [index for index, _ in _stage_candidates(detector.model)
                      if index not in protected
                      and getattr(getattr(detector.model.model[index], 'bn', None), 'weight', None) is not None]
            if epochs < 1 or not math.isfinite(coefficient) or coefficient <= 0 or not stages:
                raise ValueError('BN-γ 需要可训练的 BN 层、正的稀疏训练轮数和正则系数。')
            strengths = {index: 0.25 + 0.75 * rank / max(1, len(stages) - 1) for rank, index in enumerate(stages)}
            sparse_config = dict(config, compression=dict(config['compression'], structured=dict(
                structured_section, finetune_epochs=epochs, optimizer='AdamW',
                learning_rate=float(structured_section.get('bn_sparse_learning_rate', 0.0004)),
                train_input_size=int(structured_section.get('bn_sparse_input_size', 640)), warmup_epochs=3,
            )))
            detector, sparse_details = stage('BN-γ 稀疏训练', lambda: finetune_structured_detector(
                detector, sparse_config, data, structured_dir / 'sparse_training',
                sparse_strengths=strengths, sparse_coefficient=coefficient,
                load_checkpoint=adapter.load if adapter is not None else None,
            ))
            metadata['bn_sparse_training'] = dict(sparse_details, coefficient=coefficient, layer_strengths=strengths)
        if structured_method in {'torch_pruning', 'taylor', 'fisher', 'lamp', 'bn_gamma'}:
            targets = forward = output_signature = None
            if custom is not None:
                targets = custom.pruning_targets(detector.model)
                shape = _image_shape(structured_section.get('pruning_input_size') or config['dataset'].get('input_size', 640))
                forward = custom.make_forward(shape, detector.model, pruning=True)
                output_signature = _output_signature(forward, detector.model)
            detector.model, structured = stage('依赖图追踪与通道剪枝',
                lambda: prune_detector_dependency_aware(detector.model, config, structured_dir,
                    method=structured_method, calibrate=calibrate, target_names=targets, example_forward=forward))
            if forward is not None and _output_signature(forward, detector.model) != output_signature:
                raise RuntimeError('剪枝改变了检测输出形状，请从 pruning_targets() 排除检测头及其耦合层。')
            if custom is not None:
                structured['pruning_input_size'] = list(shape)
        elif structured_method == 'scale':
            detector, structured = stage('构建缩放模型',
                lambda: build_structured_detector(detector, weights, config.get('compression', {}), structured_dir,
                    load_checkpoint=adapter.load if adapter is not None else None))
        else:
            from ..modules.latency_search import search_detector

            detector, structured = stage('速度搜索', lambda: search_detector(
                detector, config, structured_dir, before['evaluation_map50_95'], calibrate,
                evaluate,
                lambda original, candidate: _paired_detector_benchmark(original, candidate, config),
            ))
        if custom is None and structured_method == 'lamp' and epochs > 0:
            data = _isolated_distillation_data(data, run_dir, copy_files=True)
        if custom is not None:
            before_finetune = _parameter_signature(detector.model)
            detector, finetune = stage('微调', lambda: custom.finetune(detector, structured_dir, epochs))
            if _parameter_signature(detector.model) != before_finetune:
                raise RuntimeError('微调改变了模型参数形状，请在 adapter 中保留剪枝后的模型结构。')
        else:
            detector, finetune = stage('微调',
                lambda: finetune_structured_detector(detector, config, data, structured_dir,
                    load_checkpoint=adapter.load if adapter is not None else None))
        metadata.update(structured)
        metadata.update(finetune)
        target = _unique_target(structured_dir / f'{prefix}-structured.pt')
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_signature = _parameter_signature(detector.model)
        _clean_detector_checkpoint(detector)
        stage('保存剪枝模型', lambda: save(detector, target))
        del detector
        reloaded = stage('重新加载剪枝结果', lambda: load(target))
        actual_signature = _parameter_signature(reloaded.model)
        if actual_signature != expected_signature:
            raise RuntimeError(
                '结构化模型保存后重新加载的参数形状与剪枝结果不一致。'
                f'保存前参数量={expected_signature[0]}，加载后参数量={actual_signature[0]}'
            )
        if custom is not None:
            reloaded_forward = custom.make_forward(shape, reloaded.model, pruning=True)
            if _output_signature(reloaded_forward, reloaded.model) != output_signature:
                raise RuntimeError('剪枝结果重新加载后检测输出形状与原模型不一致。')
        metadata.update(stage('剪枝后评估',
            lambda: evaluate(reloaded, 'after_structured_pruning')))
        after_speed = stage('剪枝后测速', lambda: benchmark_detector(reloaded, config, adapter=custom))
        metadata.update({f'after_{key}': value for key, value in after_speed.items()})
        metadata.update({f'before_{key}': value for key, value in before_speed.items()})
        metadata['source_model_file_size_bytes'] = source_file_size
        target_file_size = artifact_info(target)['size_bytes']
        metadata['file_size_reduction'] = (1.0 - target_file_size / source_file_size) if source_file_size else None
        before_ms = before_speed.get('benchmark_inference_ms_median')
        after_ms = after_speed.get('benchmark_inference_ms_median')
        metadata['benchmark_speedup_ratio'] = (before_ms / after_ms) if before_ms and after_ms else None
        metadata['before_structured_pruning'] = before
        metadata['map50_95_delta'] = metadata['evaluation_map50_95'] - before['evaluation_map50_95']
        structured_section = config.get('compression', {}).get('structured', {})
        structured_section = structured_section if isinstance(structured_section, Mapping) else {}
        max_drop = float(structured_section.get('max_map50_95_drop', 0.05))
        quality_gate_enabled = bool(structured_section.get('quality_gate', True))
        quality_gate_passed = metadata['map50_95_delta'] >= -max_drop
        metadata['quality_gate'] = {
            'enabled': quality_gate_enabled,
            'max_map50_95_drop': max_drop,
            'passed': quality_gate_passed,
        }
        metadata['structured_method'] = structured_method
        if structured_method == 'latency_search':
            confirmed = stage('速度搜索复测', lambda: _paired_detector_benchmark(load(input_weights), reloaded, config))
            metadata['search_recheck'] = confirmed
            confirmation_path = structured_dir / 'latency-confirmation.json'
            passed = (confirmed['paired_gain'] >= float(structured_section.get('min_latency_gain', 0.05))
                      and confirmed['candidate_p95'] <= confirmed['baseline_p95'] and quality_gate_passed)
            write_json(confirmation_path, dict(confirmed, passed=passed,
                       map50_95_delta=metadata['map50_95_delta']))
            if not passed:
                raise ValueError(f'复测未同时通过精度和延迟门禁，建议保留原模型。复测记录：{confirmation_path}')
        metadata['note'] = (
            '依赖感知剪枝物理删除通道并同步调整依赖层；'
            if structured_method != 'scale'
            else '结构化缩放真实改变网络通道数；'
        ) + '速度必须按同一设备、输入尺寸和预热条件实测。'
        if quality_gate_enabled and not quality_gate_passed:
            raise ValueError(
                '结构化压缩未通过精度门禁：mAP50-95 下降 '
                f'{-metadata["map50_95_delta"]:.4f}，允许最大下降 {max_drop:.4f}。'
                '请增加微调轮数、换用更大的目标规模，或将 compression.structured.quality_gate 设为 false 仅保留实验产物。'
            )
        weights = target
    elif module_id == 'compression.quantize.onnx':
        from ultralytics import YOLO
        from ..modules.onnx_quantization import export_onnx_precision

        data = detection_data(config)
        if data is None:
            raise ValueError('ONNX 量化与复评需要检测数据集 YAML。')
        output = run_dir / 'artifacts' / 'quantization'
        settings = config['compression'].get('quantization', {})
        max_drop = float(settings.get('max_map50_95_drop', 0.05))
        if not 0 <= max_drop <= 1:
            raise ValueError('compression.quantization.max_map50_95_drop 必须在 [0, 1] 内。')
        candidate, reference, details = export_onnx_precision(weights, load, data, output, config)
        validation_config = {**config, 'model': {**config['model'], 'device': 'cpu'},
                             'evaluation': {**config.get('evaluation', {}), 'batch_size': 1}}
        before = evaluate_detector(YOLO(str(reference), task='detect'), validation_config, run_dir, 'onnx_fp32_validation')
        after = before if candidate == reference else evaluate_detector(
            YOLO(str(candidate), task='detect'), validation_config, run_dir, 'onnx_quantized_validation')
        delta = after['evaluation_map50_95'] - before['evaluation_map50_95']
        metadata.update(details, **after, fp32_validation=before, map50_95_delta=delta,
                        fp32_artifact=str(reference), model_file_size_bytes=candidate.stat().st_size,
                        quality_gate={'max_map50_95_drop': max_drop, 'passed': delta >= -max_drop})
        report_path = output / 'onnx-comparison.json'
        write_json(report_path, {**metadata, 'artifact': candidate.name, 'fp32_artifact': reference.name})
        if delta < -max_drop:
            raise ValueError(f'ONNX {details["precision"]} 的 mAP50–95 降幅 {-delta:.4f} 超过 {max_drop:.4f}，请查看 {report_path}。')
        weights = candidate
    elif module_id == 'artifact.export':
        import shutil
        source = Path(weights).resolve()
        target = config.get('export', {}).get('path') or run_dir / 'artifacts' / 'multi_processing' / f'{source.stem}-export{source.suffix}'
        target = Path(target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target != source:
            target = _unique_target(target)
            shutil.copy2(source, target)
        if exporting_onnx:
            from ultralytics import YOLO

            metadata.update(evaluate_detector(YOLO(str(target), task='detect'), config, run_dir, 'export_validation'))
        else:
            metadata.update(evaluate(load(target), 'export_validation'))
        export_info = {'source': artifact_info(source), 'target': target}
        weights = target
    elif module_id != 'model.parameters':
        raise ValueError(f'未支持的检测压缩模块：{module_id}')
    metadata['task'] = 'detect'
    if adapter is not None:
        metadata['adapter'] = config['model']['adapter']
    metadata['model_file_size_bytes'] = artifact_info(weights)['size_bytes']
    if not head:
        baseline_metadata = metadata
        if 'before_unstructured_pruning' in metadata:
            baseline_metadata = dict(metadata['before_unstructured_pruning'])
        elif 'before_structured_pruning' in metadata:
            baseline_metadata = dict(metadata['before_structured_pruning'])
        elif 'fp32_validation' in metadata:
            baseline_metadata = {**parameter_metrics(detector.model), 'model_file_size_bytes': input_weights.stat().st_size}
        baseline_metadata['task'] = 'detect'
        original = state.add_version(name=config.get('model', {}).get('name', prefix),
            artifact=input_weights, parent_version_id=None, run_id=run_id, method='baseline',
            metadata=baseline_metadata)
        head = original['id']
        state.advance(head)
    if module_id == 'compression.prune.unstructured':
        version = state.add_version(name=f'{prefix}-pruned', artifact=weights, parent_version_id=head,
            run_id=run_id, method='unstructured_l1', metadata=metadata)
        head = version['id']
        state.advance(head)
    elif module_id == 'compression.prune.structured':
        method = metadata['structured_method']
        registry_method = 'structured_width_scaling' if method == 'scale' else f'structured_{method}'
        version = state.add_version(name=f'{prefix}-structured', artifact=weights, parent_version_id=head,
            run_id=run_id, method=registry_method, metadata=metadata)
        head = version['id']
        state.advance(head)
    elif module_id == 'compression.quantize.onnx':
        version = state.add_version(name=weights.stem, artifact=weights, parent_version_id=head,
            run_id=run_id, method=metadata['method'], metadata=metadata)
        head = version['id']
        state.advance(head)
    elif module_id in {'baseline.evaluate', 'model.parameters'}:
        state.versions[head].setdefault('metadata', {}).update(metadata)
    if 'export_info' in locals():
        metadata['verification'] = {'reload': True, 'validation': 'evaluated', 'task': 'detect'}
    result = {'module': module_id, 'status': 'succeeded', 'version_id': head,
              'artifact': artifact_info(weights), 'metadata': metadata}
    return result


def _run_detection_distillation(config, state, run_dir, run_id):
    """Validate, train, reload and register a standalone distilled detector."""
    from ..modules.detection_distillation import (
        detection_distillation_settings, train_detection_distillation,
    )

    section = config.get('distillation') or {}
    for field in ('teacher_weights', 'student_weights'):
        if not section.get(field):
            raise ValueError(f'检测蒸馏需要明确指定 distillation.{field}。')
    teacher_path = Path(section['teacher_weights']).expanduser().resolve()
    student_path = Path(section['student_weights']).expanduser().resolve()
    settings = detection_distillation_settings(config)
    try:
        from ultralytics.nn.distill_model import DistillationModel
    except ImportError as exc:
        raise RuntimeError('当前 Ultralytics 缺少 Detect 原生蒸馏，请使用 ultralytics-cn Fork。') from exc
    data = detection_data(config)
    if data is None:
        raise ValueError('检测蒸馏必须指定 dataset.data（含 train、val 的 YOLO data.yaml）。')
    teacher = load_detector(teacher_path)
    student = load_detector(student_path)
    initialization = _prepare_distillation_student(teacher, student, data, settings['seed'])
    _check_distillation_architecture(teacher, student, DistillationModel, settings['imgsz'])
    output_dir = run_dir / 'artifacts' / 'distillation'
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = student_path
    if initialization['adapted']:
        initial_path = output_dir / 'student-initialized.pt'
        student.save(initial_path)
        # Evaluate and train from the same serialized initialization. Validation
        # may fuse its in-memory copy; it must not alter the training checkpoint.
        student = load_detector(initial_path)
    initialization['checkpoint'] = str(initial_path)
    training_data = _isolated_distillation_data(data, run_dir)
    # Use one dataset snapshot and image size for both comparisons. The input
    # mirror also prevents validation/training caches from touching user data.
    evaluation_config = dict(config)
    evaluation_config['dataset'] = {**config.get('dataset', {}),
                                    'data': str(training_data), 'input_size': settings['imgsz']}
    teacher_params = parameter_metrics(teacher.model)['parameter_count']
    initial_params = parameter_metrics(student.model)['parameter_count']
    initial_signature = _parameter_signature(student.model)
    before = evaluate_detector(student, evaluation_config, run_dir, 'before_detection_distillation')
    del teacher, student

    training = train_detection_distillation(
        student_weights=initial_path, teacher_weights=teacher_path,
        data=training_data, config=config, output_dir=output_dir,
    )
    final_path = Path(training['best_checkpoint'])
    reloaded = load_detector(final_path)
    if isinstance(reloaded.model, DistillationModel):
        raise RuntimeError('蒸馏检查点仍包含 DistillationModel，尚未生成独立学生模型。')
    final_params = parameter_metrics(reloaded.model)['parameter_count']
    if _parameter_signature(reloaded.model) != initial_signature:
        raise RuntimeError(f'蒸馏后的学生网络结构与初始化不一致：初始参数量={initial_params}，最终参数量={final_params}')
    after = evaluate_detector(reloaded, evaluation_config, run_dir, 'after_detection_distillation')
    metadata = {
        'task': 'detect', 'teacher_weights': str(teacher_path),
        'student_weights': str(student_path),
        'student_initialization': initialization,
        'teacher_parameter_count': teacher_params,
        'student_initial_parameter_count': initial_params,
        'student_final_parameter_count': final_params,
        'student_vs_teacher_parameter_reduction': (1 - final_params / teacher_params) if teacher_params else None,
        'before_distillation': before,
        **after,
        'map50_delta': after['evaluation_map50'] - before['evaluation_map50'],
        'map50_95_delta': after['evaluation_map50_95'] - before['evaluation_map50_95'],
        **training,
        'model_file_size_bytes': artifact_info(final_path)['size_bytes'],
    }
    # Parent must be the explicit student initialization, never an unrelated
    # current head (which may have been populated by an earlier operation).
    head, current_path = state.current(config.get('model', {}).get('weights'))
    if not head or Path(current_path).resolve() != initial_path:
        source = state.add_version(name='yolo-student-initial', artifact=initial_path,
            parent_version_id=None, run_id=run_id, method='student_initial', metadata=before)
        head = source['id']
    version = state.add_version(name='yolo-distilled', artifact=final_path,
        parent_version_id=head, run_id=run_id, method='detection_distillation', metadata=metadata)
    state.advance(version['id'])
    return {'module': 'distillation.detection', 'status': 'succeeded',
            'version_id': version['id'], 'artifact': artifact_info(final_path),
            'metadata': metadata, 'verification': {'reload': True, 'validation': 'evaluated', 'task': 'detect'}}
