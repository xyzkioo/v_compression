"""Export fixed-shape YOLO ONNX graphs and measure precision variants on CPU."""

from __future__ import annotations

import math
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path


ONNX_PRECISIONS = ("fp32", "fp16", "int8")


def _calibration_images(data: Path, limit: int) -> list[Path]:
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

    training = check_det_dataset(str(data), autodownload=False).get("train")
    if not training:
        raise ValueError("INT8 校准需要在 data.yaml 中配置 train。")
    images = set()
    for source in training if isinstance(training, list) else [training]:
        path = Path(source)
        if path.is_dir():
            candidates = (item for item in path.rglob("*") if item.is_file())
        elif path.is_file() and path.suffix.lower() == ".txt":
            candidates = (path.parent / line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        else:
            candidates = (path,)
        images.update(item.resolve() for item in candidates if item.suffix.lower().lstrip(".") in IMG_FORMATS)
    if not images:
        raise ValueError("训练集没有可用于 INT8 校准的图片。")
    return random.Random(42).sample(sorted(images), min(limit, len(images)))


def _image_tensor(path: Path, size: int):
    import numpy as np
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.patches import imread

    image = imread(str(path))
    if image is None:
        raise ValueError(f"无法读取 INT8 校准图片：{path}")
    fitted = LetterBox(new_shape=(size, size), auto=False, scaleup=False)(image=image)
    return np.ascontiguousarray(fitted[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def _static_int8(source: Path, target: Path, data: Path, size: int, input_name: str, settings: dict):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    limit = int(settings.get("calibration_images", 64))
    if limit < 1:
        raise ValueError("compression.quantization.calibration_images 必须为正整数。")
    images = _calibration_images(data, limit)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def get_next(self):
            path = next(self.paths, None)
            return None if path is None else {input_name: _image_tensor(path, size)}

        def rewind(self):
            self.paths = iter(images)

    excluded = settings.get("nodes_to_exclude", [])
    if not isinstance(excluded, list) or any(not isinstance(node, str) for node in excluded):
        raise ValueError("compression.quantization.nodes_to_exclude 必须是 ONNX 节点名列表。")
    per_channel = settings.get("per_channel", True)
    if not isinstance(per_channel, bool):
        raise ValueError("compression.quantization.per_channel 必须是布尔值。")
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        inferred = Path(directory) / "inferred.onnx"
        quant_pre_process(source, inferred, skip_optimization=True, skip_symbolic_shape=True)
        quantize_static(
            inferred, target, Reader(), quant_format=QuantFormat.QDQ, op_types_to_quantize=["Conv"],
            per_channel=per_channel, activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"], nodes_to_exclude=excluded,
        )
    return len(images)


def _graph_statistics(graph) -> dict[str, int]:
    import onnx

    initializers = {tensor.name: tensor.data_type for tensor in graph.initializer}
    dequantized = {node.output[0]: initializers.get(node.input[0])
                   for node in graph.node if node.op_type == "DequantizeLinear"}
    convolutions = [node for node in graph.node if node.op_type == "Conv"]
    return {
        "conv_count": len(convolutions),
        "int8_conv_count": sum(dequantized.get(node.input[1]) == onnx.TensorProto.INT8 for node in convolutions),
        "fp16_initializer_count": sum(dtype == onnx.TensorProto.FLOAT16 for dtype in initializers.values()),
    }


def _session(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _benchmark(session, sample, warmup: int, repeats: int) -> dict[str, float]:
    inputs = {session.get_inputs()[0].name: sample}
    for _ in range(warmup):
        session.run(None, inputs)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.run(None, inputs)
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"latency_p50_ms": statistics.median(elapsed),
            "latency_p95_ms": sorted(elapsed)[math.ceil(0.95 * len(elapsed)) - 1]}


def _preserve_custom_yolo_forward(detector) -> None:
    from ultralytics.nn.modules import C2f

    for layer in detector.model.modules():
        cls = type(layer)
        if (isinstance(layer, C2f) and cls.forward is not C2f.forward
                and cls.forward_split is C2f.forward_split):
            # Ultralytics uses forward_split for C2f exports. An inherited split can bypass a custom forward.
            layer.forward_split = layer.forward


def _export_fp32(detector, size: int):
    from torch.onnx.errors import SymbolicValueError

    options = dict(format="onnx", imgsz=size, batch=1, dynamic=False, half=False,
                   nms=False, simplify=False, opset=17, device="cpu")
    try:
        return detector.export(**options)
    except SymbolicValueError:
        from ultralytics.engine.exporter import Exporter
        import onnx
        import torch

        class DynamoExporter(Exporter):
            def export_onnx(self):
                target = self.file.with_suffix(".onnx")
                torch.onnx.export(
                    self.model, self.im, target, input_names=["images"], output_names=["output0"],
                    dynamo=True, opset_version=18, external_data=False,
                )
                graph = onnx.load(target)
                onnx.helper.set_model_props(graph, {key: str(value) for key, value in self.metadata.items()})
                onnx.save(graph, target)
                return target

        print("[ONNX] 旧导出器不支持当前图，使用 PyTorch Dynamo 导出", flush=True)
        try:
            return DynamoExporter(overrides={**options, "opset": 18})(model=detector.model)
        except ImportError as error:
            raise RuntimeError("当前模型需要 PyTorch Dynamo ONNX 导出，请安装 onnxscript。") from error


def export_onnx_precision(weights: Path, load_detector, data: Path, output: Path, config: dict):
    """Return the reloaded ONNX artifact and matching FP32 reference measurements."""
    try:
        import onnx
        import onnxruntime  # noqa: F401
    except ImportError as error:
        raise RuntimeError("检测量化需要在模型解释器安装 onnx 和 onnxruntime。") from error
    import numpy as np

    settings = config["compression"].get("quantization", {})
    precision = str(settings.get("precision", "int8")).lower()
    if precision not in ONNX_PRECISIONS:
        raise ValueError(f"ONNX 精度须为 {', '.join(ONNX_PRECISIONS)}。")
    size = config["dataset"].get("input_size", 640)
    if isinstance(size, (list, tuple)):
        if not size or len(set(size)) != 1:
            raise ValueError("ONNX 量化使用固定方形输入，请设置 dataset.input_size。")
        size = size[0]
    size = int(size)
    repeats = int(settings.get("benchmark_repeats", 10))
    warmup = int(settings.get("benchmark_warmup", 3))
    threads = int(settings.get("cpu_threads", 4))
    if size < 32 or repeats < 1 or warmup < 0 or threads < 1:
        raise ValueError("ONNX 量化要求输入尺寸至少 32、计时次数和线程数大于 0、预热次数非负。")
    output.mkdir(parents=True, exist_ok=True)
    reference = output / "model-fp32.onnx"
    print("[ONNX] 导出 FP32", flush=True)
    with tempfile.TemporaryDirectory(prefix="onnx-export-", dir=output) as directory:
        copy = Path(directory) / "source.pt"
        shutil.copy2(weights, copy)
        detector = load_detector(copy)
        stride = int(detector.model.stride.max().item())
        if size % stride:
            raise ValueError(f"ONNX 固定输入尺寸 {size} 必须能被模型步长 {stride} 整除。")
        _preserve_custom_yolo_forward(detector)
        exported = Path(_export_fp32(detector, size)).resolve()
        if exported.parent != copy.parent.resolve():
            raise RuntimeError("ONNX 导出器返回了本次临时目录之外的路径。")
        exported.replace(reference)

    original_graph = onnx.load(reference)
    onnx.checker.check_model(original_graph)
    initial_session = _session(reference, threads)
    input_info = initial_session.get_inputs()[0]
    sample = np.zeros((1, 3, size, size), dtype=np.float32)
    if len(initial_session.get_inputs()) != 1 or input_info.shape != list(sample.shape):
        raise ValueError(f"FP32 ONNX 不是预期的固定 NCHW 输入：{input_info.shape}")
    candidate = output / f"model-{precision}.onnx"
    calibration_images = 0
    if precision == "int8":
        print("[ONNX] 校准并生成 INT8 Q/DQ", flush=True)
        calibration_images = _static_int8(reference, candidate, data, size, input_info.name, settings)
    elif precision == "fp16":
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel

        print("[ONNX] 转换 FP16", flush=True)
        converted = convert_float_to_float16(onnx.load(reference), keep_io_types=True)
        OnnxModel(converted).topological_sort()
        onnx.save(converted, candidate)

    transformed_graph = onnx.load(candidate) if candidate != reference else original_graph
    if candidate != reference:
        del transformed_graph.metadata_props[:]
        transformed_graph.metadata_props.extend(original_graph.metadata_props)
        onnx.checker.check_model(transformed_graph)
        onnx.save(transformed_graph, candidate)
    statistics_by_node = _graph_statistics(transformed_graph.graph)
    if precision == "int8" and not statistics_by_node["int8_conv_count"]:
        raise RuntimeError("INT8 ONNX 未生成带 INT8 权重的卷积。")
    if precision == "fp16" and not statistics_by_node["fp16_initializer_count"]:
        raise RuntimeError("FP16 ONNX 未生成 FP16 权重。")
    candidate_session = _session(candidate, threads) if candidate != reference else initial_session
    reference_output = initial_session.run(None, {input_info.name: sample})
    candidate_output = candidate_session.run(None, {candidate_session.get_inputs()[0].name: sample})
    if ([item.name for item in initial_session.get_outputs()] != [item.name for item in candidate_session.get_outputs()]
            or [item.shape for item in reference_output] != [item.shape for item in candidate_output]
            or not all(np.isfinite(item).all() for item in (*reference_output, *candidate_output))):
        raise RuntimeError("ONNX 复载后的输出签名不一致或含非有限数值。")
    baseline_speed = _benchmark(initial_session, sample, warmup, repeats)
    candidate_speed = _benchmark(candidate_session, sample, warmup, repeats) if candidate != reference else baseline_speed
    return candidate, reference, {
        "method": f"onnx_{precision}", "precision": precision, "backend": "onnxruntime",
        "provider": "CPUExecutionProvider", "input_size": [size, size], "batch_size": 1,
        "calibration_images": calibration_images, "cpu_threads": threads,
        "benchmark_warmup": warmup, "benchmark_repeats": repeats,
        "fp32_benchmark": baseline_speed, "benchmark_latency_p50_ms": candidate_speed["latency_p50_ms"],
        "benchmark_latency_p95_ms": candidate_speed["latency_p95_ms"],
        "benchmark_speedup_ratio": baseline_speed["latency_p50_ms"] / candidate_speed["latency_p50_ms"],
        "fp32_file_size_bytes": reference.stat().st_size,
        "file_size_reduction": 1 - candidate.stat().st_size / reference.stat().st_size,
        **statistics_by_node,
    }
