"""Run repeatable YOLO deployment checks on real images and record measured stages."""

from __future__ import annotations

import argparse
import math
import platform
import random
import statistics
import time
from pathlib import Path

from model_compression.core.run_manager import write_json
from model_compression.modules.onnx_quantization import _session


def _summary(samples: list[float]) -> dict[str, float]:
    return {"p50_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1]}


def _images(directory: Path, count: int) -> list[Path]:
    from ultralytics.data.utils import IMG_FORMATS

    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower().lstrip(".") in IMG_FORMATS)
    if not paths:
        raise ValueError(f"没有找到测试图片：{directory}")
    return random.Random(42).sample(paths, min(count, len(paths)))


def _load_input(path: Path):
    from ultralytics.utils.patches import imread

    image = imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def _image_tensor(image, size: int):
    import numpy as np
    from ultralytics.data.augment import LetterBox

    padded = LetterBox(new_shape=(size, size), auto=False)(image=image)
    return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def _nms(output, conf: float, iou: float, end_to_end: bool):
    from ultralytics.utils.nms import non_max_suppression

    return non_max_suppression(output.clone(), conf, iou, end2end=end_to_end)[0]


def _profile(session, paths: list[Path], size: int, args, end_to_end: bool):
    import torch
    from ultralytics.utils.ops import scale_boxes

    samples = {name: [] for name in ("read_decode", "preprocess", "forward", "postprocess", "end_to_end")}
    detections, raw_predictions = [], []
    input_name = session.get_inputs()[0].name
    schedule = [paths[0]] * args.warmup + paths * args.repeats
    with torch.inference_mode():
        for index, path in enumerate(schedule):
            started = time.perf_counter()
            image = _load_input(path)
            decoded = time.perf_counter()
            batch = _image_tensor(image, size)
            preprocessed = time.perf_counter()
            raw = session.run(None, {input_name: batch})
            inferred = time.perf_counter()
            kept = _nms(torch.from_numpy(raw[0]), args.conf, args.iou, end_to_end)
            kept[:, :4] = scale_boxes((size, size), kept[:, :4], image.shape)
            boxes = kept.numpy()
            completed = time.perf_counter()
            if index < args.warmup:
                continue
            durations = (decoded - started, preprocessed - decoded, inferred - preprocessed,
                         completed - inferred, completed - started)
            for values, duration in zip(samples.values(), durations):
                values.append(duration * 1000)
            if len(detections) < len(paths):
                raw_predictions.append(raw)
                detections.append(boxes)
    timings = {name: _summary(values) for name, values in samples.items()}
    timed_frames = samples["end_to_end"]
    timings["end_to_end"]["fps"] = len(timed_frames) * 1000 / sum(timed_frames)
    return timings, raw_predictions, detections


def _compare(reference, candidate) -> dict:
    import numpy as np

    ref_timings, ref_raw, ref_boxes = reference
    cand_timings, cand_raw, cand_boxes = candidate
    raw_close = True
    max_difference = 0.0
    for expected, actual in zip(ref_raw, cand_raw, strict=True):
        for before, after in zip(expected, actual, strict=True):
            if before.shape != after.shape or not np.isfinite(after).all():
                raise ValueError("推理产物的输出形状变化或出现非有限数值。")
            max_difference = max(max_difference, float(np.abs(before - after).max(initial=0)))
            raw_close &= bool(np.allclose(before, after, atol=1e-4, rtol=1e-4))
    boxes_close = sum(int(before.shape == after.shape
                          and np.allclose(before, after, atol=0.1, rtol=1e-4))
                      for before, after in zip(ref_boxes, cand_boxes, strict=True))
    return {"raw_allclose": raw_close, "max_abs_output_difference": max_difference,
            "matching_detection_frames": boxes_close, "frames": len(ref_boxes),
            "end_to_end_p50_ratio": ref_timings["end_to_end"]["p50_ms"] / cand_timings["end_to_end"]["p50_ms"],
            "reference_boxes": sum(map(len, ref_boxes)), "candidate_boxes": sum(map(len, cand_boxes))}


def _load_labels(paths: list[Path], images: Path, labels: Path):
    import torch

    shapes, targets = [], []
    for path in paths:
        height, width = _load_input(path).shape[:2]
        shapes.append((height, width))
        label_path = labels / path.relative_to(images).with_suffix(".txt")
        boxes = []
        if label_path.is_file():
            for row in label_path.read_text(encoding="utf-8").splitlines():
                if not row.strip():
                    continue
                label, x, y, box_w, box_h = map(float, row.split())
                boxes.append(((x - box_w / 2) * width, (y - box_h / 2) * height,
                              (x + box_w / 2) * width, (y + box_h / 2) * height, label))
        targets.append(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5))
    if not any(len(target) for target in targets):
        raise ValueError("所选测试图片没有检测标注，无法统计漏检和误检。")
    return list(zip(shapes, targets, strict=True))


def _threshold_sweep(raw_predictions, labels, size: int, end_to_end: bool) -> list[dict]:
    import torch
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.ops import scale_boxes

    measurements = []
    with torch.inference_mode():
        for conf in (0.15, 0.25, 0.40):
            for iou in (0.45, 0.70):
                true_positives = predicted = actual = 0
                for raw, (shape, target) in zip(raw_predictions, labels, strict=True):
                    kept = _nms(torch.from_numpy(raw[0]), conf, iou, end_to_end)
                    kept[:, :4] = scale_boxes((size, size), kept[:, :4], shape)
                    predicted += len(kept)
                    actual += len(target)
                    if not len(kept) or not len(target):
                        continue
                    overlaps = box_iou(kept[:, :4], target[:, :4])
                    overlaps.masked_fill_(kept[:, 5:6] != target[:, 4], -1)
                    for prediction in kept[:, 4].argsort(descending=True).tolist():
                        match = int(overlaps[prediction].argmax())
                        if overlaps[prediction, match] >= 0.5:
                            true_positives += 1
                            overlaps[:, match] = -1
                precision = true_positives / predicted if predicted else 0
                recall = true_positives / actual
                measurements.append({"conf": conf, "nms_iou": iou, "match_iou": 0.5,
                                     "tp": true_positives, "fp": predicted - true_positives,
                                     "fn": actual - true_positives, "precision": precision, "recall": recall})
    return measurements


def _cuda_nms(raw_predictions, args, end_to_end: bool) -> dict:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "当前解释器无法使用 CUDA。"}
    if end_to_end:
        return {"status": "skipped", "reason": "参考模型已输出后处理结果。"}
    import torchvision  # noqa: F401 - use its CUDA NMS operator

    cpu_inputs = [torch.from_numpy(output[0]) for output in raw_predictions]
    gpu_inputs = [tensor.cuda() for tensor in cpu_inputs]
    durations = {"cpu": [], "cuda": []}
    matches = []
    torch.cuda.synchronize()
    with torch.inference_mode():
        for name, inputs in (("cpu", cpu_inputs), ("cuda", gpu_inputs)):
            for _ in range(args.warmup):
                _nms(inputs[0], args.conf, args.iou, False)
            if name == "cuda":
                torch.cuda.synchronize()
            for index in range(args.repeats):
                for tensor in inputs:
                    started = time.perf_counter()
                    kept = _nms(tensor, args.conf, args.iou, False)
                    if name == "cuda":
                        torch.cuda.synchronize()
                    durations[name].append((time.perf_counter() - started) * 1000)
                    if name == "cuda" and index == 0:
                        matches.append(kept.cpu().numpy())
        expected = [_nms(tensor, args.conf, args.iou, False).numpy() for tensor in cpu_inputs]
    matching = sum(int(before.shape == after.shape and np.allclose(before, after))
                   for before, after in zip(expected, matches, strict=True))
    boxes = sum(map(len, expected))
    status = "skipped" if boxes == 0 else "passed" if matching == len(expected) else "failed"
    return {"status": status, "device": torch.cuda.get_device_name(), "matching_frames": matching,
            "frames": len(expected), "detections": boxes,
            "scope": "resident tensor to NMS output with device synchronization",
            **{name: _summary(values) for name, values in durations.items()}}


def _cuda_preprocessing(paths: list[Path], size: int, args) -> dict:
    import torch
    from torch.nn.functional import interpolate, pad

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "当前解释器无法使用 CUDA。"}
    images = [_load_input(path) for path in paths]
    pinned = [torch.from_numpy(image).pin_memory() for image in images]

    def resize(image):
        height, width = image.shape[:2]
        ratio = min(size / height, size / width)
        scaled_h, scaled_w = round(height * ratio), round(width * ratio)
        padding_h, padding_w = size - scaled_h, size - scaled_w
        top, left = padding_h // 2, padding_w // 2
        rgb = image.permute(2, 0, 1).flip(0).float().unsqueeze(0)
        resized = interpolate(rgb, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False).round()
        return pad(resized, (left, padding_w - left, top, padding_h - top), value=114) / 255

    with torch.inference_mode():
        for _ in range(args.warmup):
            resize(pinned[0].cuda(non_blocking=True))
        torch.cuda.synchronize()
        copies, preprocessing, totals = [], [], []
        deviations = []
        for repeat in range(args.repeats):
            for image, source in zip(images, pinned, strict=True):
                start = time.perf_counter()
                on_device = source.cuda(non_blocking=True)
                torch.cuda.synchronize()
                copied = time.perf_counter()
                result = resize(on_device)
                torch.cuda.synchronize()
                finished = time.perf_counter()
                copies.append((copied - start) * 1000)
                preprocessing.append((finished - copied) * 1000)
                totals.append((finished - start) * 1000)
                if repeat == 0:
                    expected = torch.from_numpy(_image_tensor(image, size))
                    deviations.append((expected - result.cpu()).abs().max().item())
    tolerance = 2.01 / 255
    return {"status": "passed" if max(deviations) <= tolerance else "failed",
            "scope": "decoded BGR image to RGB NCHW letterbox tensor, excludes image decode",
            "device": torch.cuda.get_device_name(), "max_abs_normalized_difference": max(deviations),
            "allowed_difference": tolerance, "host_to_device": _summary(copies),
            "preprocess": _summary(preprocessing), "transfer_and_preprocess": _summary(totals)}


def _cuda_streams(weights: Path, paths: list[Path], size: int, args) -> dict:
    import torch
    from model_compression.core.detection import load_detector

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "当前解释器无法使用 CUDA。"}
    network = load_detector(weights).model.cuda().eval().half()
    tensors = [torch.from_numpy(_image_tensor(_load_input(path), size)).half().pin_memory() for path in paths]
    schedule = tensors * args.repeats
    copy_stream, inference_stream = torch.cuda.Stream(), torch.cuda.Stream()
    buffers = [torch.empty_like(tensors[0], device="cuda") for _ in range(2)]
    copied = [torch.cuda.Event() for _ in buffers]
    finished = [torch.cuda.Event() for _ in buffers]

    def forward(tensor):
        output = network(tensor)
        return output[0] if isinstance(output, tuple) else output

    with torch.inference_mode():
        for _ in range(args.warmup):
            buffers[0].copy_(tensors[0], non_blocking=True)
            forward(buffers[0])
        torch.cuda.synchronize()
        start = time.perf_counter()
        serial = []
        for tensor in schedule:
            buffers[0].copy_(tensor, non_blocking=True)
            serial.append(forward(buffers[0]))
        torch.cuda.synchronize()
        serial_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        pipelined = []
        for index, tensor in enumerate(schedule):
            slot = index % len(buffers)
            with torch.cuda.stream(copy_stream):
                if index >= len(buffers):
                    copy_stream.wait_event(finished[slot])
                buffers[slot].copy_(tensor, non_blocking=True)
                copied[slot].record()
            with torch.cuda.stream(inference_stream):
                inference_stream.wait_event(copied[slot])
                pipelined.append(forward(buffers[slot]))
                finished[slot].record()
        torch.cuda.synchronize()
        pipeline_ms = (time.perf_counter() - start) * 1000
        matches = sum(torch.allclose(before, after) for before, after in zip(serial, pipelined, strict=True))
    return {"status": "passed" if matches == len(schedule) else "failed", "device": torch.cuda.get_device_name(),
            "scope": "pinned preprocessed tensor upload and PyTorch FP16 forward",
            "matching_frames": matches, "frames": len(schedule), "serial_total_ms": serial_ms,
            "two_stream_total_ms": pipeline_ms, "throughput_ratio": serial_ms / pipeline_ms}


def _simplify(source: Path, run_dir: Path) -> tuple[Path, dict]:
    import onnx
    from onnxslim import slim

    original = onnx.load(source)
    simplified = slim(str(source))
    del simplified.metadata_props[:]
    simplified.metadata_props.extend(original.metadata_props)
    destination = run_dir / "model-slim.onnx"
    onnx.checker.check_model(simplified)
    onnx.save(simplified, destination)
    return destination, {"original_nodes": len(original.graph.node), "simplified_nodes": len(simplified.graph.node)}


def main() -> int:
    import onnxruntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="YOLO FP32 ONNX with one detection output")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="YOLO label directory for the optional threshold sweep")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--cuda-nms", action="store_true")
    parser.add_argument("--cuda-preprocess", action="store_true")
    parser.add_argument("--cuda-weights", type=Path, help="Trusted YOLO .pt for the optional CUDA stream probe")
    args = parser.parse_args()
    if min(args.sample_count, args.threads, args.repeats) < 1 or args.warmup < 0:
        parser.error("sample-count, threads and repeats must be positive, warmup cannot be negative")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("conf and iou must be in [0, 1]")
    variants = {}
    for value in args.variant:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in variants or name in {"reference", "slim"}:
            parser.error("--variant expects unique NAME=PATH")
        variants[name] = Path(path)

    session = _session(args.reference, args.threads)
    inputs = session.get_inputs()
    if len(inputs) != 1 or len(session.get_outputs()) != 1:
        parser.error("expected a single-input, single-output YOLO graph")
    shape = inputs[0].shape
    if len(shape) != 4 or shape[:2] != [1, 3] or type(shape[-1]) is not int or shape[2] != shape[3]:
        parser.error("expected fixed shape [1, 3, size, size]")
    metadata = session.get_modelmeta().custom_metadata_map
    end_to_end = metadata.get("end2end", "False").lower() == "true"
    paths = _images(args.images, args.sample_count)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    report = {"runtime": "ONNX Runtime CPUExecutionProvider", "onnxruntime_version": onnxruntime.__version__,
              "host_platform": platform.platform(), "input_shape": shape, "threads": args.threads,
              "conf": args.conf, "iou": args.iou, "warmup": args.warmup, "repeats": args.repeats,
              "images": [str(path) for path in paths], "variants": {}}
    if args.simplify:
        variants["slim"], report["simplification"] = _simplify(args.reference, args.run_dir)
    print(f"[部署验收] {len(paths)} 张图片，参考模型 {args.reference.name}", flush=True)
    reference = _profile(session, paths, shape[-1], args, end_to_end)
    report["reference"] = {"file": str(args.reference), "timings": reference[0]}
    labels = _load_labels(paths, args.images, args.labels) if args.labels is not None else None
    if labels is not None:
        report["reference"]["threshold_sweep"] = _threshold_sweep(reference[1], labels, shape[-1], end_to_end)
    for name, path in variants.items():
        print(f"[部署验收] {name}", flush=True)
        candidate = _session(path, args.threads)
        if ([value.shape for value in candidate.get_inputs()] != [shape]
                or [value.name for value in candidate.get_outputs()] != [value.name for value in session.get_outputs()]):
            raise ValueError(f"候选模型输入形状或输出名称不同：{path}")
        results = _profile(candidate, paths, shape[-1], args, end_to_end)
        report["variants"][name] = {"file": str(path), "timings": results[0],
                                    "comparison": _compare(reference, results)}
        if labels is not None:
            report["variants"][name]["threshold_sweep"] = _threshold_sweep(results[1], labels, shape[-1], end_to_end)
    if args.cuda_nms:
        report["cuda_nms"] = _cuda_nms(reference[1], args, end_to_end)
    if args.cuda_preprocess:
        report["cuda_preprocessing"] = _cuda_preprocessing(paths, shape[-1], args)
    if args.cuda_weights:
        report["cuda_streams"] = _cuda_streams(args.cuda_weights, paths, shape[-1], args)
    slim_comparison = report["variants"].get("slim", {}).get("comparison")
    slim_failed = slim_comparison is not None and (
        not slim_comparison["raw_allclose"] or slim_comparison["matching_detection_frames"] != len(paths))
    cuda_failed = any(report.get(name, {}).get("status") == "failed"
                      for name in ("cuda_nms", "cuda_preprocessing", "cuda_streams"))
    report["status"] = "failed" if slim_failed or cuda_failed else "passed"
    destination = write_json(args.run_dir / "deployment-comparison.json", report)
    print(f"[部署验收] {report['status']} · {destination}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
