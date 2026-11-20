"""Evaluate bounded YOLO width/depth candidates on the selected device."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from ..core.run_manager import write_json
from .dependency_pruning import _protected_stages, _stage_candidates, prune_detector_dependency_aware


def _residual_blocks(model: Any) -> list[str]:
    from ultralytics.nn.modules.block import Bottleneck

    return [
        f"model.{index}.{name}"
        for index, stage in enumerate(list(model.model)[:-1])
        for name, block in stage.named_modules()
        if name and type(block) is Bottleneck and block.add
        and block.cv1.conv.in_channels == block.cv2.conv.out_channels
    ]


def search_detector(
    detector: Any,
    config: Mapping[str, Any],
    run_dir: Path,
    baseline_map: float,
    calibrate: Callable[[Any, str], None],
    evaluate: Callable[[Any, str], Mapping[str, Any]],
    benchmark: Callable[[Any, Any], Mapping[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    import torch

    section = config["compression"]["structured"]
    limit = int(section.get("search_candidates", 8))
    minimum = float(section.get("min_latency_gain", 0.05))
    max_drop = float(section.get("max_map50_95_drop", 0.05))
    if not 1 <= limit <= 32 or not 0 < minimum < 1 or not 0 <= max_drop <= 1:
        raise ValueError("速度搜索要求 search_candidates 为 1–32、min_latency_gain 为 (0,1)、精度容许下降为 [0,1]。")

    model = detector.model
    protected = _protected_stages(section)
    stages = sorted(((index, layer.out_channels) for index, layer in _stage_candidates(model)
                     if index not in protected), key=lambda item: (-item[1], -item[0]))
    blocks = [path for path in reversed(_residual_blocks(model)) if int(path.split(".")[1]) not in protected]
    depth = blocks[:2]
    plans = []
    if depth:
        plans.append((None, tuple(depth)))
    if stages and depth:
        plans.append((stages[0][0], tuple(depth)))
    plans.extend((None, (block,)) for block in depth)
    plans.extend((index, ()) for index, _ in stages)
    report = run_dir / "latency-search.json"
    records = []

    def save_report(selected=None):
        write_json(report, {"candidates": records, "selected_for_recheck": selected, "min_latency_gain": minimum,
                            "max_map50_95_drop": max_drop})

    best = None
    best_key = None
    for number, (stage_index, removed_blocks) in enumerate(dict.fromkeys(plans[:limit]), start=1):
        name = f"search-{number}"
        record = {"name": name, "root_stage": stage_index, "removed_blocks": list(removed_blocks)}
        records.append(record)
        candidate = copy.deepcopy(detector)
        print(f"[速度搜索] {name}: stage={stage_index}, blocks={removed_blocks}", flush=True)
        try:
            if stage_index is not None:
                candidate.model, details = prune_detector_dependency_aware(
                    candidate.model, config, run_dir, method="taylor", calibrate=calibrate,
                    selected_stages={stage_index},
                )
                record.update(pruned_layers=details["pruned_layers"])
                candidate.model.cpu()
            for path in removed_blocks:
                candidate.model.set_submodule(path, torch.nn.Identity())
            record["parameter_count"] = sum(parameter.numel() for parameter in candidate.model.parameters())
            accuracy = float(evaluate(copy.deepcopy(candidate), name)["evaluation_map50_95"])
            record["map50_95"] = accuracy
            if not math.isfinite(accuracy) or accuracy < baseline_map - max_drop:
                record["status"] = "rejected_accuracy"
                continue
            record.update(benchmark(detector, candidate))
            if record["paired_gain"] < minimum or record["candidate_p95"] > record["baseline_p95"]:
                record["status"] = "rejected_latency"
                continue
            record["status"] = "qualified"
            key = (record["paired_gain"], accuracy)
            if best_key is None or key > best_key:
                best_key, best = key, (candidate, record)
        except (ValueError, RuntimeError, AssertionError, IndexError) as error:
            record.update(status="failed", error=str(error))
        finally:
            save_report(best[1]["name"] if best else None)
    if best is None:
        save_report()
        raise ValueError(f"没有候选同时通过精度和延迟门禁，建议保留原模型。候选记录：{report}")
    winner, record = best
    params = sum(parameter.numel() for parameter in model.parameters())
    return winner, {"method": "latency_search", "search_report_path": str(report),
                    "search_winner": record, "parameter_reduction": 1 - record["parameter_count"] / params}
