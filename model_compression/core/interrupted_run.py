"""Recover the compression summary when the worker exits before writing it."""

from __future__ import annotations

import json
import re
import signal
from datetime import datetime, timezone
from pathlib import Path

from .run_manager import write_json


def write_interrupted_summary(run_dir: Path, log_path: Path, exit_code: int, started_at: str) -> None:
    summary_path = run_dir / "summary.json"
    if not run_dir.is_dir() or summary_path.exists():
        return

    config_path = run_dir / "effective_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    model = config.get("model", {})
    modules = config.get("modules", {})
    statuses: dict[str, str] = {}
    current_module = last_stage = None
    with log_path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            match = re.match(r"^\[([A-Za-z][A-Za-z0-9_.-]*)\]\s*(开始|完成|失败|跳过)", line)
            if match:
                module_id, status = match.groups()
                statuses[module_id] = {"开始": "running", "完成": "succeeded",
                                       "失败": "failed", "跳过": "skipped"}[status]
                current_module = module_id if status == "开始" else None
                last_stage = None
            if current_module == "compression.prune.structured":
                stage = re.match(r"^\[结构化剪枝\]\s*(?:开始|完成)：(.+?)\s*$", line)
                if stage:
                    last_stage = stage.group(1)
    if current_module:
        statuses[current_module] = "failed"

    signal_name = None
    if exit_code < 0:
        try:
            signal_name = signal.Signals(-exit_code).name
        except ValueError:
            signal_name = f"SIG{-exit_code}"
    reason = f"子进程未能正常完成，退出码 {exit_code}。"
    if signal_name:
        reason = f"子进程被 {signal_name} 终止（退出码 {exit_code}）。"
    summary = {
        "run_id": run_dir.name,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "model": {
            "name": model.get("name", "model"),
            "task": model.get("task", "classify"),
            "adapter": model.get("adapter", "torch"),
            "source_weights": model.get("weights"),
        },
        "executed_modules": [key for key, enabled in modules.items() if enabled],
        "errors": [reason],
        "module_status": statuses,
        "failure_details": {
            "exit_code": exit_code, "signal": signal_name, "last_module": current_module,
            "last_stage": last_stage, "reason": reason,
        },
    }
    if current_module == "compression.prune.structured":
        summary["structured_pruning"] = {
            "status": "failed", "module": current_module, "error": reason, "last_stage": last_stage,
        }
    write_json(summary_path, summary)
