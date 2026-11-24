"""Wrap the Linux v_compression desktop bundle in a Debian installer."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "v-compression"


def main() -> int:
    if not sys.platform.startswith("linux"):
        raise SystemExit("请在 Linux 构建 .deb。")
    version = json.loads((ROOT / "component.json").read_text(encoding="utf-8"))["version"]
    bundle = ROOT / "dist" / "v_compression"
    launcher = bundle / "v_compression"
    if not launcher.is_file() or not (bundle / "_internal").is_dir():
        raise SystemExit("请先运行 packaging/build_app.py 构建 Linux 目录包。")
    if json.loads((bundle / "component.json").read_text(encoding="utf-8"))["version"] != version:
        raise SystemExit("目录包版本与当前源码不一致，请重新构建。")
    architecture = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
    destination = bundle.parent / f"{PACKAGE}_{version}_{architecture}.deb"
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".deb-") as directory:
        temporary = Path(directory)
        stage = temporary / "package"
        stage.mkdir()
        stage.chmod(0o755)
        app = stage / "opt" / "v_compression"
        shutil.copytree(bundle, app, symlinks=True, ignore=shutil.ignore_patterns("inputs", "user-data"))
        (app / ".system-install").touch()

        command = stage / "usr" / "bin" / "v_compression"
        command.parent.mkdir(parents=True)
        command.write_text('#!/bin/sh\nexec /opt/v_compression/v_compression "$@"\n', encoding="utf-8")
        command.chmod(0o755)
        desktop = stage / "usr" / "share" / "applications" / "v_compression.desktop"
        desktop.parent.mkdir(parents=True)
        desktop.write_text(
            "[Desktop Entry]\nType=Application\nName=v_compression\nComment=模型压缩工作台\n"
            "Exec=v_compression\nIcon=applications-science\nTerminal=false\nCategories=Science;\n",
            encoding="utf-8",
        )
        control = stage / "DEBIAN" / "control"
        control.parent.mkdir()
        control.write_text(
            f"Package: {PACKAGE}\nVersion: {version}\nArchitecture: {architecture}\n"
            "Maintainer: v_compression project\nSection: science\nPriority: optional\n"
            "Depends: libegl1, libgl1, libnss3, libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, "
            "libxcb-keysyms1, libxcb-shape0, libasound2 | libasound2t64, libpulse0, pkexec | policykit-1\n"
            "Description: Model compression workbench\n"
            " Desktop workbench for evaluation, quantization, pruning and distillation.\n",
            encoding="utf-8",
        )
        for root, directories, files in os.walk(stage):
            for name in directories + files:
                entry = Path(root) / name
                if not entry.is_symlink():
                    entry.chmod(0o755 if entry.is_dir() or entry.stat().st_mode & 0o100 else 0o644)
        built = temporary / destination.name
        subprocess.run(["dpkg-deb", "-z3", "--root-owner-group", "--build", str(stage), str(built)], check=True)
        built.replace(destination)
    with destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    destination.with_suffix(".deb.sha256").write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
