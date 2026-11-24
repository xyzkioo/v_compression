"""Build the Windows installer from this project's desktop directory."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", type=Path, help="Inno Setup 6 ISCC.exe 的路径")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("请在 Windows 构建安装程序。")
    bundle = ROOT / "dist" / "v_compression"
    if not (bundle / "v_compression.exe").is_file() or not (bundle / "_internal").is_dir():
        raise SystemExit("请先运行 packaging/build_app.py 构建 Windows 目录包。")
    version = json.loads((ROOT / "component.json").read_text(encoding="utf-8"))["version"]
    if json.loads((bundle / "component.json").read_text(encoding="utf-8"))["version"] != version:
        raise SystemExit("目录包版本与当前源码不一致，请重新构建。")
    compiler = args.iscc or shutil.which("ISCC.exe")
    if compiler is None:
        for variable in ("LOCALAPPDATA", "ProgramFiles(x86)", "ProgramFiles"):
            base = os.environ.get(variable)
            if base:
                folder = Path(base) / "Programs" if variable == "LOCALAPPDATA" else Path(base)
                candidate = folder / "Inno Setup 6" / "ISCC.exe"
                if candidate.is_file():
                    compiler = candidate
                    break
    if compiler is None or not Path(compiler).is_file():
        raise SystemExit("未找到 Inno Setup 6，请用 --iscc 指定 ISCC.exe。")
    subprocess.run(
        [str(compiler), f"/DAppVersion={version}", f"/DSourceDir={bundle}", f"/DOutputDir={bundle.parent}",
         str(ROOT / "packaging" / "v_compression.iss")],
        cwd=ROOT,
        check=True,
    )
    destination = bundle.parent / f"v_compression-{version}-windows-x64-setup.exe"
    with destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    destination.with_suffix(".exe.sha256").write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
