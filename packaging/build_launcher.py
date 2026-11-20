#!/usr/bin/env python3
"""Build the small native Linux folder launcher, without creating a deb."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    compiler = shutil.which('gcc')
    if compiler is None:
        raise SystemExit('找不到 gcc；只在重新构建原生启动器时需要安装编译器。')
    manifest = json.loads((ROOT / 'component.json').read_text())
    output = ROOT / manifest['launcher']
    descriptor, temporary = tempfile.mkstemp(prefix='.native-launcher-', dir=ROOT)
    os.close(descriptor)
    target = Path(temporary)
    try:
        subprocess.run([compiler, '-O2', '-Wall', '-Wextra', '-Werror', '-no-pie', '-s',
                        str(ROOT / 'packaging/native_launcher.c'), '-o', str(target)], check=True)
        target.chmod(0o755)
        target.replace(output)
    finally:
        target.unlink(missing_ok=True)
    print(output)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
