#!/usr/bin/env python3
"""Build this project's desktop folder; never builds a deb."""
import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    manifest = json.loads((ROOT / 'component.json').read_text())
    owner, slug = manifest['component'], manifest['slug']
    if not (ROOT / 'vtools_ui/webapp/frontend/dist/index.html').is_file():
        raise SystemExit('请先在本项目前端目录执行 npm run build。')
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
               '--name', slug, '--paths', str(ROOT), '--collect-all', 'webview', '--collect-all', 'PySide6',
               '--distpath', str(ROOT / 'dist'), '--workpath', str(ROOT / 'build'), '--specpath', str(ROOT / 'build'),
               '--add-data', str(ROOT / 'vtools_ui/webapp/frontend/dist') + os.pathsep + 'vtools_ui/webapp/frontend/dist']
    for package in (owner, 'vtools_runtime', 'env_test'):
        command += ['--add-data', str(ROOT / package) + os.pathsep + package]
    command += [str(ROOT / 'vtools_ui/__main__.py')]
    subprocess.run(command, cwd=ROOT, check=True)
    target = ROOT / 'dist' / slug
    (target / 'component.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for document in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(ROOT / document, target / document)
    shutil.copytree(ROOT / 'licenses', target / 'licenses', dirs_exist_ok=True)
    for name in ('inputs', 'user-data'):
        (target / name).mkdir(exist_ok=True)
    print(target)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
