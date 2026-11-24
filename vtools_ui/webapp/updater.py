"""Find and securely install updates from this project's GitHub Releases."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vtools_ui import __version__
from vtools_ui.standalone import UPDATE_TOKEN_ENV, is_installed

from .core import STATE_DIR
from .update_download import MAX_PACKAGE_BYTES, download_package, ignore_progress


REPOSITORY = "xyzkioo/v_compression"
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}"
MAX_RELEASE_BYTES = 4 * 1024 * 1024


def _request(path: str, token: str, *, accept: str, downloaded: int = 0) -> Request:
    headers = {"Accept": accept, "User-Agent": f"v_compression/{__version__}",
               "X-GitHub-Api-Version": "2022-11-28"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = Request(f"{API_ROOT}/{path}", headers=headers)
    if token:
        # Keep the token on the API request when GitHub redirects to the CDN.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def _download_request(info: dict, token: str, downloaded: int) -> Request:
    if token:
        return _request(f"releases/assets/{info['asset_id']}", token,
                        accept="application/octet-stream", downloaded=downloaded)
    version, name = info["latest_version"], info["asset_name"]
    headers = {"User-Agent": f"v_compression/{__version__}"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    return Request(f"https://github.com/{REPOSITORY}/releases/download/v{version}/{name}", headers=headers)


def _architecture() -> str:
    machine = platform.machine().lower()
    return {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64"}.get(machine, machine)


def _version(text: str) -> tuple[int, ...]:
    fields = tuple(int(part) for part in text.split("."))
    return fields + (0,) * (4 - len(fields))


def _installer_name(version: str) -> str:
    if os.name == "nt":
        return f"v_compression-{version}-windows-x64-setup.exe"
    return f"v-compression_{version}_{_architecture()}.deb"


def _published_installers(releases: list) -> list[tuple[str, dict]]:
    installers = []
    for release in releases:
        if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
            continue
        tag = str(release.get("tag_name", ""))
        if not re.fullmatch(r"v\d+(?:\.\d+){1,3}", tag):
            continue
        version = tag[1:]
        asset = next((item for item in release.get("assets", [])
                      if isinstance(item, dict) and item.get("name") == _installer_name(version)), None)
        if asset:
            installers.append((version, asset))
    return sorted(installers, key=lambda row: _version(row[0]), reverse=True)


def check_update() -> dict:
    info = {"current_version": __version__}
    if not is_installed():
        return {**info, "status": "unsupported", "message": "目录版请通过替换程序文件更新，保留 user-data。"}
    token = os.environ.get(UPDATE_TOKEN_ENV, "").strip()
    request = _request("releases?per_page=30", token, accept="application/vnd.github+json")
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read(MAX_RELEASE_BYTES + 1)
    except HTTPError as error:
        error.close()
        if not token and error.code in {401, 404}:
            return {**info, "status": "authentication_required",
                    "message": f"GitHub 发布不可匿名读取。私有仓库可配置 {UPDATE_TOKEN_ENV} 并重启，或登录 GitHub Releases 手动下载。"}
        if token and error.code in {401, 403, 404}:
            raise RuntimeError(f"GitHub 拒绝读取发布信息（HTTP {error.code}），请确认令牌有仓库读取权限。") from error
        raise RuntimeError(f"检查更新失败，HTTP {error.code}") from error
    except (URLError, OSError) as error:
        raise RuntimeError(f"检查更新失败：{error}") from error
    if len(payload) > MAX_RELEASE_BYTES:
        raise RuntimeError("GitHub 发布信息过大")
    releases = json.loads(payload)
    if not isinstance(releases, list):
        raise RuntimeError("GitHub 发布信息格式错误")
    installers = _published_installers(releases)
    if not installers:
        return {**info, "status": "missing_asset", "message": "GitHub Releases 暂无适用于当前平台的安装包。"}
    version, asset = installers[0]
    if _version(version) <= _version(__version__):
        return {**info, "status": "current", "latest_version": version, "message": "当前已是最新版本。"}
    size, asset_id = asset.get("size"), asset.get("id")
    digest = str(asset.get("digest", "")).lower()
    if (type(asset_id) is not int or not asset_id > 0 or type(size) is not int
            or not 0 < size <= MAX_PACKAGE_BYTES or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)):
        return {**info, "status": "untrusted", "latest_version": version,
                "message": "最新安装包缺少完整的大小或 SHA-256 校验信息。"}
    return {**info, "status": "available", "latest_version": version, "message": f"发现更新 {version}",
            "asset_name": _installer_name(version), "asset_id": asset_id,
            "asset_size": size, "asset_digest": digest.removeprefix("sha256:")}


def _install_linux_package(package: Path, version: str, environment: dict[str, str]) -> str:
    fields = subprocess.check_output(
        ["dpkg-deb", "--field", str(package), "Package", "Version", "Architecture"],
        text=True, env=environment,
    ).splitlines()
    expected = {"Package: v-compression", f"Version: {version}", f"Architecture: {_architecture()}"}
    if set(fields) != expected:
        raise RuntimeError("安装包名称、版本或架构与发布信息不符")
    result = subprocess.run(["pkexec", "/usr/bin/apt-get", "install", "-y", str(package)],
                            capture_output=True, text=True, timeout=900, env=environment)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-800:]
        raise RuntimeError(f"安装更新失败：{detail or f'退出码 {result.returncode}'}")
    return f"已安装 {version}，请关闭并重新打开工作台。"


def install_update(report: Callable[[dict], None] = ignore_progress) -> dict[str, str]:
    report({"phase": "checking"})
    info = check_update()
    if info["status"] != "available":
        raise RuntimeError(str(info["message"]))
    if os.name != "nt" and shutil.which("pkexec") is None:
        raise RuntimeError("缺少 pkexec，请登录 GitHub Releases 下载 .deb 并通过 apt 安装。")
    token = os.environ.get(UPDATE_TOKEN_ENV, "").strip()

    def make_request(downloaded: int) -> Request:
        return _download_request(info, token, downloaded)

    package = download_package(STATE_DIR / "updates" / info["asset_name"], info["asset_size"],
                               info["asset_digest"], make_request, report)
    report({"phase": "installing"})
    environment = dict(os.environ)
    environment.pop(UPDATE_TOKEN_ENV, None)
    if os.name == "nt":
        subprocess.Popen([str(package)], cwd=str(package.parent), close_fds=True, env=environment)
        return {"message": "安装包已通过 SHA-256 校验，安装向导已启动。"}
    return {"message": _install_linux_package(package, info["latest_version"], environment)}
