"""Release updater discovery, token boundaries and installer handoff."""

import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler

from vtools_ui.process_environment import build_tool_environment
from vtools_ui.standalone import UPDATE_TOKEN_ENV
from vtools_ui.webapp import updater


class UpdateTests(unittest.TestCase):
    def test_private_install_requires_token_and_portable_build_is_unsupported(self):
        error = HTTPError(updater.API_ROOT, 404, "Not Found", {}, io.BytesIO())
        with patch.dict(os.environ, {}, clear=True), patch.object(updater, "is_installed", return_value=True), \
                patch.object(updater, "urlopen", side_effect=error) as open_url:
            self.assertEqual(updater.check_update()["status"], "authentication_required")
            self.assertIsNone(open_url.call_args.args[0].get_header("Authorization"))
        with patch.object(updater, "is_installed", return_value=False):
            self.assertEqual(updater.check_update()["status"], "unsupported")

    def test_public_installer_download_resumes_without_token(self):
        version = "9.9.9"
        name = updater._installer_name(version)
        asset = {"name": name, "id": 123, "size": 20, "digest": "sha256:" + "a" * 64}
        releases = [{"tag_name": f"v{version}", "assets": [asset]}]
        with patch.dict(os.environ, {}, clear=True), patch.object(updater, "is_installed", return_value=True), \
                patch.object(updater, "urlopen", return_value=io.BytesIO(json.dumps(releases).encode())) as open_url:
            info = updater.check_update()
        self.assertEqual(info["status"], "available")
        self.assertIsNone(open_url.call_args.args[0].get_header("Authorization"))
        request = updater._download_request(info, "", 4)
        self.assertEqual(request.full_url, f"https://github.com/{updater.REPOSITORY}/releases/download/v{version}/{name}")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.get_header("Range"), "bytes=4-")

    def test_download_token_does_not_follow_cdn_redirect_or_model_environment(self):
        request = updater._request("releases/assets/123", "repo-token", accept="application/octet-stream", downloaded=4)
        self.assertEqual(request.get_header("Authorization"), "Bearer repo-token")
        redirect = HTTPRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "https://release-assets.githubusercontent.com/signed",
        )
        self.assertIsNone(redirect.get_header("Authorization"))
        self.assertEqual(redirect.get_header("Range"), "bytes=4-")
        env = build_tool_environment("python", Path.cwd(), {UPDATE_TOKEN_ENV: "repo-token"})
        self.assertNotIn(UPDATE_TOKEN_ENV, env)

    def test_selects_latest_signed_installation_asset_without_exposing_token(self):
        version = "9.9.9"
        asset = {"name": updater._installer_name(version), "id": 123, "size": 20, "digest": "sha256:" + "a" * 64}
        releases = [{"tag_name": "v9.9.10", "assets": []}, {"tag_name": f"v{version}", "assets": [asset]}]
        with patch.dict(os.environ, {UPDATE_TOKEN_ENV: "repo-token"}), \
                patch.object(updater, "is_installed", return_value=True), \
                patch.object(updater, "urlopen", return_value=io.BytesIO(json.dumps(releases).encode())) as open_url:
            info = updater.check_update()
            request = open_url.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer repo-token")
        self.assertEqual(info["status"], "available")
        self.assertEqual(info["asset_id"], 123)
        self.assertEqual(info["asset_digest"], "a" * 64)
        self.assertNotIn("repo-token", json.dumps(info))


if __name__ == "__main__":
    unittest.main()
