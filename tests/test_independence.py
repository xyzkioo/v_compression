"""Verify private source ownership and protocol boundaries."""
import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from vtools_ui.standalone import COMPONENT, configure
from vtools_ui.webapp import core
from vtools_ui.webapp.catalog import APP, ROOT, TOOLS
from vtools_ui.process_environment import build_tool_environment
from vtools_runtime.component import COMPONENTS, normalize_request


class IndependenceTests(unittest.TestCase):
    def test_only_this_backend_is_registered(self):
        self.assertEqual(set(COMPONENTS), {COMPONENT})
        self.assertEqual(set(TOOLS), {APP['primary_tool'], 'environment'} | ({'dataset_quality'} if APP['primary_tool'] == 'diagnostics' else set()))
        with self.assertRaises(ValueError):
            configure('invalid-component')

    def test_all_sources_are_private_without_links_or_cross_imports(self):
        forbidden = {'model_visualization', 'model_diagnostics', 'speed_test', 'model_compression', 'run_tools'} - {COMPONENT}
        for path in ROOT.rglob('*'):
            if any(part in {'.venv', '.git', '__pycache__', 'node_modules', 'build', 'dist', 'inputs', 'user-data'} for part in path.relative_to(ROOT).parts):
                continue
            self.assertFalse(path.is_symlink(), str(path))
            if path.suffix != '.py':
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    self.assertNotIn(name.split('.')[0], forbidden, str(path))

    def test_child_pythonpath_cannot_load_sibling_source(self):
        env = build_tool_environment(os.sys.executable, ROOT, {'PYTHONPATH': '/tmp/other-vtools-source'})
        self.assertEqual(env['PYTHONPATH'], str(ROOT))

    def test_unknown_tool_and_protocol_fields_rejected(self):
        with self.assertRaises(ValueError):
            core.build_command({'tool_id': 'another-tool'})
        request = {'schema_version': 1, 'component': COMPONENT, 'config': 'a.yaml', 'run_dir': 'new-run', 'inputs': {'not_an_input': 'file'}, 'options': {}}
        with self.assertRaises(ValueError):
            normalize_request(COMPONENT, request, ROOT)

    def test_config_template_copy_keeps_inputs_relative_and_source_unchanged(self):
        source = ROOT / TOOLS[APP['primary_tool']]['config']
        original = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory, patch.object(core, 'STATE_DIR', Path(directory)):
            config = core.load_config(str(source))
            target = Path(config['path'])
            self.assertTrue(target.is_relative_to(Path(directory)))
            project_root = (target.parent / config['data']['project']['root']).resolve()
            self.assertEqual(project_root, ROOT / 'inputs')
        self.assertEqual(source.read_bytes(), original)

    def test_copying_custom_config_preserves_adapter_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            source = workspace / 'custom' / 'config.yaml'
            source.parent.mkdir()
            original = 'project:\n  root: ./files\nmodel:\n  adapter: detect.py\n'
            source.write_text(original, encoding='utf-8')
            with patch.object(core, 'ROOT', workspace), patch.object(core, 'STATE_DIR', workspace / 'state'):
                config = core.load_config(str(source))
            self.assertEqual(Path(config['data']['project']['root']), source.parent / 'files')
            self.assertEqual(config['data']['model']['adapter'], 'detect.py')
            self.assertEqual(source.read_text(encoding='utf-8'), original)

    def test_system_install_stores_config_and_inputs_in_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / "app"
            app.mkdir()
            (app / ".system-install").touch()
            runtime = app / "_internal"
            template = runtime / TOOLS[APP["primary_tool"]]["config"]
            template.parent.mkdir(parents=True)
            template.write_text("project:\n  root: ../../inputs\n", encoding="utf-8")
            data = root / "xdg"
            profile = data / "v_compression"
            with patch.dict(os.environ, {"XDG_DATA_HOME": str(data)}, clear=True), \
                    patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "executable", str(app / "v_compression")), \
                    patch.object(core, "ROOT", runtime), patch.object(core, "STATE_DIR", profile):
                configure()
                self.assertEqual(Path(os.environ["VTOOLS_USER_DATA"]), profile)
                config = core.load_config(str(template))
                self.assertTrue(Path(config["path"]).is_relative_to(profile))
                self.assertEqual((Path(config["path"]).parent / config["data"]["project"]["root"]).resolve(),
                                 profile / "inputs")
                self.assertTrue((profile / "inputs").is_dir())
                self.assertFalse((app / "user-data").exists())


if __name__ == '__main__':
    unittest.main()
