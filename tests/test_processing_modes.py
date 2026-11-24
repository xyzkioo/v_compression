"""Rebuilt regression coverage for the independent compression tool's mode policy."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_compression.core.module_selection import COMPRESSION_MODES, compression_mode_modules
from model_compression.core.pruning_methods import STRUCTURED_METHODS, normalize_structured_method
from vtools_runtime.component import execute_request, normalize_request


class ProcessingModeTests(unittest.TestCase):
    def test_worker_exit_before_summary_preserves_failed_stage(self):
        def interrupted_worker(_component, _request, _snapshot, root, _result, stdout, _stderr, _stream_logs):
            run_dir = root / "artifacts"
            run_dir.mkdir()
            (run_dir / "effective_config.json").write_text(
                '{"model":{"task":"detect"},"modules":{"compression.prune.structured":true}}',
                encoding="utf-8",
            )
            stdout.write(("[compression.prune.structured] 开始\n"
                          "[结构化剪枝] 开始：剪枝前测速\n").encode("utf-8"))
            return -9

        with tempfile.TemporaryDirectory() as tmp, patch(
            "vtools_runtime.component._run_worker", side_effect=interrupted_worker,
        ):
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("project: {}\n", encoding="utf-8")
            run_dir = root / "run"
            result = execute_request("model_compression", {
                "schema_version": 1, "component": "model_compression",
                "config": str(config), "run_dir": str(run_dir),
            })
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["backend_exit_code"], -9)
            self.assertIn("artifacts/summary.json", [item["path"] for item in result["artifacts"]])
            summary = json.loads((run_dir / "artifacts" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module_status"], {"compression.prune.structured": "failed"})
            self.assertEqual(summary["failure_details"]["last_stage"], "剪枝前测速")

    def test_structured_methods_are_shared_by_catalog_and_cli(self):
        from model_compression.run_model_compression import build_parser
        from vtools_ui.webapp.catalog import public_catalog

        self.assertEqual(public_catalog()['compression']['structured_methods'], STRUCTURED_METHODS)
        for method in STRUCTURED_METHODS:
            with self.subTest(method=method):
                self.assertEqual(build_parser().parse_args(['--structured-method', method]).structured_method, method)
        self.assertEqual(normalize_structured_method('dependency'), 'torch_pruning')
        with self.assertRaises(ValueError):
            normalize_structured_method('unsupported')

    def test_each_declared_mode_resolves_for_its_supported_tasks(self):
        for mode, definition in COMPRESSION_MODES.items():
            for task in definition['tasks']:
                with self.subTest(mode=mode, task=task):
                    modules = compression_mode_modules(mode, task)
                    if definition['primary_module'] is None:
                        self.assertEqual(modules, ['model.parameters', 'baseline.evaluate'])
                    else:
                        self.assertIn(definition['primary_module'], modules)
                    self.assertEqual(len(modules), len(set(modules)))

    def test_unknown_mode_and_unsupported_task_raise(self):
        with self.assertRaises(ValueError):
            compression_mode_modules('unknown-mode', 'detect')
        for mode in COMPRESSION_MODES:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                compression_mode_modules(mode, 'unsupported-task')

    def test_returned_modes_do_not_mutate_global_configuration(self):
        for mode, definition in COMPRESSION_MODES.items():
            task = definition['tasks'][0]
            first = compression_mode_modules(mode, task)
            first.clear()
            self.assertEqual(definition['modules'], compression_mode_modules(mode, task))

    def test_single_method_selection_keeps_the_original_run_layout(self):
        request = {
            'schema_version': 1, 'component': 'model_compression', 'config': 'config.yaml',
            'run_dir': 'new-run', 'options': {'structured_methods': ['bn_gamma']},
        }
        options = normalize_request('model_compression', request, Path.cwd())['options']
        self.assertEqual(options['structured_method'], 'bn_gamma')
        self.assertNotIn('structured_methods', options)
        self.assertEqual(options['only'], COMPRESSION_MODES['structured']['modules'])

    def test_invalid_method_selections_raise(self):
        invalid_options = [
            {'structured_methods': []},
            {'structured_methods': ['unknown']},
            {'structured_methods': ['dependency', 'torch_pruning']},
            {'structured_methods': ['taylor'], 'structured_method': 'lamp'},
            {'structured_methods': ['lamp'], 'only': ['baseline.evaluate']},
        ]
        for options in invalid_options:
            request = {
                'schema_version': 1, 'component': 'model_compression', 'config': 'config.yaml',
                'run_dir': 'new-run', 'options': options,
            }
            with self.subTest(options=options), self.assertRaises(ValueError):
                normalize_request('model_compression', request, Path.cwd())

    def test_method_failure_preserves_independent_results_and_runs_the_next_method(self):
        with tempfile.TemporaryDirectory() as directory, patch('vtools_runtime.component._run_worker', side_effect=[2, 0]) as worker:
            root = Path(directory).resolve()
            config_dir = root / 'source'
            config_dir.mkdir()
            config = config_dir / 'config.yaml'
            config.write_text('project:\n  root: ./models\n', encoding='utf-8')
            run_dir = root / 'results'
            requests = {
                'schema_version': 1, 'component': 'model_compression', 'config': str(config),
                'run_dir': str(run_dir), 'inputs': {'weights': str(root / 'original.pt')},
                'options': {'structured_methods': ['bn_gamma', 'lamp']},
            }
            result = execute_request('model_compression', requests)
            artifacts = run_dir / 'artifacts'
            batch = json.loads((artifacts / 'structured-methods.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'failed')
            self.assertEqual([entry['status'] for entry in batch['methods']], ['failed', 'succeeded'])
            self.assertEqual(worker.call_count, 2)
            for entry in batch['methods']:
                saved_request = json.loads((artifacts / entry['method'] / 'request.json').read_text(encoding='utf-8'))
                self.assertEqual(saved_request['inputs'], requests['inputs'])
                self.assertEqual(saved_request['options']['structured_method'], entry['method'])
            import yaml

            child_config = yaml.safe_load((artifacts / 'lamp' / 'config.snapshot.yaml').read_text(encoding='utf-8'))
            self.assertEqual(Path(child_config['project']['root']), config_dir / 'models')


if __name__ == '__main__':
    unittest.main()
