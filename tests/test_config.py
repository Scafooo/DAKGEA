import tempfile
import unittest
from pathlib import Path

import yaml

from src.config.loader import Config, PROJECT_ROOT


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class ConfigLoaderTests(unittest.TestCase):
    def test_config_merges_and_resolves_paths(self):
        global_cfg = {
            "paths": {
                "log_file": "logs/pipeline.log",
                "raw_data": "data/raw",
            },
            "logging": {"level": "INFO"},
            "model": {"embedding_dim": 64},
        }
        model_cfg = {
            "model": {"embedding_dim": 128, "dropout": 0.2},
            "paths": {"model_dir": "models"},
        }
        exp_cfg = {
            "paths": {"raw_data": "data/custom/raw"},
            "experiment": {"epochs": 5},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            global_file = tmp_path / "global.yaml"
            model_file = tmp_path / "model.yaml"
            exp_file = tmp_path / "experiment.yaml"
            _write_yaml(global_file, global_cfg)
            _write_yaml(model_file, model_cfg)
            _write_yaml(exp_file, exp_cfg)

            cfg = Config(
                global_path=global_file, model_path=model_file, exp_path=exp_file
            ).get()

        expected_raw_data = (PROJECT_ROOT / "data/custom/raw").resolve()
        expected_model_dir = (PROJECT_ROOT / "models").resolve()
        expected_log_file = (PROJECT_ROOT / "logs/pipeline.log").resolve()

        self.assertEqual(cfg["paths"]["raw_data"], expected_raw_data)
        self.assertEqual(cfg["paths"]["model_dir"], expected_model_dir)
        self.assertEqual(cfg["paths"]["log_file"], expected_log_file)
        self.assertEqual(cfg["model"]["embedding_dim"], 128)
        self.assertEqual(cfg["model"]["dropout"], 0.2)
        self.assertEqual(cfg["experiment"]["epochs"], 5)


if __name__ == "__main__":
    unittest.main()
