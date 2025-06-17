
import pytest

from src.utils.config_loader import ConfigLoader


@pytest.mark.unit
class TestConfigLoader:
    def test_load_and_cache(self, tmp_path, monkeypatch):
        cfg_content = "key: value"
        cfg_file = tmp_path / "sample.yaml"
        cfg_file.write_text(cfg_content)

        loader = ConfigLoader(config_dir=str(tmp_path))
        cfg1 = loader.load_config("sample")
        cfg2 = loader.load_config("sample")  # Should come from cache
        assert cfg1 == {"key": "value"}
        assert cfg1 is cfg2  # Same object from cache

    def test_env_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "123")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("number: ${TEST_VAR}")
        loader = ConfigLoader(config_dir=str(tmp_path))
        cfg = loader.load_config("cfg")
        assert cfg["number"] == 123

    def test_missing_file_raises(self):
        loader = ConfigLoader(config_dir="nonexistent")
        with pytest.raises(FileNotFoundError):
            loader.load_config("nope")
