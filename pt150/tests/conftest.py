import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ncm150.config import load_cfg

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full-size 150M model builds on CPU (several GB RAM)")

@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data"); r = np.random.default_rng(0)
    r.integers(0, 128, 16 * 37, dtype=np.uint16).tofile(d / "train_000.bin")
    r.integers(0, 128, 16 * 9, dtype=np.uint16).tofile(d / "train_001.bin")
    r.integers(0, 128, 16 * 8, dtype=np.uint16).tofile(d / "val.bin")
    return str(d)

def tiny_cfg(arm, data, out, *extra):
    return load_cfg("cpu_tiny.yaml", overrides=[f"arm={arm}", f"data.dir={data}", f"out_dir={out}", *extra])
