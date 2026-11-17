"""Yang et al. 2024 (Drones 8(4):136) 數值重現。"""
import os
from pathlib import Path

# 同學長 模擬程式/main.py 第 20 行：dive_sim 環境裡 numpy（MKL）和 PyTorch 各帶一份 OpenMP，
# 載入 YOLO 時會出現「OMP: Error #15」並直接結束。必須在任何 numpy / torch 匯入前設定。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent


def load_params(path=None, **overrides):
    """讀 params.yaml，可用關鍵字參數覆寫個別值。"""
    path = Path(path) if path else PKG_DIR / "params.yaml"
    with open(path, encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    for key, value in overrides.items():
        if key not in p:
            raise KeyError(f"params.yaml 沒有參數 {key!r}")
        p[key] = value
    return p
