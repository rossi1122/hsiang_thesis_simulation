"""重現論文 Section 4 的三組數值實驗，結果存到 results/<日期>_<實驗名>/。

用法（在 repo 根目錄）：
    python -m yang2024.experiments fig4      # LSM vs 提出方法，V_c = 7 m/s
    python -m yang2024.experiments fig5      # P1,soft vs 提出方法，V_c = 9 m/s
    python -m yang2024.experiments fig6      # LSM vs 提出方法，目標複雜運動
    python -m yang2024.experiments all
可加 --T 秒數縮短模擬、--set key=value 覆寫 params.yaml（例如 --set switch_mapping=proof）。
"""
import argparse
import json
import platform
import time
from datetime import datetime

import numpy as np
import yaml

from . import REPO_DIR, load_params
from .metrics import summarize
from .models import fig6_profile
from .mpc import YangMPC
from .plots import comparison_figure, features_figure
from .sim import simulate

LABELS = {"lsm": "LSM method", "p1soft": "P1,soft", "switch": "proposed method"}

# 論文數值（Section 4，單位 px，f = 537）
PAPER = {
    "fig4": {"lsm": 226.55, "switch": 153.63},
    "fig5": {"p1soft": 228.56, "switch": 238.38},
}

EXPERIMENTS = {
    "fig4": dict(controllers=("switch", "lsm"), Vc=7.0, T=150.0, profile=None,
                 title="Fig. 4 reproduction: V_c = 7 m/s"),
    "fig5": dict(controllers=("switch", "p1soft"), Vc=9.0, T=150.0, profile=None,
                 title="Fig. 5 reproduction: V_c = 9 m/s"),
    "fig6": dict(controllers=("switch", "lsm"), Vc=6.0, T=500.0, profile=fig6_profile,
                 title="Fig. 6 reproduction: complex target motion (profile read from figure)"),
}


def run(name, params, T=None, out_root=None):
    spec = EXPERIMENTS[name]
    T = spec["T"] if T is None else T
    out_dir = (out_root or REPO_DIR / "results") / f"{datetime.now():%Y%m%d}_yang2024_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mpc = YangMPC(params)
    runs, metrics = {}, {}
    for ctrl in spec["controllers"]:
        t0 = time.time()
        log = simulate(params, ctrl, Vc=spec["Vc"], T=T, target_profile=spec["profile"], mpc=mpc)
        runs[LABELS[ctrl]] = log
        m = summarize(log, params)
        m["wall_time_s"] = round(time.time() - t0, 1)
        if name in PAPER and ctrl in PAPER[name] and T == spec["T"]:
            m["paper_rmse_px"] = PAPER[name][ctrl]
        metrics[ctrl] = m
        np.savez_compressed(out_dir / f"log_{ctrl}.npz", **log)

    comparison_figure(runs, params, spec["title"], out_dir / f"{name}.png")
    features_figure(runs, out_dir / f"{name}_s2.png")

    record = {
        "experiment": name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "V_c": spec["Vc"], "T": T,
        "metrics": metrics,
        "params": params,
        "python": platform.python_version(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    with open(out_dir / "params_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(params, fh, allow_unicode=True, sort_keys=False)
    return out_dir, metrics


def _print(name, metrics):
    print(f"\n== {name} ==")
    keys = ["rmse_s2_px", "paper_rmse_px", "R_max_m", "time_outside_RM_s",
            "longest_outside_s", "u_sat_ratio", "solve_time_mean_ms", "solve_failures"]
    print("controller".ljust(10) + "".join(k.rjust(20) for k in keys))
    for ctrl, m in metrics.items():
        row = "".join(
            (f"{m[k]:20.2f}" if isinstance(m.get(k), float) else str(m.get(k, "-")).rjust(20))
            for k in keys)
        print(ctrl.ljust(10) + row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--T", type=float, default=None, help="模擬秒數（預設照論文）")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="覆寫 params.yaml 的值，例如 switch_mapping=proof")
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    params = load_params(**overrides)

    names = list(EXPERIMENTS) if args.which == "all" else [args.which]
    for name in names:
        out_dir, metrics = run(name, params, T=args.T)
        _print(name, metrics)
        print(f"結果存在 {out_dir.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
