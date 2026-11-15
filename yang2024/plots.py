"""畫成論文 Fig. 4–6 的樣式：軌跡、u_psi、水平距離 R、(u2, v2)。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLES = [dict(color="#2a9d8f", lw=1.4), dict(color="#7b6fd6", lw=1.2, ls="--")]


def comparison_figure(runs, params, title, path):
    """runs: {標籤: log}。第一個實線、第二個虛線（同論文）。"""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    (a_traj, a_u), (a_R, a_s) = ax
    for (label, log), st in zip(runs.items(), STYLES):
        a_traj.plot(log["x"], log["y"], label=label, **st)
        a_u.plot(log["t"], log["u_psi"], label=label, **st)
        a_R.plot(log["t"], log["R"], label=label, **st)
        a_s.plot(log["t"], log["u2"], label=f"u2 {label}", **st)
    first = next(iter(runs.values()))
    a_traj.plot(first["tx"], first["ty"], color="k", lw=2, label="target")
    a_traj.set(xlabel="x (m)", ylabel="y (m)", title="(a) Trajectory")
    a_traj.set_aspect("equal", adjustable="datalim")
    a_u.axhline(-params["u_max"], color="k", ls="--", lw=1)
    a_u.axhline(params["u_max"], color="k", ls="--", lw=1)
    a_u.set(xlabel="t (s)", ylabel="u_psi (rad/s)", title="(b) u_psi")
    a_R.axhline(params["R_M"], color="k", ls="--", lw=1)
    a_R.set(xlabel="t (s)", ylabel="R (m)", title="(c) Relative horizontal distance")
    a_s.set(xlabel="t (s)", ylabel="u2 (px)", title="(d) u2 (v2: see *_s2.png)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def features_figure(runs, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for (label, log), st in zip(runs.items(), STYLES):
        ax[0].plot(log["t"], log["u2"], label=label, **st)
        ax[1].plot(log["t"], log["v2"], label=label, **st)
    ax[0].set_ylabel("u2 (px)")
    ax[1].set_ylabel("v2 (px)")
    ax[1].set_xlabel("t (s)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
