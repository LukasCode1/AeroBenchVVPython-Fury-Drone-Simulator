"""
run_experiment.py
-----------------
Research harness: runs Monte Carlo trials across swarm sizes (and, optionally,
threat geometry) and reports mothership survival rate, drone attrition, and
missile miss-distance statistics -- the core numbers for a "does a Fury-style
escort swarm help the mothership survive a SAM shot" study.

Reproducibility
---------------
Every trial draws from its own independent, named random stream, addressed by
the coordinate ``[MASTER_SEED, n_drones, trial]``. Re-running the sweep
reproduces it exactly; re-running a single trial in isolation reproduces that
trial exactly; and adding a new swarm size does not perturb the trials already
run. The earlier harness called ``np.random.seed(None)``, so none of that held.

Uncertainty
-----------
A survival rate estimated from N trials is a binomial proportion and is reported
with a 95% Wilson score interval. At the default 60 trials per configuration the
interval is roughly +/- 10 points, which is wider than most of the differences
between configurations -- so the intervals are plotted, not just recorded, and
``required_trials`` prints the sample size that would actually be needed to
resolve an effect of a given size.

Usage:
    python run_experiment.py
Produces:
    output/experiment_results.csv    one row per trial
    output/experiment_summary.csv    aggregated by swarm size, with CIs
    output/survival_vs_swarm_size.png
    output/sam_outcomes.png
    output/sample_trajectory.png     one illustrative 3-drone engagement
"""

import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_style
from engagement import run_engagement

N_TRIALS_PER_CONFIG = 60
SWARM_SIZES = [0, 1, 2, 3, 4, 6, 8]
SAM_RANGE = 20000.0
MASTER_SEED = 20260912
OFFSET_LIMIT_DEG = 25.0


# --------------------------------------------------------------------- stats

def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    Preferred over the normal approximation, which misbehaves badly for
    proportions near 0 or 1 -- exactly where survival rates live.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def required_trials(p1, p2, alpha=0.05, power=0.80):
    """Trials per arm needed to resolve a change from p1 to p2."""
    from statistics import NormalDist
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


# ------------------------------------------------------------------ sweeping

def run_all():
    rows = []
    for n_drones in SWARM_SIZES:
        for trial in range(N_TRIALS_PER_CONFIG):
            # Independent, addressable stream per trial. The offset is drawn
            # from the trial's own stream so geometry is reproducible too.
            rng = np.random.default_rng([MASTER_SEED, n_drones, trial])
            offset_angle = np.radians(rng.uniform(-OFFSET_LIMIT_DEG, OFFSET_LIMIT_DEG))
            result, mothership, drones, sam = run_engagement(
                n_drones=n_drones,
                sam_range=SAM_RANGE,
                sam_offset_angle=offset_angle,
                seed=rng,
            )
            rows.append({
                "n_drones": n_drones,
                "trial": trial,
                "offset_deg": np.degrees(offset_angle),
                "mothership_survived": result.mothership_survived,
                "n_drones_lost": result.n_drones_lost,
                "time_to_resolution": result.time_to_resolution,
                "min_miss_distance": result.min_missile_miss_distance,
                "sam_outcome": result.sam_outcome,
                "sam_victim": result.sam_victim,
            })
    return pd.DataFrame(rows)


def summarize(df):
    summary = df.groupby("n_drones").agg(
        survivors=("mothership_survived", "sum"),
        n_trials=("trial", "count"),
        survival_rate=("mothership_survived", "mean"),
        mean_drones_lost=("n_drones_lost", "mean"),
        mean_miss_distance=("min_miss_distance", "mean"),
    ).reset_index()

    ci = [wilson_interval(int(s), int(n))
          for s, n in zip(summary["survivors"], summary["n_trials"])]
    summary["ci_low"] = [c[0] for c in ci]
    summary["ci_high"] = [c[1] for c in ci]
    summary["ci_halfwidth_pts"] = (summary["ci_high"] - summary["ci_low"]) * 50

    print(summary.to_string(index=False))

    base = summary.loc[summary["n_drones"] == SWARM_SIZES[0], "survival_rate"].iloc[0]
    best = summary["survival_rate"].max()
    print(f"\nSurvival {base:.2f} (no escort) -> {best:.2f} (best configuration).")
    print(f"95% Wilson interval half-width at n={N_TRIALS_PER_CONFIG}: "
          f"+/- {summary['ci_halfwidth_pts'].mean():.0f} points.")
    if abs(best - base) > 1e-9:
        lo, hi = (base, best) if best > base else (best, base)
        print(f"Resolving {lo:.2f} -> {hi:.2f} at 80% power would need "
              f"{required_trials(lo, hi)} trials per arm.")
    return summary


# ------------------------------------------------------------------- figures

def plot_survival(summary):
    """Survival rate vs swarm size, with the uncertainty that goes with it.

    Deliberately a single y-axis. The earlier version of this figure plotted
    survival rate and mean drones lost on two y-scales; the alignment between
    two scales is arbitrary, so such a chart invents a relationship that is not
    in the data. Drone attrition is now its own panel.
    """
    viz_style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    x = summary["n_drones"]
    y = summary["survival_rate"] * 100
    lo = summary["ci_low"] * 100
    hi = summary["ci_high"] * 100

    ax1.fill_between(x, lo, hi, color=viz_style.SERIES[0], alpha=0.15, linewidth=0)
    ax1.plot(x, y, "o-", color=viz_style.SERIES[0])
    ax1.set_xlabel("Escort swarm size (drones)")
    ax1.set_ylabel("Mothership survival rate (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title(f"Mothership survival vs escort swarm size\n"
                  f"{N_TRIALS_PER_CONFIG} trials per size, band = 95% Wilson interval",
                  fontsize=12)

    # Direct-label the endpoints rather than every point. Labels flip below the
    # marker near the top of the axes so they never collide with the title.
    for idx in (0, len(x) - 1):
        value = y.iloc[idx]
        dy = -16 if value > 90 else 12
        ax1.annotate(f"{value:.0f}%", (x.iloc[idx], value),
                     textcoords="offset points", xytext=(0, dy),
                     ha="center", fontsize=9, color=viz_style.INK_SECONDARY)

    ax2.plot(x, summary["mean_drones_lost"], "s-", color=viz_style.SERIES[1])
    ax2.set_xlabel("Escort swarm size (drones)")
    ax2.set_ylabel("Mean drones lost per engagement")
    ax2.set_title("Escort attrition")
    if summary["mean_drones_lost"].max() <= 0:
        # An all-zero series on an auto-scaled axis renders noise as signal.
        # Pin the scale and say so in words instead.
        ax2.set_ylim(0, 1)
        ax2.text(0.5, 0.5, "No escort losses in any trial",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=11, color=viz_style.INK_MUTED)
    else:
        ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig("output/survival_vs_swarm_size.png", bbox_inches="tight")
    plt.close(fig)


def plot_sam_outcomes(df):
    """How the shot actually resolved -- the accounting the survival rate hides."""
    viz_style.apply()
    order = ["kill", "miss", "energy_exhausted", "unresolved"]
    labels = {"kill": "Warhead killed something", "miss": "Burst, no kill",
              "energy_exhausted": "Ran out of energy", "unresolved": "Unresolved at t_max"}
    colors = {"kill": viz_style.CRITICAL, "miss": viz_style.SERIES[0],
              "energy_exhausted": viz_style.SERIES[2], "unresolved": viz_style.INK_MUTED}

    pivot = (df.groupby(["n_drones", "sam_outcome"]).size()
               .unstack(fill_value=0).reindex(columns=order, fill_value=0))
    frac = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bottom = np.zeros(len(frac))
    for key in order:
        if frac[key].sum() == 0:
            continue
        ax.bar(frac.index, frac[key], bottom=bottom, label=labels[key],
               color=colors[key], width=0.62, linewidth=2.0,
               edgecolor=viz_style.SURFACE)   # 2px surface gap, not a border
        bottom += frac[key].values

    ax.set_xlabel("Escort swarm size (drones)")
    ax.set_ylabel("Share of trials (%)")
    ax.set_ylim(0, 100)
    ax.set_title("How the SAM shot resolved")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig("output/sam_outcomes.png", bbox_inches="tight")
    plt.close(fig)


def plot_sample_trajectory():
    """One engagement, shown at two scales.

    The overview panel drops equal-aspect on purpose: the engagement spans tens
    of km downrange but only hundreds of metres across, so an equal-aspect view
    of the whole thing collapses every manoeuvre onto a single line. The endgame
    panel restores equal aspect over the last few km, where the geometry is the
    point.
    """
    viz_style.apply()
    result, mothership, drones, sam = run_engagement(
        n_drones=3, sam_range=SAM_RANGE, sam_offset_angle=0.0,
        seed=[MASTER_SEED, 999], log_trajectories=True)

    # Final role per drone, for legend labels that mean something.
    final_roles = result.roles_over_time[-1] if result.roles_over_time else {}

    mh = np.array(mothership.history)
    sh = np.array(sam.history, dtype=object)
    sx = np.array([r[1] for r in sh], dtype=float)
    sy = np.array([r[2] for r in sh], dtype=float)

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(12, 4.8))

    def draw(a):
        a.plot(mh[:, 1] / 1000, mh[:, 2] / 1000, color=viz_style.MOTHERSHIP,
               lw=2.2, label="Mothership (F-16)", zorder=5)
        for i, d in enumerate(drones):
            if len(d.history) == 0:
                continue
            dh = np.array(d.history)
            role = final_roles.get(d.name, "ESCORT")
            a.plot(dh[:, 1] / 1000, dh[:, 2] / 1000, lw=1.6,
                   color=viz_style.SERIES[i % len(viz_style.SERIES)],
                   label=f"{d.name} ({role.title()})")
        a.plot(sx / 1000, sy / 1000, color=viz_style.THREAT, lw=1.8,
               linestyle=(0, (4, 3)), label="SAM", zorder=4)
        a.set_xlabel("North (km)")
        a.set_ylabel("East (km)")

    draw(ax)
    ax.set_title("Engagement overview\ncross-track scale exaggerated", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()

    draw(axz)
    # Frame the endgame on where the shot actually resolved, not on wherever the
    # mothership's last logged sample happened to be.
    cx, cy = sx[-1] / 1000, sy[-1] / 1000
    pad = 1.2
    axz.set_xlim(cx - pad, cx + pad)
    axz.set_ylim(cy - pad, cy + pad)
    axz.set_aspect("equal", adjustable="box")
    axz.set_title("Endgame (equal scale)", fontsize=12)

    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04))

    outcome = ("mothership destroyed" if not result.mothership_survived
               else f"mothership survived, SAM {result.sam_outcome.replace('_', ' ')}")
    miss = result.min_missile_miss_distance
    fig.suptitle(f"Sample engagement - {outcome}"
                 + (f", closest approach {miss:.1f} m" if miss is not None else ""),
                 x=0.01, ha="left", fontsize=12, color=viz_style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("output/sample_trajectory.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = run_all()
    summary = summarize(df)
    summary.to_csv("output/experiment_summary.csv", index=False)
    df.to_csv("output/experiment_results.csv", index=False)
    plot_survival(summary)
    plot_sam_outcomes(df)
    plot_sample_trajectory()
    print("\nDone. See output/ for CSVs and plots.")
