"""
run_rcs_sweep.py
----------------
Does an escort drone make the manned aircraft safer, or does it just tell the
defender where to look?

This is the criticism most often levelled at manned-unmanned teaming: a drone
with a larger radar cross section is seen further out, and seeing it starts the
defender's clock on a patch of sky the manned aircraft is also flying through.
Set against that, every missile spent on the drone is one not spent on the
aircraft.

Both effects are real and they pull in opposite directions, so the sign of the
escort's contribution is an empirical question. This sweep answers it over two
parameters: how visible the escort is, and how many missiles the defender has.

Usage:
    python run_rcs_sweep.py
Produces:
    output/rcs_sweep_results.csv
    output/rcs_crossover.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_style
from detection import run_ingress, SamBattery, SearchRadar, RCS_REFERENCE
from mcstats import wilson_interval, newcombe_interval

N_TRIALS = 800
MASTER_SEED = 20260912

# Escort RCS sweep, spanning "stealthier than the aircraft it escorts" through
# "conventional fighter". Open published ranges put a stealth aircraft at
# 0.001-0.01 m^2, a cruise missile at 0.1-1, and a fighter at 1-10.
RCS_GRID = np.logspace(-4, 1.5, 14)

MAGAZINES = [1, 2, 4, 8]

MANNED_RCS = 0.005          # notional stealth strike aircraft
ESCORT_CROSS_TRACK = 8000.0  # abeam separation (m)
RELEASE_RANGE = 30_000.0     # how far in the manned aircraft has to get


def survival(escort_rcs, magazine, n=N_TRIALS, tag=0):
    """Fraction of trials in which the manned aircraft reaches its release point."""
    battery = SamBattery(magazine=magazine)
    wins = 0
    for i in range(n):
        rng = np.random.default_rng([MASTER_SEED, tag, magazine, i])
        res = run_ingress(manned_rcs_m2=MANNED_RCS, escort_rcs_m2=escort_rcs,
                          escort_cross_track_m=ESCORT_CROSS_TRACK,
                          release_range_m=RELEASE_RANGE,
                          battery=battery, rng=rng)
        wins += res["manned_survived"]
    return wins


def shots_at_manned(escort_rcs, magazine, n=N_TRIALS, tag=0):
    battery = SamBattery(magazine=magazine)
    total = 0
    for i in range(n):
        rng = np.random.default_rng([MASTER_SEED, tag, magazine, i])
        res = run_ingress(manned_rcs_m2=MANNED_RCS, escort_rcs_m2=escort_rcs,
                          escort_cross_track_m=ESCORT_CROSS_TRACK,
                          release_range_m=RELEASE_RANGE,
                          battery=battery, rng=rng)
        total += res["manned_shots_taken"]
    return total / n


def run_sweep():
    rows = []
    for magazine in MAGAZINES:
        # Baseline: same aircraft, same threat, no escort at all.
        base_wins = survival(None, magazine, tag=1)
        base_shots = shots_at_manned(None, magazine, tag=1)

        for rcs in RCS_GRID:
            wins = survival(rcs, magazine, tag=2)
            lo, hi = newcombe_interval(wins, N_TRIALS, base_wins, N_TRIALS)
            rows.append({
                "magazine": magazine,
                "escort_rcs_m2": rcs,
                "escort_detection_range_km": SearchRadar().detection_range(rcs) / 1000,
                "survival": wins / N_TRIALS,
                "baseline_survival": base_wins / N_TRIALS,
                "delta": wins / N_TRIALS - base_wins / N_TRIALS,
                "delta_ci_low": lo,
                "delta_ci_high": hi,
                "shots_at_manned": shots_at_manned(rcs, magazine, tag=2),
                "baseline_shots_at_manned": base_shots,
                "n_trials": N_TRIALS,
            })
    return pd.DataFrame(rows)


def plot_crossover(df):
    viz_style.apply()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- left: the crossover itself, as a change against flying alone
    ax.axhline(0, color=viz_style.INK, lw=1.2, zorder=3)

    for i, magazine in enumerate(MAGAZINES):
        sub = df[df.magazine == magazine]
        colour = viz_style.SERIES[i]
        ax.fill_between(sub.escort_rcs_m2, sub.delta_ci_low * 100,
                        sub.delta_ci_high * 100, color=colour, alpha=0.12, linewidth=0)
        ax.plot(sub.escort_rcs_m2, sub.delta * 100, "o-", color=colour,
                markersize=4, label=f"{magazine} missile" + ("s" if magazine > 1 else ""))

    ax.set_xscale("log")
    ax.set_xlabel("Escort radar cross section (m$^2$)")
    ax.set_ylabel("Change in manned survival (percentage points)")
    ax.set_title("Escorting helps only while the defender runs out of missiles\n"
                 f"{N_TRIALS} trials per point, bands = 95% Newcombe intervals",
                 fontsize=12)
    ax.legend(title="Defender magazine", loc="lower left")

    # Anchor the abstract x-axis to recognisable objects. The labels live in a
    # reserved band above the data rather than on top of it.
    lo, hi = ax.get_ylim()
    band = (hi - lo) * 0.16
    ax.set_ylim(lo, hi + band)
    label_y = hi + band * 0.45
    for label, rcs in sorted(RCS_REFERENCE.items(), key=lambda kv: kv[1]):
        if RCS_GRID[0] <= rcs <= RCS_GRID[-1]:
            ax.axvline(rcs, color=viz_style.GRID, lw=1.0, zorder=0)
            ax.text(rcs, label_y, label, fontsize=8, color=viz_style.INK_MUTED,
                    ha="center", va="center")

    ax.annotate("escort makes no difference", (RCS_GRID[-1], 0),
                textcoords="offset points", xytext=(-4, 5), ha="right",
                fontsize=8.5, color=viz_style.INK_SECONDARY)

    # --- right: the mechanism
    for i, magazine in enumerate(MAGAZINES):
        sub = df[df.magazine == magazine]
        ax2.plot(sub.escort_rcs_m2, sub.shots_at_manned, "o-",
                 color=viz_style.SERIES[i], markersize=4,
                 label=f"{magazine} missile" + ("s" if magazine > 1 else ""))
        ax2.axhline(sub.baseline_shots_at_manned.iloc[0], color=viz_style.SERIES[i],
                    lw=1.0, linestyle=(0, (2, 3)), alpha=0.7)

    ax2.set_xscale("log")
    ax2.set_xlabel("Escort radar cross section (m$^2$)")
    ax2.set_ylabel("Missiles fired at the manned aircraft")
    ax2.set_title("Why: a shallow magazine empties into the drone\n"
                  "dotted = same magazine with no escort present", fontsize=12)

    fig.tight_layout()
    fig.savefig("output/rcs_crossover.png", bbox_inches="tight")
    plt.close(fig)


def plot_detection_physics():
    """The fourth-root law, with recognisable objects on it.

    One curve, one colour: this figure has a single job, which is to show why
    low observability is bought in decibels and paid for in fourth roots.
    """
    viz_style.apply()
    radar = SearchRadar()
    grid = np.logspace(-4, 2.2, 300)
    ranges = np.array([radar.detection_range(s) for s in grid]) / 1000

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(grid, ranges, color=viz_style.SERIES[0], lw=2.2, zorder=2)

    for label, rcs in sorted(RCS_REFERENCE.items(), key=lambda kv: kv[1]):
        r = radar.detection_range(rcs) / 1000
        ax.plot([rcs], [r], "o", color=viz_style.SERIES[0], markersize=7,
                markeredgecolor=viz_style.SURFACE, markeredgewidth=2, zorder=3)
        # Labels sit below the curve where there is room, and flip above near
        # the bottom of the axes so they never run into the x-axis.
        low = r < 0.18 * (ranges.max() * 1.12)
        ax.annotate(f"{label}\n{r:.0f} km", (rcs, r), textcoords="offset points",
                    xytext=(9, 4 if low else -4), fontsize=8.5,
                    color=viz_style.INK_SECONDARY, va="bottom" if low else "top")

    ax.set_xscale("log")
    ax.set_xlabel("Radar cross section (m$^2$)")
    ax.set_ylabel("Detection range (km)")
    ax.set_title("Detection range scales as the fourth root of RCS\n"
                 "a 10 dB signature reduction buys 44% less detection range, not 90%",
                 fontsize=12)
    ax.set_xlim(grid[0], grid[-1] * 3)
    ax.set_ylim(0, ranges.max() * 1.12)

    fig.tight_layout()
    fig.savefig("output/detection_range_vs_rcs.png", bbox_inches="tight")
    plt.close(fig)


def report(df):
    print(f"Manned aircraft RCS {MANNED_RCS} m^2, escort {ESCORT_CROSS_TRACK/1000:.0f} km abeam, "
          f"release at {RELEASE_RANGE/1000:.0f} km, {N_TRIALS} trials per point.\n")
    for magazine in MAGAZINES:
        sub = df[df.magazine == magazine]
        base = sub.baseline_survival.iloc[0]
        helps = sub[sub.delta_ci_low > 0]
        hurts = sub[sub.delta_ci_high < 0]
        print(f"magazine {magazine}: no-escort survival {base:.3f}")
        if len(helps):
            print(f"    escort helps for RCS {helps.escort_rcs_m2.min():.4g} "
                  f"- {helps.escort_rcs_m2.max():.4g} m^2 "
                  f"(best {helps.delta.max()*100:+.0f} points)")
        if len(hurts):
            print(f"    escort HURTS for RCS {hurts.escort_rcs_m2.min():.4g} "
                  f"- {hurts.escort_rcs_m2.max():.4g} m^2 "
                  f"(worst {hurts.delta.min()*100:+.0f} points)")
        if not len(helps) and not len(hurts):
            print("    no significant effect at any tested RCS")


if __name__ == "__main__":
    df = run_sweep()
    df.to_csv("output/rcs_sweep_results.csv", index=False)
    report(df)
    plot_crossover(df)
    plot_detection_physics()
    print("\nDone. See output/rcs_crossover.png")
