"""
Smoke tests for the two study harnesses.

The model has good unit coverage, but until now nothing exercised the scripts
that actually produce the figures. That is the expensive failure mode: a sweep
runs for a minute, finishes, and then falls over in a plotting call, or writes
a CSV with a column the README quotes but the code no longer produces.

These run each harness end to end at a deliberately tiny sample size — enough to
prove the pipeline holds together, not enough to mean anything statistically.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_experiment       # noqa: E402
import run_rcs_sweep        # noqa: E402


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run a harness in a scratch directory with the output/ folder it expects."""
    (tmp_path / "output").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_swarm_harness_runs_end_to_end(workdir, monkeypatch):
    monkeypatch.setattr(run_experiment, "N_TRIALS_PER_CONFIG", 2)
    monkeypatch.setattr(run_experiment, "SWARM_SIZES", [0, 1])

    df = run_experiment.run_all()
    assert len(df) == 4
    for column in ("mothership_survived", "n_drones_lost", "sam_outcome",
                   "min_miss_distance", "offset_deg"):
        assert column in df.columns

    summary = run_experiment.summarize(df)
    # The README quotes these; if they stop being produced, the docs are wrong.
    for column in ("survival_rate", "ci_low", "ci_high", "mean_drones_lost"):
        assert column in summary.columns

    run_experiment.plot_survival(summary)
    run_experiment.plot_sam_outcomes(df)
    run_experiment.plot_sample_trajectory()

    for name in ("survival_vs_swarm_size.png", "sam_outcomes.png",
                 "sample_trajectory.png"):
        written = workdir / "output" / name
        assert written.exists() and written.stat().st_size > 0


def test_rcs_harness_runs_end_to_end(workdir, monkeypatch):
    monkeypatch.setattr(run_rcs_sweep, "N_TRIALS", 3)
    monkeypatch.setattr(run_rcs_sweep, "RCS_GRID", np.array([0.005, 1.0]))
    monkeypatch.setattr(run_rcs_sweep, "MAGAZINES", [1, 4])

    df = run_rcs_sweep.run_sweep()
    assert len(df) == 4
    for column in ("magazine", "escort_rcs_m2", "delta",
                   "delta_ci_low", "delta_ci_high", "shots_at_manned"):
        assert column in df.columns

    # A delta must sit inside its own interval, or the interval is wrong.
    assert ((df.delta >= df.delta_ci_low) & (df.delta <= df.delta_ci_high)).all()

    run_rcs_sweep.plot_crossover(df)
    run_rcs_sweep.plot_detection_physics()

    for name in ("rcs_crossover.png", "detection_range_vs_rcs.png"):
        written = workdir / "output" / name
        assert written.exists() and written.stat().st_size > 0


def test_report_survives_a_degenerate_sweep(workdir, monkeypatch, capsys):
    """report() does interval comparisons and string formatting on the summary;
    it should not raise when a tiny sweep produces no significant effect."""
    monkeypatch.setattr(run_rcs_sweep, "N_TRIALS", 3)
    monkeypatch.setattr(run_rcs_sweep, "RCS_GRID", np.array([0.005]))
    monkeypatch.setattr(run_rcs_sweep, "MAGAZINES", [2])

    run_rcs_sweep.report(run_rcs_sweep.run_sweep())
    assert "magazine 2" in capsys.readouterr().out
