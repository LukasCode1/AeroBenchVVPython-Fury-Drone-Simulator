"""
Tests for the mission-level detection and shot-allocation model.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import SearchRadar, SamBattery, run_ingress     # noqa: E402
from mcstats import wilson_interval, newcombe_interval          # noqa: E402


# ------------------------------------------------------------ radar equation

def test_detection_range_scales_as_fourth_root_of_rcs():
    """The whole reason low observability is expensive: a 10x RCS reduction
    buys only a 10^0.25 = 1.78x reduction in detection range."""
    radar = SearchRadar()
    assert radar.detection_range(10.0) / radar.detection_range(1.0) == pytest.approx(10 ** 0.25)
    assert radar.detection_range(0.1) / radar.detection_range(1.0) == pytest.approx(10 ** -0.25)


def test_ten_db_rcs_reduction_cuts_detection_range_by_about_44_percent():
    radar = SearchRadar()
    ratio = radar.detection_range(0.1) / radar.detection_range(1.0)
    assert 0.55 < ratio < 0.57


def test_pd_is_one_half_at_the_reference_range():
    """The SNR scale is anchored so this holds by construction; if the anchor
    drifts, every absolute detection range in the study moves with it."""
    radar = SearchRadar(detection_range_ref_m=150_000.0, rcs_ref_m2=1.0)
    assert radar.p_detect_per_scan(1.0, 150_000.0) == pytest.approx(0.5, abs=1e-9)


def test_pd_is_monotonic_in_range_and_in_rcs():
    radar = SearchRadar()
    by_range = [radar.p_detect_per_scan(1.0, r) for r in (50e3, 100e3, 150e3, 200e3)]
    assert all(a > b for a, b in zip(by_range, by_range[1:]))
    by_rcs = [radar.p_detect_per_scan(s, 120e3) for s in (0.001, 0.01, 0.1, 1.0)]
    assert all(a < b for a, b in zip(by_rcs, by_rcs[1:]))


# ------------------------------------------------------------------- battery

def test_cold_battery_cannot_fire_before_its_reaction_time():
    battery = SamBattery(reaction_time_s=75.0)
    assert not battery.can_fire(0.0)          # nothing detected yet
    battery.alert(100.0)
    assert not battery.can_fire(150.0)        # alerted, still spinning up
    assert battery.can_fire(175.0)


def test_alert_clock_starts_once_and_does_not_restart():
    battery = SamBattery(reaction_time_s=50.0)
    battery.alert(10.0)
    battery.alert(200.0)                       # a second contact must not reset it
    assert battery.alerted_at == 10.0


def test_battery_magazine_is_not_depleted_across_trials():
    """run_ingress copies a caller-supplied battery. Without that, a sweep
    silently runs out of missiles partway through and every later trial
    reports an unopposed ingress."""
    battery = SamBattery(magazine=2)
    fired = [run_ingress(escort_rcs_m2=5.0, battery=battery,
                         rng=np.random.default_rng([1, i]))["shots_fired"]
             for i in range(6)]
    assert battery.shots_fired == 0            # caller's instance untouched
    assert max(fired) > 0 and all(f <= 2 for f in fired)


# ------------------------------------------------------------------ geometry

def test_escort_ahead_shares_the_manned_bearing():
    """An escort directly ahead is on the same radial as the aircraft it is
    screening, so it cannot decouple its cue from the manned aircraft however
    far forward it flies. Cross-track separation is what buys angular
    separation -- this is why lead distance is not a substitute for it."""
    radar = SearchRadar(cue_sector_deg=5.0)
    manned_range = 100_000.0

    ahead = np.arctan2(0.0, manned_range - 40_000.0)
    assert abs(ahead) < radar.cue_sector       # still inside the cued sector

    abeam = np.arctan2(20_000.0, manned_range)
    assert abs(abeam) > radar.cue_sector       # genuinely separated


def test_more_visible_escort_is_detected_further_out():
    ranges = []
    for rcs in (0.01, 1.0):
        detections = [run_ingress(escort_rcs_m2=rcs, escort_cross_track_m=8000.0,
                                  rng=np.random.default_rng([2, i]))["escort_detection_range_m"]
                      for i in range(200)]
        hits = [d for d in detections if d is not None]
        ranges.append(np.mean(hits))
    assert ranges[1] > ranges[0]


# ----------------------------------------------------------- reproducibility

def test_same_stream_reproduces_the_ingress():
    a = run_ingress(escort_rcs_m2=0.5, rng=np.random.default_rng([7, 7]))
    b = run_ingress(escort_rcs_m2=0.5, rng=np.random.default_rng([7, 7]))
    assert a == b


# ----------------------------------------------------------------- intervals

def test_wilson_interval_stays_inside_the_unit_range():
    """The normal approximation produces bounds outside [0, 1] at the extremes,
    which is exactly where survival rates sit."""
    lo, hi = wilson_interval(0, 60)
    assert lo == 0.0 and 0 < hi < 0.1
    lo, hi = wilson_interval(60, 60)
    assert hi <= 1.0 and hi == pytest.approx(1.0)
    assert 0.9 < lo < 1.0


def test_newcombe_interval_brackets_the_difference():
    lo, hi = newcombe_interval(80, 100, 50, 100)
    assert lo < 0.30 < hi
    assert lo > 0                               # a 30-point gap at n=100 is real


def test_newcombe_interval_covers_zero_for_identical_rates():
    lo, hi = newcombe_interval(50, 100, 50, 100)
    assert lo < 0 < hi
