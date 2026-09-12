"""
Regression tests for the endgame model.

The headline test here is ``test_flyby_is_caught_that_range_sampling_misses``:
it reconstructs the defect that invalidated every result this project produced
before the CPA solver existed, and fails if range-sampling ever comes back.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lethality import (closest_approach, probability_of_kill,        # noqa: E402
                       hazard_to_step_probability)


# ------------------------------------------------------------ CPA geometry

def test_head_on_pass_recovers_lateral_offset():
    """Two objects closing head-on with a fixed lateral offset: the miss
    distance is exactly that offset, whatever the step size."""
    rel_pos = np.array([1000.0, 7.0, 0.0])     # 1 km ahead, 7 m to the side
    rel_vel = np.array([-1100.0, 0.0, 0.0])    # pure closure
    _, miss, flyby = closest_approach(rel_pos, rel_vel, dt=10.0)
    assert miss == pytest.approx(7.0, abs=1e-9)
    assert flyby


def test_cpa_clamps_to_step_when_still_closing():
    """If the minimum lies beyond the step, the answer is the end-of-step range,
    and it is not reported as a fly-by."""
    rel_pos = np.array([1000.0, 0.0, 0.0])
    rel_vel = np.array([-100.0, 0.0, 0.0])
    t_star, miss, flyby = closest_approach(rel_pos, rel_vel, dt=1.0)
    assert t_star == pytest.approx(1.0)
    assert miss == pytest.approx(900.0)
    assert not flyby


def test_zero_relative_motion_is_stable():
    _, miss, flyby = closest_approach([3.0, 4.0, 0.0], [0.0, 0.0, 0.0], dt=0.05)
    assert miss == pytest.approx(5.0)
    assert not flyby


def test_flyby_is_caught_that_range_sampling_misses():
    """THE regression test for the original defect.

    A missile passes 2 m from its target, but the step is long enough that the
    sampled range is 550 m at both ends. The old ``range < lethal_radius``
    check sees nothing and the target lives; the CPA solver reports 2 m.
    """
    closing = 1100.0
    dt = 1.0
    rel_pos = np.array([closing * dt / 2, 2.0, 0.0])   # 550 m out, 2 m offset
    rel_vel = np.array([-closing, 0.0, 0.0])

    start_range = np.linalg.norm(rel_pos)
    end_range = np.linalg.norm(rel_pos + rel_vel * dt)
    assert start_range > 500 and end_range > 500          # sampling sees nothing

    _, miss, flyby = closest_approach(rel_pos, rel_vel, dt)
    assert flyby
    assert miss == pytest.approx(2.0, abs=1e-9)
    assert probability_of_kill(miss, r50=12.0) > 0.85     # and it is lethal


# --------------------------------------------------------- damage function

def test_pk_is_max_at_zero_miss():
    assert probability_of_kill(0.0, r50=12.0, pk_max=0.9) == pytest.approx(0.9)


def test_pk_is_half_max_at_r50():
    assert probability_of_kill(12.0, r50=12.0, pk_max=0.9) == pytest.approx(0.45)


def test_pk_decays_monotonically():
    vals = [probability_of_kill(m, r50=12.0) for m in (0, 5, 10, 20, 50, 100)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 1e-3


# ----------------------------------------------------- timestep invariance

def test_hazard_conversion_is_timestep_invariant():
    """A 5 Hz hazard must produce the same survival over 1 s of modeled time
    regardless of how many steps that second is cut into."""
    rate, span = 5.0, 1.0
    reference = np.exp(-rate * span)
    for dt in (0.5, 0.05, 0.005, 0.001):
        n = int(round(span / dt))
        p_step = hazard_to_step_probability(rate, dt)
        survival = (1 - p_step) ** n
        assert survival == pytest.approx(reference, rel=1e-9)


def test_legacy_per_tick_probability_was_not_invariant():
    """Documents the behavior that was replaced: a fixed per-tick probability
    makes the modeled effect depend on the integration step."""
    p_tick, span = 0.35, 1.0
    survivals = {dt: (1 - p_tick) ** int(round(span / dt)) for dt in (0.05, 0.025)}
    assert survivals[0.05] != pytest.approx(survivals[0.025], rel=0.01)
