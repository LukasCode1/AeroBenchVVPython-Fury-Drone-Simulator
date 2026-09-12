"""
End-to-end invariants for the engagement model.

These are slower than the unit tests (each one runs the full nonlinear F-16
model) but they are the ones that would have caught the original defect at the
level a reader of the README actually cares about: a perfectly guided missile
against a target that never manoeuvres should almost always kill it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engagement import run_engagement            # noqa: E402
from f16_mothership import F16Mothership         # noqa: E402
from missile import GenericSAM                   # noqa: E402


def test_unopposed_sam_reaches_its_target():
    """No escorts, no countermeasures, non-manoeuvring target: closest approach
    should be essentially zero. Before the CPA fix this returned 16-55 m and the
    target always lived."""
    misses = []
    for i in range(5):
        res, _, _, _ = run_engagement(n_drones=0, sam_offset_angle=np.radians(-20 + 10 * i),
                                      seed=[1, i])
        misses.append(res.min_missile_miss_distance)
    assert max(misses) < 1.0, f"unopposed PN missile missed by {max(misses):.2f} m"


def test_unopposed_kill_rate_matches_damage_function():
    """With zero miss distance the kill rate must land on pk_max, not on
    whatever the sampling grid happens to allow."""
    kills = 0
    n = 60
    for i in range(n):
        res, _, _, _ = run_engagement(n_drones=0,
                                      sam_offset_angle=np.radians(np.linspace(-25, 25, n)[i]),
                                      seed=[2, i])
        kills += not res.mothership_survived
    rate = kills / n
    # pk_max = 0.9; +/- 4 sigma at n=60 is about 0.12
    assert 0.78 <= rate <= 1.0, f"unopposed kill rate {rate:.2f} is not consistent with pk_max=0.9"


def test_outcome_is_timestep_invariant():
    """The same scenario must resolve the same way at three integration steps.

    Stated as agreement BETWEEN timesteps rather than against a fixed number, so
    the test measures the invariance property itself and does not have to be
    rewritten whenever modeled behavior legitimately changes.
    """
    rates = {}
    n = 20
    for dt in (0.05, 0.025, 0.01):
        kills = 0
        for i in range(n):
            res, _, _, _ = run_engagement(n_drones=3, dt=dt,
                                          sam_offset_angle=np.radians(np.linspace(-25, 25, n)[i]),
                                          seed=[3, i])
            kills += not res.mothership_survived
        rates[dt] = kills / n

    spread = max(rates.values()) - min(rates.values())
    assert spread <= 0.15, f"outcome depends on integration step: {rates}"


def test_same_seed_reproduces_exactly():
    a = run_engagement(n_drones=3, sam_offset_angle=0.2, seed=[4, 4])[0]
    b = run_engagement(n_drones=3, sam_offset_angle=0.2, seed=[4, 4])[0]
    assert a.mothership_survived == b.mothership_survived
    assert a.sam_outcome == b.sam_outcome
    assert a.min_missile_miss_distance == pytest.approx(b.min_missile_miss_distance)


def test_seeding_one_trial_does_not_disturb_another():
    """Independent streams: running an unrelated engagement in between must not
    change a later result. The old global np.random.seed() broke this."""
    first = run_engagement(n_drones=2, sam_offset_angle=0.1, seed=[5, 1])[0]
    run_engagement(n_drones=5, sam_offset_angle=-0.3, seed=[5, 2])
    again = run_engagement(n_drones=2, sam_offset_angle=0.1, seed=[5, 1])[0]
    assert first.sam_outcome == again.sam_outcome
    assert first.min_missile_miss_distance == pytest.approx(again.min_missile_miss_distance)


def test_pn_command_respects_the_g_limit_in_every_direction():
    """Per-axis clipping would allow up to sqrt(3) times the stated limit along
    a diagonal; the magnitude clamp must not."""
    sam = GenericSAM("t", 0.0, 0.0, 0.0, max_accel_g=25.0)
    sam.vel = np.array([900.0, 0.0, 0.0])
    rng_state = np.random.default_rng(0)
    for _ in range(200):
        target_pos = rng_state.normal(scale=2000.0, size=3)
        target_vel = rng_state.normal(scale=250.0, size=3)
        a = sam._pn_command(target_pos, target_vel)
        assert np.linalg.norm(a) <= sam.max_accel + 1e-6


def test_lead_pursuit_solves_the_collision_triangle():
    """A 260 m/s drone must be able to body-block a 900 m/s missile closing
    head-on from 20 km. The previous lead law missed by 959 m, which made the
    INTERCEPT role inert and forced drone attrition to zero in every trial."""
    from aircraft import Vehicle, Autopilot

    sam_pos = np.array([20000.0, 0.0, 6000.0])
    sam_vel = np.array([-900.0, 0.0, 0.0])
    drone = Vehicle("d", 0.0, 0.0, 6000.0, 0.0, 230.0,
                    max_bank=np.radians(85), max_g=12.0,
                    roll_rate_limit=np.radians(300), max_accel=20.0,
                    min_speed=40.0, max_speed=260.0)

    dt, t, best = 0.05, 0.0, np.inf
    while t < 30.0:
        b, g, a = Autopilot.intercept_lead(drone, sam_pos, sam_vel)
        drone.step(dt, b, g, a)
        sam_pos = sam_pos + sam_vel * dt
        t += dt
        best = min(best, float(np.linalg.norm(drone.pos() - sam_pos)))
    assert best < 30.0, f"interceptor closest approach {best:.1f} m exceeds the 30 m fuze radius"


def test_lead_pursuit_degrades_gracefully_when_uncatchable():
    """A target running away faster than the interceptor has no collision
    solution; the command must still be finite and point at the target."""
    from aircraft import Vehicle, Autopilot

    drone = Vehicle("d", 0.0, 0.0, 6000.0, 0.0, 230.0, max_speed=260.0)
    bank, gamma, accel = Autopilot.intercept_lead(
        drone, np.array([5000.0, 0.0, 6000.0]), np.array([900.0, 0.0, 0.0]))
    assert all(np.isfinite(v) for v in (bank, gamma, accel))


def test_mothership_velocity_matches_finite_difference():
    """The analytic world-frame velocity must agree with the trajectory the
    integrator actually produces."""
    dt = 0.01
    mo = F16Mothership("F16", x=0.0, y=0.0, z=6000.0, heading=0.35, v=230.0, step=dt)
    for _ in range(5):
        p0, v_analytic = mo.pos(), mo.velocity_vector()
        mo.step(dt)
        v_fd = (mo.pos() - p0) / dt
        assert np.linalg.norm(v_analytic - v_fd) < 0.05      # O(dt) difference only
        assert np.linalg.norm(v_analytic) == pytest.approx(230.0, rel=1e-3)
