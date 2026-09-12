"""
lethality.py
------------
Endgame geometry and damage functions.

This module exists because the original engagement model checked for a hit by
sampling ``range < lethal_radius`` once per integration step. At a 900 m/s
missile speed against a 230 m/s target, one 0.05 s step advances the geometry
by roughly 56 m, so a 15 m lethal radius was about a quarter of a single step:
whether a target died depended on where the sample grid happened to land
relative to closest approach, not on the physics. Measured effect, zero drones
and zero countermeasures against a non-maneuvering target: the kill/no-kill
outcome flipped on half a degree of launch geometry, and a 2.5 deg offset sweep
recorded 0 kills in 21 shots.

The fix is to stop sampling range and start solving for it.

Closest point of approach (CPA)
-------------------------------
Over one step the relative motion is treated as linear (the same assumption the
integrator already makes). With relative position ``r0`` and relative velocity
``v`` held constant across the step,

    r(t) = r0 + v t                    0 <= t <= dt

and |r(t)|^2 is a convex quadratic in t, minimized at

    t* = -(r0 . v) / (v . v)

Clamping t* to [0, dt] and evaluating |r(t*)| gives the true miss distance for
that step regardless of how coarse dt is. When the unclamped t* falls strictly
inside the step, the pass is a fly-by: the range was decreasing at the start of
the step and increasing at the end, which is exactly the condition a proximity
fuze detects.

Damage function
---------------
Miss distance alone does not decide a kill. Lethality is modeled with the
standard Gaussian ("Carleton") damage function

    Pk(m) = Pk_max * exp(-ln2 * (m / r50)^2)

parameterized so that ``r50`` is the miss distance at which kill probability
falls to half of its maximum. This is generic, published, textbook material and
is not calibrated to any real warhead.

Rate vs. per-step probability
-----------------------------
Any "probability that X happens this tick" is a timestep-dependent quantity: an
0.35 per-tick probability applied 20 times a second is a near-certainty within
one second, and halving dt silently halves the effect. Effects that occur
continuously in time are specified as a hazard rate (per second) and converted
to a per-step probability by ``hazard_to_step_probability``, which makes results
invariant to the integration step.
"""

import numpy as np

LN2 = np.log(2.0)


def closest_approach(rel_pos, rel_vel, dt):
    """Solve for the closest point of approach over one step of length ``dt``.

    Parameters
    ----------
    rel_pos : (3,) array   target position minus interceptor position, at t=0
    rel_vel : (3,) array   target velocity minus interceptor velocity
    dt      : float        step length (s)

    Returns
    -------
    t_star : float   time within [0, dt] of closest approach
    miss   : float   range at t_star (m)
    flyby  : bool    True if the unclamped minimum falls strictly inside the
                     step, i.e. the range stopped decreasing and started
                     increasing during this step -- the proximity-fuze trigger.
    """
    rel_pos = np.asarray(rel_pos, dtype=float)
    rel_vel = np.asarray(rel_vel, dtype=float)

    vv = float(rel_vel @ rel_vel)
    if vv < 1e-12:
        # No relative motion: range is constant across the step.
        return 0.0, float(np.linalg.norm(rel_pos)), False

    t_unclamped = -float(rel_pos @ rel_vel) / vv
    t_star = min(max(t_unclamped, 0.0), dt)
    miss = float(np.linalg.norm(rel_pos + rel_vel * t_star))
    flyby = 0.0 < t_unclamped < dt
    return t_star, miss, flyby


def probability_of_kill(miss_distance, r50, pk_max=0.9):
    """Gaussian (Carleton) damage function.

    ``r50`` is the miss distance at which kill probability is half of ``pk_max``.
    Generic parameterization; not calibrated to any real warhead.
    """
    if r50 <= 0.0:
        return 0.0
    return float(pk_max * np.exp(-LN2 * (miss_distance / r50) ** 2))


def hazard_to_step_probability(rate_per_second, dt):
    """Convert a continuous-time hazard rate to a per-step probability.

        p = 1 - exp(-lambda * dt)

    Guarantees that the modeled effect over a fixed span of wall-clock time is
    the same whether the sim runs at dt=0.05 or dt=0.005.
    """
    if rate_per_second <= 0.0:
        return 0.0
    return float(1.0 - np.exp(-rate_per_second * dt))


def velocity_of(entity, fallback=None):
    """Best-available world-frame velocity vector (m/s) for a sim entity.

    Vehicles expose ``velocity_vector()``; the missile carries ``vel`` directly.
    ``fallback`` is returned when neither is available.
    """
    if hasattr(entity, "velocity_vector"):
        return np.asarray(entity.velocity_vector(), dtype=float)
    if hasattr(entity, "vel"):
        return np.asarray(entity.vel, dtype=float)
    if fallback is not None:
        return np.asarray(fallback, dtype=float)
    return np.zeros(3)
