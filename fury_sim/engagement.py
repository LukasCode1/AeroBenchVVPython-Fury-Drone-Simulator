"""
engagement.py
-------------
Ties the mothership (flown with AeroBenchVV's actual 13-state nonlinear
F-16 model, see f16_mothership.py), drone swarm, and a generic SAM threat
together into a single timestepped engagement, and records outcome metrics
for research/statistics.

Randomness
----------
Each engagement owns an explicit ``numpy.random.Generator``. The earlier model
called ``np.random.seed()``, which mutates global process state: trials could
not be reproduced independently, seeding one trial perturbed every later one,
and the Monte Carlo harness passed ``seed=None`` so nothing was reproducible at
all. Callers now pass either a seed (any value accepted by ``default_rng``,
including a list used as a stream coordinate) or a Generator directly.

Interception
------------
There is no longer a separate "body-block" rule. The earlier model destroyed the
SAM for free whenever an INTERCEPT drone came within 20 m of it -- a check that
could almost never fire, since the missile advanced ~45 m per step, and which
cost the drone nothing when it did. Interception is now resolved by the missile's
own proximity fuze (see missile.py): a drone that gets into the missile's path
forces the warhead to be expended, which defeats the shot, and is itself inside
the burst when it happens. Drone attrition is therefore a modeled outcome rather
than a value that was structurally always zero.
"""

import numpy as np
from f16_mothership import F16Mothership
from swarm import Drone, swarm_policy_manual, command_drone, jam_probability_fn
from missile import GenericSAM


class EngagementResult:
    def __init__(self):
        self.mothership_survived = None
        self.n_drones_lost = 0
        self.time_to_resolution = None
        self.min_missile_miss_distance = None
        self.sam_outcome = None      # kill | miss | energy_exhausted | unresolved
        self.sam_victim = None       # name of whatever the warhead killed, if any
        self.roles_over_time = []


def run_engagement(n_drones=3, sam_range=20000.0, sam_offset_angle=0.0,
                   dt=0.05, t_max=90.0, policy=swarm_policy_manual,
                   denial_radius=1200.0, seed=None, log_trajectories=False):
    """
    Runs one engagement:
      - mothership flies a straight/level ingress
      - a SAM launches from ground, sam_range meters away, offset by
        sam_offset_angle radians from nose-on
      - n_drones escort drones execute the swarm policy each tick
    Returns an EngagementResult.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    mothership = F16Mothership("F16", x=0.0, y=0.0, z=6000.0, heading=0.0, v=230.0, step=dt)

    drones = [Drone(f"drone_{i}", x=-200.0 - 50 * i, y=(-1)**i * 150.0, z=6000.0,
                    heading=0.0, v=230.0) for i in range(n_drones)]

    sam_x = sam_range * np.cos(sam_offset_angle)
    sam_y = sam_range * np.sin(sam_offset_angle)
    sam = GenericSAM("sam_1", x=sam_x, y=sam_y, z=0.0, rng=rng)
    # initial launch velocity aimed roughly at mothership
    aim = mothership.pos() - sam.pos
    sam.vel = aim / np.linalg.norm(aim) * sam.speed

    result = EngagementResult()
    t = 0.0

    while t < t_max:
        # 1. mothership: straight-and-level ingress, flown by AeroBenchVV's own
        # 13-state nonlinear F-16 model + waypoint autopilot (see f16_mothership.py)
        mothership.step(dt)

        # 2. swarm role assignment + commands
        threats = [sam] if sam.alive and not sam.detonated else []
        roles = policy(drones, mothership, threats)
        result.roles_over_time.append(dict(roles))

        threat = threats[0] if threats else None
        for i, d in enumerate(drones):
            if not d.alive:
                continue
            role = roles.get(d.name, "ESCORT")
            b, g, a = command_drone(d, role, mothership, threat, i, n_drones)
            d.step(dt, b, g, a)

        # 3. SAM seeker acquisition with countermeasures, then flyout. The
        # missile resolves its own hit/miss by solving for closest approach
        # inside the step, against every live candidate.
        if sam.alive and not sam.detonated:
            candidates = [mothership] + [d for d in drones if d.alive]
            jam_fn_builder = jam_probability_fn(drones, roles, dt, denial_radius=denial_radius)

            def jam_prob_fn(cand):
                return jam_fn_builder(cand, sam.pos)

            sam.acquire(candidates, jam_prob_fn=jam_prob_fn)
            sam.step(dt)

        if log_trajectories:
            mothership.log(t)
            for d in drones:
                d.log(t)
            sam.log(t)

        t += dt

        if sam.detonated or not sam.alive:
            break
        if not mothership.alive:
            break

    result.mothership_survived = mothership.alive
    result.n_drones_lost = sum(1 for d in drones if not d.alive)
    result.time_to_resolution = t
    result.min_missile_miss_distance = (None if not np.isfinite(sam.min_miss_distance)
                                        else float(sam.min_miss_distance))
    result.sam_outcome = sam.outcome if sam.outcome is not None else "unresolved"
    result.sam_victim = sam.killed_name
    return result, mothership, drones, sam
