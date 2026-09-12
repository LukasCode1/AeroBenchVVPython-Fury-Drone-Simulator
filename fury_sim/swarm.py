"""
swarm.py
--------
Fury-style escort drone agents and swarm-level tactics.

Each drone can be assigned one of several roles each tick by a
(programmable, or AI-driven) swarm controller:

    ESCORT    - hold a formation slot relative to the mothership
    DECOY     - fly a track designed to present a more attractive /
                closer-range target to an inbound missile's seeker,
                pulling its lock away from the mothership
    JAMMER    - fly within an effective "denial radius" of a threat's
                seeker to raise the probability the seeker loses/denies lock
    INTERCEPT - fly a lead-pursuit intercept path toward the inbound
                missile itself (kinetic body-block / hard-kill concept)

The role-assignment policy itself is pluggable: `swarm_policy_manual` is a
simple hand-coded heuristic; `swarm_policy_ai_hook` shows where you'd plug
in a learned (RL) or LLM-driven policy instead -- e.g. AeroBenchVV-style
controllers, or a policy trained with an RL library, could replace this
function directly.
"""

import numpy as np
from aircraft import Vehicle, Autopilot
from lethality import hazard_to_step_probability

ROLE_ESCORT = "ESCORT"
ROLE_DECOY = "DECOY"
ROLE_JAMMER = "JAMMER"
ROLE_INTERCEPT = "INTERCEPT"


class Drone(Vehicle):
    def __init__(self, name, x, y, z, heading, v=180.0):
        super().__init__(name, x, y, z, heading, v,
                          max_bank=np.radians(85), max_g=12.0,
                          roll_rate_limit=np.radians(300), max_accel=20.0,
                          min_speed=40.0, max_speed=260.0)
        self.role = ROLE_ESCORT
        self.jam_power = 1.0     # relative jamming/decoy effectiveness, 0-1 scale tunable per experiment


def formation_slot(mothership, index, n, spacing=250.0):
    """Simple trailing-wedge formation slot position, offset behind/beside mothership."""
    side = 1 if index % 2 == 0 else -1
    rank = (index // 2) + 1
    # body-relative offset rotated into world frame by mothership heading
    dx_body = -spacing * rank
    dy_body = side * spacing * 0.6 * rank
    hx, hy = np.cos(mothership.heading), np.sin(mothership.heading)
    ox, oy = -hy, hx  # perpendicular
    x = mothership.x + dx_body * hx + dy_body * ox
    y = mothership.y + dx_body * hy + dy_body * oy
    z = mothership.z
    return np.array([x, y, z])


def swarm_policy_manual(drones, mothership, threats):
    """
    Hand-coded heuristic policy for role assignment each tick:
      - No active threats -> all ESCORT.
      - Threat present -> assign nearest drone(s) to DECOY/JAMMER/INTERCEPT
        based on geometry, remaining drones hold ESCORT.
    Returns dict: drone.name -> role
    """
    roles = {d.name: ROLE_ESCORT for d in drones}
    live_threats = [t for t in threats if getattr(t, "alive", True) and not getattr(t, "detonated", False)]
    if not live_threats or not drones:
        return roles

    threat = min(live_threats, key=lambda t: np.linalg.norm(t.pos - mothership.pos()))
    live_drones = [d for d in drones if d.alive]
    if not live_drones:
        return roles

    # rank drones by distance to the threat's projected path (proxy: distance to threat)
    ranked = sorted(live_drones, key=lambda d: np.linalg.norm(d.pos() - threat.pos))

    if len(ranked) >= 1:
        roles[ranked[0].name] = ROLE_INTERCEPT
    if len(ranked) >= 2:
        roles[ranked[1].name] = ROLE_JAMMER
    if len(ranked) >= 3:
        roles[ranked[2].name] = ROLE_DECOY
    return roles


def swarm_policy_ai_hook(drones, mothership, threats, model_fn=None):
    """
    Placeholder for an AI/RL-driven policy. `model_fn` should be a callable:
        model_fn(state_dict) -> {drone_name: role}
    where state_dict packages positions/velocities/threat geometry however
    your trained policy expects. If model_fn is None, falls back to the
    manual heuristic so the sim still runs.
    """
    if model_fn is None:
        return swarm_policy_manual(drones, mothership, threats)

    state = {
        "mothership": {"pos": mothership.pos().tolist(), "heading": mothership.heading, "v": mothership.v},
        "drones": [{"name": d.name, "pos": d.pos().tolist(), "v": d.v, "alive": d.alive} for d in drones],
        "threats": [{"name": t.name, "pos": t.pos.tolist(), "vel": t.vel.tolist()} for t in threats
                    if getattr(t, "alive", True)],
    }
    return model_fn(state)


def command_drone(drone, role, mothership, threat, index, n_drones):
    """Convert a role into a (bank, gamma, accel) autopilot command for this tick."""
    if role == ROLE_ESCORT or threat is None:
        slot = formation_slot(mothership, index, n_drones)
        return Autopilot.fly_to_point(drone, slot, desired_speed=mothership.v)

    if role == ROLE_DECOY:
        # Fly toward the threat's expected seeker cone but offset from the
        # mothership, presenting a closer/louder target to pull lock away.
        lure_point = threat.pos + (mothership.pos() - threat.pos) * 0.15
        return Autopilot.fly_to_point(drone, lure_point, desired_speed=drone.max_speed)

    if role == ROLE_JAMMER:
        # Hold a position between threat and mothership, within denial radius.
        mid = threat.pos + (mothership.pos() - threat.pos) * 0.5
        return Autopilot.fly_to_point(drone, mid, desired_speed=drone.max_speed * 0.8)

    if role == ROLE_INTERCEPT:
        return Autopilot.intercept_lead(drone, threat.pos, threat.vel)

    slot = formation_slot(mothership, index, n_drones)
    return Autopilot.fly_to_point(drone, slot, desired_speed=mothership.v)


def jam_probability_fn(drones, roles, dt, denial_radius=1200.0,
                       jammer_rate_hz=8.6, decoy_rate_hz=13.9):
    """
    Builds the jam_prob_fn passed to GenericSAM.acquire(). Returns a function
    of the candidate target giving the probability that jamming or decoy pull
    denies seeker lock during a step of length ``dt``.

    Countermeasure effectiveness is specified as a HAZARD RATE (per second),
    not as a per-tick probability. The earlier model applied a fixed ~0.35
    probability every tick: at dt=0.05 that is 20 rolls per second, so lock was
    denied within about a second regardless of geometry, and halving the
    timestep silently halved the modeled effect. Rates make the result
    invariant to dt -- see lethality.hazard_to_step_probability.

    Concurrent jammers and decoys are competing risks, so their rates add and
    the sum is converted to a probability once.

    Default rates are the values that reproduce the legacy per-tick numbers at
    the legacy dt=0.05 (-ln(1-0.35)/0.05 = 8.6 Hz, -ln(1-0.5)/0.05 = 13.9 Hz),
    so behavior is unchanged at the original step size while becoming
    step-invariant. They are placeholders pending real calibration, not
    measurements of any jamming technique.
    """
    def fn(candidate_target, threat_pos):
        total_rate = 0.0
        for d in drones:
            if not d.alive:
                continue
            role = roles.get(d.name)
            if role == ROLE_JAMMER:
                radius = denial_radius
                rate = jammer_rate_hz
            elif role == ROLE_DECOY:
                radius = denial_radius * 1.5
                rate = decoy_rate_hz
            else:
                continue
            dist = np.linalg.norm(d.pos() - threat_pos)
            if dist < radius:
                total_rate += rate * d.jam_power * (1 - dist / radius)
        return hazard_to_step_probability(total_rate, dt)
    return fn
