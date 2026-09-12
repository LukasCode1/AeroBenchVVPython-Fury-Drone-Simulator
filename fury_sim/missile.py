"""
missile.py
----------
Generic surface-to-air missile (SAM) model using Proportional Navigation (PN),
the standard, publicly-published guidance law:

    a_cmd = N' * Vc * (d/dt) u_los

where u_los is the unit line-of-sight vector, Vc is closing velocity, and N' is
the navigation constant (typically 3-5). This is textbook material (see Zarchan,
"Tactical and Strategic Missile Guidance") and is NOT modeling any specific
real-world seeker, airframe, or classified parameter set.

The missile also has a generic "seeker" that can be defeated probabilistically
by drone countermeasures (decoy lure, RF/IR jamming, or a physical body-block
that spoofs closing geometry). These are abstracted as hazard rates, not real
electronic-warfare techniques.

Endgame model
-------------
Hit/miss is resolved by solving for the closest point of approach within each
step rather than by sampling range at step boundaries -- see lethality.py for
why that distinction decides every result this simulation produces. The fuze:

  * is armed after ``arm_time`` seconds of flight;
  * triggers on the first fly-by (range minimum) inside ``fuze_arm_range`` of
    ANY live candidate, not only the tracked target, so a drone that puts
    itself in the missile's path is exposed to the burst;
  * rolls a Gaussian damage function against the true CPA miss distance.

The warhead is expended on trigger whether or not the roll kills, so a missile
resolves to exactly one of: killed something, missed, or ran out of energy.

Known simplifications (deliberate, not oversights): constant speed with no drag,
no motor burn/coast phases, and no energy bleed in turns -- bounded instead by a
hard ``max_flight_time``. Single-target burst; no fragment pattern.
"""

import numpy as np

from lethality import closest_approach, probability_of_kill, velocity_of

G = 9.81


class GenericSAM:
    def __init__(self, name, x, y, z, v=900.0, nav_constant=4.0,
                 max_accel_g=25.0, seeker_fov=np.radians(60), seeker_range=25000.0,
                 r50=12.0, pk_max=0.9, fuze_arm_range=30.0, arm_time=1.0,
                 max_flight_time=60.0, endgame_range=3000.0,
                 max_substep_distance=5.0, max_substeps=64, rng=None):
        self.name = name
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.speed = v
        self.N = nav_constant
        self.max_accel = max_accel_g * G
        self.seeker_fov = seeker_fov
        self.seeker_range = seeker_range

        # Endgame / lethality parameters (generic, uncalibrated)
        self.r50 = r50
        self.pk_max = pk_max
        self.fuze_arm_range = fuze_arm_range
        self.arm_time = arm_time
        self.max_flight_time = max_flight_time

        # Endgame integration refinement
        self.endgame_range = endgame_range
        self.max_substep_distance = max_substep_distance
        self.max_substeps = max_substeps

        self.rng = rng if rng is not None else np.random.default_rng()

        self.target = None              # currently locked Vehicle-like object
        self.locked = False
        self.alive = True
        self.detonated = False
        self.miss_distance = None       # CPA of the burst, once it happens
        self.min_miss_distance = np.inf  # best CPA seen against any candidate
        self.flight_time = 0.0
        self.killed_name = None
        self.outcome = None             # kill | miss | energy_exhausted | None
        self.history = []

        self._candidates = []

    # ----------------------------------------------------------------- seeker

    def acquire(self, candidates, jam_prob_fn=None):
        """
        candidates: list of vehicles (aircraft/drones) with .pos() and alive
        jam_prob_fn: optional fn(candidate) -> probability this candidate's
                     jamming/decoy denies lock this tick
        Chooses nearest valid target within seeker range/FOV; can be denied by
        jamming or pulled off by a decoy.

        The candidate list is retained for fuze evaluation, so the warhead can
        engage whatever it actually passes close to, not only what it tracks.
        """
        self._candidates = [c for c in candidates if getattr(c, "alive", True)]

        best = None
        best_rng = np.inf
        for c in self._candidates:
            rel = c.pos() - self.pos
            rng = np.linalg.norm(rel)
            if rng > self.seeker_range:
                continue
            los_dir = rel / max(rng, 1e-6)
            fwd = self.vel / max(np.linalg.norm(self.vel), 1e-6) if np.linalg.norm(self.vel) > 1 else los_dir
            ang = np.arccos(np.clip(np.dot(los_dir, fwd), -1, 1))
            if ang > self.seeker_fov / 2:
                continue
            if rng < best_rng:
                best_rng = rng
                best = c

        if best is None:
            self.locked = False
            self.target = None
            return

        # Countermeasure roll: jam/decoy can deny or redirect lock
        if jam_prob_fn is not None:
            p_deny = jam_prob_fn(best)
            if self.rng.random() < p_deny:
                self.locked = False
                self.target = None
                return

        self.target = best
        self.locked = True

    # --------------------------------------------------------------- guidance

    def _pn_command(self, target_pos, target_vel):
        """True proportional navigation acceleration command (m/s^2).

        Closing velocity is the projection of relative velocity onto the LOS,
        Vc = -(r . v_rel)/|r| -- positive while closing. The original model used
        the scalar sum of the two speeds regardless of geometry.

        The LOS rate is taken analytically as the component of relative velocity
        perpendicular to the LOS divided by range, rather than by
        finite-differencing the unit LOS vector across steps.
        """
        rel = target_pos - self.pos
        rng = float(np.linalg.norm(rel))
        if rng < 1e-6:
            return np.zeros(3)
        los = rel / rng

        v_rel = target_vel - self.vel
        closing_speed = -float(rel @ v_rel) / rng

        los_rate = (v_rel - float(v_rel @ los) * los) / rng
        a_cmd = self.N * closing_speed * los_rate

        # Limit the MAGNITUDE of lateral acceleration. Clipping each axis
        # independently would permit up to sqrt(3) times the stated g-limit
        # along a diagonal.
        mag = float(np.linalg.norm(a_cmd))
        if mag > self.max_accel:
            a_cmd = a_cmd * (self.max_accel / mag)
        return a_cmd

    def _substep_count(self, dt):
        """Refine the integration step in the endgame.

        Away from any target one step is fine. Inside ``endgame_range`` the
        missile is turning hard and the miss distance is set by geometry a
        0.05 s step cannot resolve, so the step is subdivided until the missile
        advances no more than ``max_substep_distance`` per sub-step.
        """
        if not self._candidates:
            return 1
        rng = min(float(np.linalg.norm(c.pos() - self.pos)) for c in self._candidates)
        if rng > self.endgame_range:
            return 1
        travel = self.speed * dt
        n = int(np.ceil(travel / max(self.max_substep_distance, 1e-6)))
        return int(np.clip(n, 1, self.max_substeps))

    # ------------------------------------------------------------------- step

    def step(self, dt):
        if not self.alive or self.detonated:
            return

        if self.flight_time >= self.max_flight_time:
            self.alive = False
            self.outcome = self.outcome or "energy_exhausted"
            return

        # Freeze candidate kinematics for the duration of this step; each is
        # linearly extrapolated across sub-steps, the same assumption the CPA
        # solution makes.
        tracked = self.target if (self.locked and self.target is not None
                                  and getattr(self.target, "alive", True)) else None
        frozen = [(c, np.array(c.pos(), dtype=float), velocity_of(c))
                  for c in self._candidates if getattr(c, "alive", True)]

        n = self._substep_count(dt)
        h = dt / n

        for k in range(n):
            tau = k * h

            if tracked is not None:
                entry = next((e for e in frozen if e[0] is tracked), None)
                if entry is not None:
                    t_pos = entry[1] + entry[2] * tau
                    a_cmd = self._pn_command(t_pos, entry[2])
                    new_vel = self.vel + a_cmd * h
                    speed = float(np.linalg.norm(new_vel))
                    if speed > 1e-6:
                        # Constant-speed turn: PN acceleration steers, it does
                        # not accelerate. Energy loss is not modeled.
                        self.vel = new_vel / speed * self.speed

            prev_pos = self.pos.copy()
            self.pos = self.pos + self.vel * h
            self.flight_time += h

            # --- fuze: solve CPA against every live candidate over this sub-step
            armed = self.flight_time >= self.arm_time
            best = None
            for c, c_pos0, c_vel in frozen:
                if not getattr(c, "alive", True):
                    continue
                rel_pos = (c_pos0 + c_vel * tau) - prev_pos
                rel_vel = c_vel - self.vel
                _, miss, flyby = closest_approach(rel_pos, rel_vel, h)
                if miss < self.min_miss_distance:
                    self.min_miss_distance = miss
                if armed and flyby and miss <= self.fuze_arm_range:
                    if best is None or miss < best[1]:
                        best = (c, miss)

            if best is not None:
                self._detonate(*best)
                return

            if self.flight_time >= self.max_flight_time:
                self.alive = False
                self.outcome = "energy_exhausted"
                return

    def _detonate(self, victim, miss):
        """Expend the warhead against ``victim`` at CPA distance ``miss``."""
        self.detonated = True
        self.alive = False
        self.miss_distance = miss

        pk = probability_of_kill(miss, self.r50, self.pk_max)
        if self.rng.random() < pk:
            victim.alive = False
            self.killed_name = victim.name
            self.outcome = "kill"
        else:
            self.outcome = "miss"

    def log(self, t):
        self.history.append((t, *self.pos, self.locked, self.target.name if self.target else None))
