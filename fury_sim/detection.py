"""
detection.py
------------
Mission-level detection and shot-allocation model, used to ask whether adding an
escort drone makes the manned aircraft *easier to find*.

This is the question the engagement-level model in `engagement.py` structurally
cannot answer. That model starts with a missile already in flight, so it can only
measure what happens once you are being shot at. The criticism most often made of
manned-unmanned teaming is upstream of that: a drone with a larger radar cross
section is detected further out, and its detection cues the defender onto a patch
of sky that the manned aircraft is also flying through.

Two effects pull in opposite directions:

  * **Cueing (bad).** Detecting the escort concentrates radar revisit on that
    bearing, which raises the manned aircraft's own detection probability if it
    is inside the cued sector.
  * **Shot absorption (good).** A detected escort is a target the battery can
    spend missiles on, and a magazine emptied into drones is not being emptied
    into the manned aircraft.

Which one wins depends on the escort's RCS and on how far it flies from the
aircraft it is protecting. Sweeping that is the point of `run_rcs_sweep.py`.

Detection model
---------------
Signal-to-noise ratio follows the radar range equation's dependence on target
size and range:

    SNR(sigma, R) = SNR_ref * (sigma / sigma_ref) * (R_ref / R)^4

so detection range scales as the fourth root of RCS -- a 10 dB reduction in RCS
buys roughly a 44% reduction in detection range, not a 90% one.

Single-scan detection probability uses the closed-form Swerling Case 1 result
for a slowly fluctuating (Rayleigh) target, which is the standard model for an
aircraft:

    Pd = Pfa ^ (1 / (1 + SNR))

Cumulative detection is then rolled per scan rather than computed analytically,
so the same Monte Carlo machinery and random streams apply as everywhere else.

Parameters
----------
Every number here is notional and drawn from open published ranges (e.g. fighter
1-10 m^2, cruise missile 0.1-1 m^2, stealth aircraft 0.001-0.01 m^2). Nothing is
calibrated to a real radar, a real battery, or a real aircraft. The single-shot
kill probability defaults to the value measured by this project's own
engagement-level model against an undefended, non-manoeuvring aircraft, which is
the one place the two model resolutions are tied together.
"""

import copy

import numpy as np

# Open-source notional RCS values (m^2), for reference and for labelling plots.
RCS_REFERENCE = {
    "large aircraft": 100.0,
    "fighter": 5.0,
    "cruise missile": 0.5,
    "small UAS": 0.05,
    "stealth aircraft": 0.005,
}


class SearchRadar:
    """Volume-search radar with cued revisit on established contacts."""

    def __init__(self, detection_range_ref_m=150_000.0, rcs_ref_m2=1.0,
                 p_fa=1e-6, scan_interval_s=10.0, cue_speedup=4.0,
                 cue_sector_deg=5.0):
        self.detection_range_ref = detection_range_ref_m
        self.rcs_ref = rcs_ref_m2
        self.p_fa = p_fa
        self.scan_interval = scan_interval_s
        self.cue_speedup = cue_speedup
        self.cue_sector = np.radians(cue_sector_deg)

        # Anchor the SNR scale so that Pd = 0.5 per scan at the reference range
        # against the reference RCS. Inverting the Swerling 1 expression:
        #     1 + SNR = ln(Pfa) / ln(Pd)
        self.snr_ref = np.log(p_fa) / np.log(0.5) - 1.0

    def snr(self, rcs_m2, range_m):
        if range_m <= 0:
            return np.inf
        return (self.snr_ref * (rcs_m2 / self.rcs_ref)
                * (self.detection_range_ref / range_m) ** 4)

    def p_detect_per_scan(self, rcs_m2, range_m):
        """Swerling Case 1 single-scan detection probability."""
        snr = self.snr(rcs_m2, range_m)
        if not np.isfinite(snr):
            return 1.0
        return float(self.p_fa ** (1.0 / (1.0 + snr)))

    def detection_range(self, rcs_m2):
        """Range at which Pd per scan is 0.5, i.e. the fourth-root scaling."""
        return self.detection_range_ref * (rcs_m2 / self.rcs_ref) ** 0.25


class Contact:
    """One radar-visible entity on the ingress."""

    def __init__(self, name, rcs_m2, cross_track_m=0.0, is_manned=False):
        self.name = name
        self.rcs = rcs_m2
        self.cross_track = cross_track_m
        self.is_manned = is_manned

        self.alive = True
        self.detected = False
        self.detected_at_range = None
        self.detected_at_time = None
        self.shots_taken = 0
        self.next_scan_t = 0.0


class SamBattery:
    """Magazine-limited battery that shoots what the radar has found.

    The battery starts cold. ``reaction_time_s`` is the delay between the first
    contact it holds on *anything* and its first shot at *anyone* -- acquisition,
    track establishment, identification and the launch decision.

    That delay is the mechanism by which an escort can hurt the aircraft it is
    escorting. A drone detected early does not just get itself shot at; it starts
    the defender's clock. The manned aircraft then arrives against a battery that
    is already alert rather than one still spinning up, and loses the head start
    its own low observability was supposed to buy.

    Pulling the other way, every missile spent on the escort is one not available
    for the manned aircraft. Which effect dominates is what the RCS sweep
    measures, and the answer depends on magazine depth.
    """

    def __init__(self, max_range_m=80_000.0, magazine=2, refire_interval_s=20.0,
                 reaction_time_s=75.0, single_shot_pk=0.87):
        self.max_range = max_range_m
        self.magazine = magazine
        self.refire_interval = refire_interval_s
        self.reaction_time = reaction_time_s
        self.pk = single_shot_pk
        self.next_shot_t = 0.0
        self.shots_fired = 0
        self.alerted_at = None

    def alert(self, t):
        """First contact on anything starts the reaction clock, once."""
        if self.alerted_at is None:
            self.alerted_at = t

    def can_fire(self, t):
        if self.shots_fired >= self.magazine or t < self.next_shot_t:
            return False
        if self.alerted_at is None:
            return False
        return t >= self.alerted_at + self.reaction_time


def run_ingress(manned_rcs_m2=0.005, escort_rcs_m2=None, escort_cross_track_m=0.0,
                escort_lead_m=0.0, start_range_m=120_000.0, release_range_m=40_000.0,
                speed_mps=230.0, dt=1.0, radar=None, battery=None, rng=None):
    """Fly a formation inbound and see what the defender manages to do about it.

    Geometry: the radar sits at the origin, the manned aircraft runs straight in
    along the threat axis from ``start_range_m`` to ``release_range_m``, and the
    escort flies ``escort_cross_track_m`` abeam of it (and optionally
    ``escort_lead_m`` ahead).

    Cross-track offset is the parameter that matters for cueing, and along-track
    lead is not a substitute for it: an escort flying directly ahead of the
    aircraft it is screening sits on the *same bearing* from the radar, so its
    angular separation is zero and any cue it generates lands squarely on the
    manned aircraft no matter how far forward it is.

    Returns a dict of outcome metrics for one trial.
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    radar = radar or SearchRadar()
    # The battery carries per-engagement state (magazine, refire clock), so a
    # caller-supplied one is copied rather than depleted across trials.
    battery = copy.deepcopy(battery) if battery is not None else SamBattery()

    manned = Contact("manned", manned_rcs_m2, is_manned=True)
    contacts = [manned]
    if escort_rcs_m2 is not None:
        contacts.append(Contact("escort", escort_rcs_m2,
                                cross_track_m=escort_cross_track_m))

    t = 0.0
    total_time = (start_range_m - release_range_m) / speed_mps

    while t <= total_time and manned.alive:
        manned_range = start_range_m - speed_mps * t

        # Positions in the radar's frame: manned on-axis, escort offset.
        along = {manned.name: manned_range}
        bearing = {manned.name: 0.0}
        ranges = {manned.name: max(manned_range, 1.0)}
        for c in contacts:
            if c.is_manned:
                continue
            x = max(manned_range - escort_lead_m, 1.0)
            ranges[c.name] = float(np.hypot(x, c.cross_track))
            bearing[c.name] = float(np.arctan2(c.cross_track, x))
            along[c.name] = x

        # --- radar: an established contact cues its own angular sector
        held = [c for c in contacts if c.detected and c.alive]

        for c in contacts:
            if not c.alive or c.detected:
                continue
            interval = radar.scan_interval
            if held:
                sep = min(abs(bearing[c.name] - bearing[h.name]) for h in held)
                if sep < radar.cue_sector:
                    interval = radar.scan_interval / radar.cue_speedup

            if t >= c.next_scan_t:
                p = radar.p_detect_per_scan(c.rcs, ranges[c.name])
                if rng.random() < p:
                    c.detected = True
                    c.detected_at_range = ranges[c.name]
                    c.detected_at_time = t
                    battery.alert(t)
                c.next_scan_t = t + interval

        # --- battery: engage the closest detected, live target in range
        if battery.can_fire(t):
            targets = [c for c in contacts
                       if c.detected and c.alive and ranges[c.name] <= battery.max_range]
            if targets:
                victim = min(targets, key=lambda c: ranges[c.name])
                battery.shots_fired += 1
                battery.next_shot_t = t + battery.refire_interval
                victim.shots_taken += 1
                if rng.random() < battery.pk:
                    victim.alive = False

        t += dt

    escort = next((c for c in contacts if not c.is_manned), None)
    return {
        "manned_survived": manned.alive,
        "manned_detected": manned.detected,
        "manned_detection_range_m": manned.detected_at_range,
        "manned_shots_taken": manned.shots_taken,
        "escort_detected": escort.detected if escort else None,
        "escort_detection_range_m": escort.detected_at_range if escort else None,
        "escort_survived": escort.alive if escort else None,
        "escort_shots_taken": escort.shots_taken if escort else 0,
        "shots_fired": battery.shots_fired,
    }
