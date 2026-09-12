# Verification and validation

What has been checked, how, and what the measured numbers were. Every result
below is reproduced by `python -m pytest tests/ -q` unless noted otherwise.

Terminology follows the usual split: **verification** asks whether the model was
built right (does the code implement the intended equations), **validation**
asks whether the right model was built (does the behavior match an independent
reference). No accreditation is claimed — this is a research testbed, not an
accredited analysis tool.

---

## 1. The defect this document exists because of

The engagement model originally resolved hit/miss by sampling
`range < lethal_radius` once per integration step. At 900 m/s missile speed
against a 230 m/s target, one 0.05 s step advances the geometry by ~56 m, so the
15 m lethal radius was roughly a quarter of a single step.

**Measured before the fix** — zero drones, zero countermeasures, perfectly
guided missile, non-manoeuvring target:

| Launch offset | Internal min range | Outcome |
|---|---|---|
| +20.0° | 16.74 m | survives |
| +21.5° | under 15 m | killed |
| +22.5° | 45.30 m | survives |

The outcome flipped on half a degree of launch geometry. A 2.5°-spaced sweep
from −25° to +25° recorded **0 kills in 21 shots**; random continuous offsets
produced ~30% kills. That ~30% was the "SAM lethality" every published number in
this project rested on, and it was sampling phase, not physics.

**After the fix**, the same unopposed sweep gives closest approach < 1 m on every
shot and a kill rate that matches the damage function. Both are asserted in
`tests/test_engagement.py`.

---

## 2. Verification

### 2.1 Closest-approach solver

`lethality.closest_approach` minimizes |r₀ + v t|² over the step analytically.
Checked against cases with known answers in `tests/test_lethality.py`:

| Case | Expected | Result |
|---|---|---|
| Head-on pass, 7 m lateral offset | miss = 7.000 m | exact to 1e-9 |
| Still closing at end of step | clamps to end-of-step range, not flagged a fly-by | pass |
| Zero relative motion | miss = \|r₀\|, no fly-by | pass |
| 2 m pass, both sampled ranges > 500 m | fly-by detected, miss = 2.000 m | pass |

The last row is the regression test for §1: range-sampling sees nothing at either
step boundary, the solver reports 2 m and a lethal Pk.

### 2.2 Damage function

Gaussian (Carleton) form, `Pk(m) = Pk_max · exp(−ln2 · (m/r50)²)`. Verified that
`Pk(0) = Pk_max`, `Pk(r50) = Pk_max/2`, and that it decays monotonically to
< 1e-3 by 100 m.

### 2.3 Timestep invariance

Any effect expressed as a per-tick probability silently scales with the
integration step. Countermeasures are now hazard rates converted by
`p = 1 − exp(−λ·dt)`.

- A 5 Hz hazard reproduces `exp(−5)` survival over 1 s of modeled time at
  dt = 0.5, 0.05, 0.005 and 0.001, to a relative tolerance of 1e-9.
- The replaced behavior is pinned by a test that *asserts the old scheme was not
  invariant*, so the regression is visible rather than merely absent.
- End-to-end: the same 3-drone scenario resolves to a **0.900 kill rate at
  dt = 0.05, 0.025 and 0.01**.

### 2.4 Acceleration limiting

The PN command previously used `np.clip` per axis, which permits up to √3 times
the stated g-limit along a diagonal. The magnitude clamp is checked over 200
randomized geometries: ‖a_cmd‖ ≤ 25 g in all of them.

### 2.5 Lead pursuit / collision triangle

`Autopilot.intercept_lead` previously estimated time-to-go as
`range / own_speed` and then halved the resulting lead. Against a 900 m/s
missile that made t_go about five times too large before halving, so the aim
point sat kilometres beyond the target.

Measured, head-on intercept from 20 km, 260 m/s drone against a 900 m/s missile:

| Guidance | Closest approach |
|---|---|
| Previous (`range / own_speed`, lead halved) | 959.3 m |
| Collision-triangle solution | 9.5 m |

The fuze radius is 30 m, so the first never intercepts and the second does. This
mattered beyond one role: with the INTERCEPT drone unable to reach the missile,
modeled drone attrition was **structurally zero in all 420 trials**, which is
why the earlier README could report "no drone was lost in any trial" as if it
were a finding. It was an artifact of a guidance bug.

The corrected law is verified against the closed-form quadratic root and against
an uncatchable geometry (target faster and opening), where it must degrade to
pure pursuit without producing non-finite commands.

### 2.6 Reproducibility

Each engagement owns an explicit `numpy.random.Generator` addressed by
`[MASTER_SEED, n_drones, trial]`. Verified that the same seed reproduces the
outcome exactly, and that running an unrelated engagement in between does not
perturb a later result — which the previous global `np.random.seed()` could not
guarantee.

Checked against the committed sweep by re-running six trials individually, out
of order, from their stream coordinates alone:

| n_drones | trial | survived | drones lost | miss (m) | vs. CSV |
|---|---|---|---|---|---|
| 0 | 0 | False | 0 | 0.000004 | match |
| 1 | 17 | True | 1 | 5.667852 | match |
| 3 | 42 | True | 1 | 6.882022 | match |
| 4 | 7 | True | 1 | 7.594835 | match |
| 8 | 59 | True | 1 | 7.377848 | match |
| 6 | 23 | True | 0 | 6.882864 | match |

Any row of `output/experiment_results.csv` can therefore be re-derived and
debugged on its own without reproducing the whole sweep.

---

## 3. Validation

### 3.1 Mothership velocity against the integrator

`F16Mothership.velocity_vector()` rotates body-axis velocity into the navigation
frame using the same direction cosine matrix AeroBenchVV's own equations of
motion use for the POSN/POSE/ALT derivatives. Compared against a finite
difference of the trajectory the integrator actually produced:

```
 t    | analytic (m/s)            | finite difference         | error
 0.00 |  216.06   78.87    0.00   |  216.05   78.87    0.02   | 0.017
 0.05 |  216.02   78.85    0.17   |  216.01   78.85    0.18   | 0.017
```

Residual 0.017 m/s on a 230 m/s state vector (0.007%), and constant — i.e. it is
the O(dt) truncation of the forward difference, not a frame or unit error.
Speed and heading both recover their commanded values (‖v‖ = 230.0 m/s,
atan2(vy, vx) = 0.3499 rad against a commanded 0.35).

### 3.2 Proportional navigation against published theory

The defining property of PN is that against a non-manoeuvring target it drives
the line-of-sight rate to zero, putting the missile on a constant-bearing
collision course; classical analysis also holds that the LOS rate only nulls for
N′ > 2, with N′ = 2 marginal. Measured over a 20 km flyout against a 230 m/s
non-manoeuvring target:

| N′ | LOS rate, start → end (rad/s) | Miss distance |
|---|---|---|
| 2 | 0.00347 → 0.00010 | 0.000 m |
| 3 | 0.00347 → 0.00000 | 0.000 m |
| 4 | 0.00347 → 0.00000 | 0.000 m |
| 5 | 0.00347 → 0.00000 | 0.000 m |

N′ = 2 converges an order of magnitude more slowly than N′ ≥ 3, as theory
predicts. Reference: Zarchan, *Tactical and Strategic Missile Guidance*.

### 3.3 Lethality calibration

With miss distance driven to ~0 by §3.2, realized kill rate must equal `pk_max`.
Over 120 unopposed shots: **0.892 measured against 0.900 expected** (0.3 standard
errors). The damage function is wired correctly end to end.

---

## 4. Known limitations

These are modeling choices, stated so they are not mistaken for validated
behavior.

- **Miss distance has no spread.** With perfect seeker information, no guidance
  lag and a target that never manoeuvres, PN produces a near-zero miss on every
  shot, so the damage function always evaluates at Pk_max. Seeker noise, target
  manoeuvre and guidance lag are what generate real miss-distance distributions
  and none are modeled. The damage function is therefore *correct but not yet
  exercised*; it will matter as soon as the target defends.
- **The missile has no energy model.** Constant speed, no drag, no motor
  burn/coast, no energy bleed in turns — bounded only by `max_flight_time`.
- **Countermeasure rates are placeholders.** The 8.6 Hz jammer and 13.9 Hz decoy
  hazards were chosen to reproduce the previous per-tick values at the previous
  timestep, not measured from anything.
- **Single-target burst.** No fragment pattern, no multiple kills per warhead.
- **The drones use a reduced-order flight model**, so their manoeuvring limits
  are approximate and not derived from a specific airframe.
- **Nothing here represents a real system.** Guidance law and aerodynamic model
  are published textbook material; seeker, warhead and countermeasure parameters
  are generic and uncalibrated.
