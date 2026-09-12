# Escort swarm survivability study

A Monte Carlo testbed for a single question: does an autonomous escort drone
swarm improve the survivability of a manned strike aircraft against a
surface-to-air missile shot, and what does that protection cost in drones?

The manned aircraft (the "mothership") is flown with AeroBenchVV's actual
nonlinear F-16 flight dynamics model, so the ingress trajectory is the same
six-degree-of-freedom aircraft model used elsewhere in this repository, not a
simplified stand-in.

Structured as a study plan: purpose, scope, assumptions, measures of
effectiveness, method, results, limitations.

---

## 1. Purpose of study

Quantify the change in mothership survival probability produced by adding
escort drones in defined roles (decoy, jammer, hard-kill interceptor) against a
fixed missile threat and a fixed ingress profile, and identify the point of
diminishing returns as swarm size grows.

## 2. Scope and limitations

**In scope:** one manned aircraft, one SAM shot, 0-8 escort drones, a single
ingress geometry, engagement-level physics.

**Out of scope, and therefore not claimed:** detection and radar cross-section
effects, multi-shot salvos, adversary decision-making, basing and sortie
generation, operator workload, and cost. The survival numbers below say nothing
about whether an escort drone is *worth buying* — that requires a cost-exchange
metric this study does not yet compute.

No model here represents the seeker, airframe, warhead, or guidance parameters
of any real system. The guidance law and aerodynamic model are published
textbook material; all seeker, warhead and countermeasure parameters are
generic and uncalibrated.

## 3. Critical assumptions

| # | Assumption | Why it matters |
|---|---|---|
| A1 | The mothership flies straight and level and never defends | Removes the single most effective real countermeasure. Survival numbers are a lower bound. |
| A2 | Drones have perfect knowledge of missile position and velocity | No detection, tracking, or cueing latency is modeled, so intercept performance is an upper bound. |
| A3 | One SAM per engagement | No magazine depth, no saturation, no re-attack. |
| A4 | Missile flies at constant speed, no drag or motor burn | Bounded only by a hard flight-time limit. |
| A5 | Countermeasure effectiveness is a hazard rate, not a modeled waveform | Rates are placeholders chosen to reproduce prior behavior, not measurements. |

A1 and A2 are the two that most affect the headline result. Both are stated
again alongside the finding.

## 4. Measures of effectiveness

| MOE | Definition |
|---|---|
| Mothership survival rate | Fraction of trials in which the mothership is not destroyed, with a 95% Wilson score interval |
| Escort attrition | Mean drones lost per engagement |
| Miss distance | Closest point of approach between missile and whatever it bursts against |
| Shot resolution | Whether the SAM killed, burst without killing, or ran out of energy |

## 5. Method

| Entity | Dynamics model | Fidelity |
|---|---|---|
| Mothership | AeroBenchVV 13-state nonlinear F-16, LQR inner loop, waypoint outer loop | Full nonlinear ODE, `scipy.integrate.RK45` |
| Escort drones | Energy-state / bank-to-turn point mass | Reduced order, closed-form update |
| SAM | Proportional navigation (Zarchan), generic seeker with FOV, range and hazard-rate lock denial | Reduced order, textbook guidance law |

The mothership gets the higher-fidelity treatment deliberately: the study is
about the survivability of that one aircraft. Drones and missile use
reduced-order models so thousands of trials stay tractable (~0.1-0.25 s per
trial).

**Hit/miss is resolved by solving for closest approach inside each integration
step**, not by sampling range at step boundaries. That distinction decides every
number in this document — see [`VALIDATION.md`](VALIDATION.md) §1.

**Sampling:** 60 trials per swarm size; SAM launched from 20 km at an offset
drawn uniformly from ±25° off nose-on. Every trial draws from its own
independent random stream addressed by `[MASTER_SEED, n_drones, trial]`, so the
sweep is reproducible, individual trials are reproducible in isolation, and
adding a configuration does not perturb existing ones.

## 6. Results

| Escort drones | Survival rate | 95% CI | Mean drones lost | Mean miss distance |
|---|---|---|---|---|
| 0 | 0.13 | [0.07, 0.24] | 0.00 | 0.00 m |
| 1 | 1.00 | [0.94, 1.00] | 0.67 | 6.29 m |
| 2 | 1.00 | [0.94, 1.00] | 0.65 | 6.35 m |
| 3 | 1.00 | [0.94, 1.00] | 0.82 | 6.60 m |
| 4 | 1.00 | [0.94, 1.00] | 0.78 | 6.59 m |
| 6 | 1.00 | [0.94, 1.00] | 0.63 | 6.80 m |
| 8 | 1.00 | [0.94, 1.00] | 0.77 | 6.63 m |

![Survival vs swarm size](output/survival_vs_swarm_size.png)
![How the shot resolved](output/sam_outcomes.png)
![Sample trajectory](output/sample_trajectory.png)

One engagement, rendered by `animate_engagement.py`. The escort drone flies onto
the missile's collision triangle, forces the warhead to be expended 4 km short of
the mothership, and is destroyed by the burst. The mothership survives.

![Engagement animation](engagement_anim3d.gif)

### Findings

**F1 — Unescorted, the aircraft dies.** A perfectly guided PN missile against a
non-manoeuvring F-16 achieves essentially zero miss distance, and the 13%
survival rate is exactly the complement of the modeled warhead lethality
(Pk_max = 0.9). This is the baseline the escort has to beat.

**F2 — One interceptor is sufficient, and more add nothing.** Survival goes to
1.00 with a single escort drone and stays there. The effect is large enough to
resolve with 4 trials per arm; the remaining seven configurations are
indistinguishable from each other. Under this threat, the marginal value of the
second through eighth drone is zero.

**F3 — Protection is paid for in drones.** Mean attrition is 0.63-0.82 drones
per engagement across every non-zero swarm size — the interceptor forces the
warhead to be expended and is inside the burst when it happens. Miss distance
settles near 6.5 m, where the damage function gives roughly a 76% chance of
killing the drone. The exchange is one drone, most of the time, for one manned
aircraft.

**F4 — Soft kill did almost nothing here.** Jammers and decoys denied seeker
lock on under 2% of ticks, and denial rarely changed an outcome. The reason is
A1: against a target holding a constant course, a missile that loses lock coasts
along a heading that is already a collision course. Electronic attack without a
defensive manoeuvre is close to worthless, which is consistent with how the two
are employed together in practice.

### What these numbers are not

F2's "1.00 survival" is an upper bound produced by A2 — the interceptor is
perfectly cued. Insert any realistic detection, track and cueing delay and the
interceptor's achievable closest approach grows, at which point the 30 m fuze
radius stops being a certainty. **The right reading of F2 is that hard-kill
body-block is geometrically feasible in this scenario, not that it is reliable.**

## 7. Repository layout

| File | Role |
|---|---|
| `lethality.py` | Closest-approach solver, damage function, hazard-rate conversion |
| `f16_mothership.py` | Wraps `code/aerobench` as the mothership; metres/feet conversion at the boundary |
| `aircraft.py` | Reduced-order `Vehicle` model and autopilot, used for escort drones |
| `swarm.py` | `Drone` and swarm role logic: escort, decoy, jammer, interceptor. Role policy is pluggable |
| `missile.py` | `GenericSAM`: PN guidance, seeker acquisition, proximity fuze, damage roll |
| `engagement.py` | `run_engagement()`: one mothership, one swarm, one SAM, one outcome |
| `run_experiment.py` | Monte Carlo harness, Wilson intervals, summary plots |
| `animate_engagement.py` | Renders one engagement as an animated 3D GIF |
| `viz_style.py` | Shared chart palette and chrome |
| `tests/` | Unit and end-to-end regression tests |
| `VALIDATION.md` | Verification and validation evidence |

## 8. Running it

```
cd fury_sim
python -m pytest tests/ -q      # 18 tests, ~25 s
python run_experiment.py        # Monte Carlo sweep, ~1 min
python animate_engagement.py    # renders one engagement as a 3D GIF
```

Requires `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest` and `Pillow`. No
separate install step for the AeroBenchVV dependency: `f16_mothership.py` adds
`code/` to `sys.path` at import time.

### Animation options

| Flag | Meaning | Default |
|---|---|---|
| `--n-drones` | escort swarm size | 1 |
| `--seed` | trial seed | 1 |
| `--sam-offset-deg` | SAM launch angle off nose-on | 0 |
| `--max-frames` | rendered frame budget | 200 |
| `--hold-seconds` | final-frame freeze before the loop restarts | 2.5 |
| `--elev`, `--azim` | 3D camera angle | 25, -60 |
| `--fps` | playback frame rate | 20 |

At this simulation's scale an intercepted SAM and a hit on the mothership look
nearly identical — both end with the SAM marker converging on the same cluster.
The animation therefore states the outcome in text rather than leaving it to
marker proximity, and marks whatever was actually destroyed with a red X.

## 9. Known limitations

Beyond the assumptions in §3:

- **Miss distance has no spread.** With perfect seeker information, no guidance
  lag and a non-manoeuvring target, PN produces a near-zero miss every time, so
  the damage function always evaluates at its maximum against the mothership.
  Seeker noise, target manoeuvre and guidance lag generate real miss-distance
  distributions and none are modeled.
- **No detection model.** Nothing in this study depends on radar cross-section,
  which means it cannot address whether an escort drone makes the formation
  easier to find — the question that motivates most current scepticism about
  manned-unmanned teaming.
- **No cost model.** F3 describes an exchange in units of aircraft, not dollars.
- **Countermeasure rates are placeholders**, and soft-kill effectiveness is
  therefore not a result this study can defend.
- **Escort drones use a reduced-order flight model**, so their manoeuvring
  limits are approximate and not derived from a specific airframe.

## 10. Attribution

Built on top of AeroBenchVV, the F-16 verification benchmark in the rest of this
repository. See the top-level `README.md` and `LICENSE` for citation information
and license terms.
