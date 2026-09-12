"""
mcstats.py
----------
Interval estimates for Monte Carlo proportions, kept in one place so every
figure in this project reports uncertainty the same way.

Survival rates here are binomial proportions estimated from a few hundred
trials. Reported as bare numbers they invite comparisons the sample size cannot
support: at 60 trials the 95% interval on a rate near 0.8 is about +/- 10
points, which is wider than most of the differences a sweep gets read for.
"""

import math
from statistics import NormalDist


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation, which misbehaves badly near 0 and
    1 -- exactly where survival rates live, and exactly where it can produce
    bounds outside [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_interval(successes1, n1, successes2, n2, z=1.96):
    """Newcombe hybrid score interval for the difference of two independent
    proportions, p1 - p2.

    Used for "does the escort help or hurt", which is a difference between two
    separately estimated rates. Combining two Wilson intervals this way keeps
    the good behavior near 0 and 1 that a naive sqrt(SE1^2 + SE2^2) band loses.

    Reference: Newcombe, "Interval estimation for the difference between
    independent proportions", Statistics in Medicine, 1998.
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = successes1 / n1, successes2 / n2
    l1, u1 = wilson_interval(successes1, n1, z)
    l2, u2 = wilson_interval(successes2, n2, z)
    delta = p1 - p2
    lower = delta - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = delta + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lower, upper)


def required_trials(p1, p2, alpha=0.05, power=0.80):
    """Trials per arm needed to resolve a change from p1 to p2."""
    if p1 == p2:
        return float("inf")
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)
