#!/usr/bin/env python3
"""
power_a1c_optimal.py

Power / minimum detectable difference (MDD) for the between-group A1C comparison
(gastroparesis vs matched comparator), two-sided alpha = 0.05, power = 0.80.

Approximation: two-sample comparison of mean A1C at each follow-up year, using the
observed sample sizes. (The primary analysis is a mixed-effects model; a closed-form
two-sample calculation at the primary timepoint is a standard approximation.)

Inputs are the per-year n and SD from a1c_stats_timepoint_optimal.csv
(also in the workbook, sheet "A1c Trajectory").

Two SD assumptions are reported:
  (a) assumed SD = 1.4% (close to baseline A1C variability; used for the paper's statement)
  (b) observed SDs at each year
"""
import math
from scipy.stats import norm

ALPHA, POWER, TARGET_DIFF, ASSUMED_SD = 0.05, 0.80, 0.4, 1.4
Z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)          # 1.96 + 0.84 = 2.80

#        year: (GP n, GP SD, Comp n, Comp SD)   from a1c_stats_timepoint_optimal.csv
DATA = {1: (186, 1.15, 192, 1.30),
        2: (128, 1.54, 147, 1.63),
        3: ( 93, 1.81, 108, 1.56),
        4: ( 60, 2.08,  83, 1.73),
        5: ( 40, 1.69,  62, 1.58)}

def power_for(diff, se):
    return norm.cdf(diff / se - norm.ppf(1 - ALPHA / 2))

print(f"alpha={ALPHA}, power={POWER}, target difference={TARGET_DIFF} points\n")
print(f"{'Year':<5}{'n GP/Comp':<12}{'MDD (SD=1.4)':<15}{'Power@0.4 (SD=1.4)':<21}"
      f"{'MDD (observed SD)':<19}{'Power@0.4 (observed)'}")
for yr, (n1, sd1, n2, sd2) in DATA.items():
    se_a = ASSUMED_SD * math.sqrt(1 / n1 + 1 / n2)
    se_o = math.sqrt(sd1**2 / n1 + sd2**2 / n2)
    print(f"{yr:<5}{f'{n1}/{n2}':<12}{Z*se_a:<15.2f}{power_for(TARGET_DIFF, se_a):<21.0%}"
          f"{Z*se_o:<19.2f}{power_for(TARGET_DIFF, se_o):.0%}")
