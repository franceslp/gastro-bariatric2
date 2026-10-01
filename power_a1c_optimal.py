#!/usr/bin/env python3
"""
power_a1c_optimal.py

Sensitivity power analysis for the between-group A1C comparison
(gastroparesis vs matched comparator): given the achieved sample sizes,
the minimum detectable difference (MDD) at 80% power, two-sided alpha = 0.05.

Method (two-sample comparison of mean A1C at each follow-up year):
    SE  = sqrt(SD_gp^2 / n_gp + SD_comp^2 / n_comp)
    MDD = (z_{1-alpha/2} + z_{power}) * SE  =  (1.96 + 0.84) * SE  ~= 2.80 * SE
SDs and n are the OBSERVED values at each year (not the observed difference,
so this is not "observed power").

Primary timepoint reported in the paper: Year 1 -> MDD = 0.35 points.

Inputs: per-year n and SD from a1c_stats_timepoint_optimal.csv
(also in the workbook, sheet "A1c Trajectory").
"""
import math
from scipy.stats import norm

ALPHA, POWER = 0.05, 0.80
Z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)          # 1.96 + 0.84 = 2.80

#        year: (GP n, GP SD, Comp n, Comp SD)   from a1c_stats_timepoint_optimal.csv
DATA = {1: (186, 1.15, 192, 1.30),
        2: (128, 1.54, 147, 1.63),
        3: ( 93, 1.81, 108, 1.56),
        4: ( 60, 2.08,  83, 1.73),
        5: ( 40, 1.69,  62, 1.58)}

print(f"Sensitivity power analysis: alpha={ALPHA} (two-sided), power={POWER:.0%}, "
      f"multiplier={Z:.2f}\n")
print(f"{'Year':<6}{'n GP/Comp':<12}{'SD GP/Comp':<14}{'SE':<8}{'MDD (points)'}")
for yr, (n1, sd1, n2, sd2) in DATA.items():
    se = math.sqrt(sd1**2 / n1 + sd2**2 / n2)
    flag = "   <- primary timepoint (reported in paper)" if yr == 1 else ""
    print(f"{yr:<6}{f'{n1}/{n2}':<12}{f'{sd1}/{sd2}':<14}{se:<8.3f}{Z*se:.2f}{flag}")
