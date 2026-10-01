import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GP_COLOR = '#1b7a7a'
COMP_COLOR = '#b3b3b3'

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ---------- Panel A: A1C Trajectory ----------
ax = axes[0, 0]
years = [0, 1, 2, 3, 4, 5]
gp_mean = [7.29, 6.29, 6.52, 6.70, 6.95, 6.80]
gp_sd   = [1.59, 1.15, 1.54, 1.81, 2.08, 1.69]
gp_n    = [227, 186, 128, 93, 60, 40]
comp_mean = [7.26, 6.26, 6.50, 6.50, 6.76, 6.51]
comp_sd   = [1.43, 1.30, 1.63, 1.56, 1.73, 1.58]
comp_n    = [227, 192, 147, 108, 83, 62]

gp_se = [sd/np.sqrt(n) for sd, n in zip(gp_sd, gp_n)]
comp_se = [sd/np.sqrt(n) for sd, n in zip(comp_sd, comp_n)]

ax.errorbar(years, gp_mean, yerr=gp_se, marker='o', color=GP_COLOR, label='Gastroparesis',
            capsize=3, linewidth=2, markersize=6)
ax.errorbar(years, comp_mean, yerr=comp_se, marker='o', color=COMP_COLOR, label='Comparator',
            capsize=3, linewidth=2, markersize=6)
ax.set_xlabel("Year post-surgery", fontsize=11, labelpad=8)
ax.set_ylabel("Mean A1C (%)", fontsize=11)
ax.set_title("A. A1C Trajectory", fontsize=13, fontweight='bold', loc='left')
ax.legend(loc='upper right', frameon=False)

# ---- Number-at-risk row (SOARD requires denominators at each timepoint) ----
ax.text(-0.11, -0.145, "n followed", transform=ax.transAxes, ha='left', va='top',
        fontsize=8.5, fontweight='bold')
for yr, n in zip(years, gp_n):
    ax.text(yr, -0.145, str(n), transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8.5, color=GP_COLOR)
for yr, n in zip(years, comp_n):
    ax.text(yr, -0.195, str(n), transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8.5, color=COMP_COLOR)

# ---------- Panel B: ED & Hospitalization Utilization ----------
ax = axes[0, 1]
labels = ['ED visit\n(OR 0.88)', 'Hospitalization\n(OR 0.98)']
gp_pct = [129/227*100, 83/227*100]
comp_pct = [136/227*100, 84/227*100]
pvals = ['p=0.57', 'p=1.00']

x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, gp_pct, w, color=GP_COLOR, label='Gastroparesis')
ax.bar(x + w/2, comp_pct, w, color=COMP_COLOR, label='Comparator')
for i, p in enumerate(pvals):
    ax.text(x[i], max(gp_pct[i], comp_pct[i]) + 3, p, ha='center', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Patients with \u22651 event (%)", fontsize=11)
ax.set_ylim(0, 90)
ax.set_title("B. ED & Hospitalization Utilization", fontsize=13, fontweight='bold', loc='left')
ax.legend(loc='upper right', frameon=False)

# ---------- Panel C: New-Onset Diabetes Complications ----------
ax = axes[1, 0]
labels = ['Any complication\n(1-year)', 'Any complication\n(5-year)']
gp_pct = [12/136*100, 28/136*100]
comp_pct = [15/127*100, 27/127*100]
pvals = ['p=0.55', 'p=1.00']

x = np.arange(len(labels))
ax.bar(x - w/2, gp_pct, w, color=GP_COLOR, label='Gastroparesis')
ax.bar(x + w/2, comp_pct, w, color=COMP_COLOR, label='Comparator')
for i, p in enumerate(pvals):
    ax.text(x[i], max(gp_pct[i], comp_pct[i]) + 1, p, ha='center', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Incidence among at-risk (%)", fontsize=11)
ax.set_ylim(0, 30)
ax.set_title("C. New-Onset Diabetes Complications", fontsize=13, fontweight='bold', loc='left')
ax.legend(loc='upper right', frameon=False)

# ---------- Panel D: New Insulin Initiation ----------
ax = axes[1, 1]
labels = ['Rapid-acting\n(HR 0.94)', 'Long-acting\n(HR 1.24)']
gp_pct = [32/134*100, 28/139*100]
comp_pct = [36/127*100, 26/138*100]
pvals = ['p=0.79', 'p=0.43']

x = np.arange(len(labels))
ax.bar(x - w/2, gp_pct, w, color=GP_COLOR, label='Gastroparesis')
ax.bar(x + w/2, comp_pct, w, color=COMP_COLOR, label='Comparator')
for i, p in enumerate(pvals):
    ax.text(x[i], max(gp_pct[i], comp_pct[i]) + 1.2, p, ha='center', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("New initiation (%)", fontsize=11)
ax.set_ylim(0, 40)
ax.set_title("D. New Insulin Initiation", fontsize=13, fontweight='bold', loc='left')
ax.legend(loc='upper right', frameon=False)

plt.tight_layout(h_pad=3)

# Vector PDF (preferred) + 300-dpi TIFF backup; PNG kept for pasting into Word
base = os.path.join(OUTPUT_DIR, "Figure_2")
plt.savefig(base + ".pdf", bbox_inches='tight')
plt.savefig(base + ".tiff", dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.savefig(base + ".png", dpi=300, bbox_inches='tight')
print(f"saved: {base}.pdf / .tiff / .png")
