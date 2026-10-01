import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = [
    ("ED visit (OR)",                 0.88, 0.61, 1.28, "0.88 (0.61\u20131.28)", "p=0.57", False),
    ("Hospitalization (OR)",          0.98, 0.67, 1.44, "0.98 (0.67\u20131.44)", "p=1.00", False),
    ("ED visit rate (IRR)",           1.07, 0.84, 1.35, "1.07 (0.84\u20131.35)", "p=0.60", False),
    ("Hospitalization rate (IRR)",    1.10, 0.83, 1.45, "1.10 (0.83\u20131.45)", "p=0.51", False),
    ("Hospitalizations among\nhospitalized patients$^a$ (IRR)", 1.90, 1.34, 2.69, "1.90 (1.34\u20132.69)", "p<0.001", True),
    ("Rapid-acting insulin (HR)",     0.94, 0.58, 1.51, "0.94 (0.58\u20131.51)", "p=0.79", False),
    ("Long-acting insulin (HR)",      1.24, 0.73, 2.12, "1.24 (0.73\u20132.12)", "p=0.43", False),
]

fig, ax = plt.subplots(figsize=(11, 7))
y_positions = list(range(len(rows), 0, -1))

for (label, est, lo, hi, txt, p, highlight), y in zip(rows, y_positions):
    color = '#c0392b' if highlight else '#2c3e50'
    ax.plot([lo, hi], [y, y], color=color, lw=2, zorder=1)
    ax.scatter([est], [y], color=color, marker='s', s=110, zorder=2)

XMIN, XMAX = 0.4, 3.0
ax.axvline(1.0, color='black', linestyle='--', lw=1)
ax.set_xscale('log')
ax.set_xlim(XMIN, XMAX)
ax.set_xticks([0.4, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])
ax.set_xticklabels(['0.4', '0.5', '0.6', '1.0', '1.5', '2.0', '2.5', '3.0'])
ax.minorticks_off()
ax.set_yticks(y_positions)
ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xlabel("Effect estimate (OR / IRR / HR), 95% CI", fontsize=11)
ax.set_ylim(0.3, len(rows) + 0.7)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Reserve space on the right for the value/p-value text columns
plt.subplots_adjust(right=0.68, bottom=0.18, top=0.85)

for (label, est, lo, hi, txt, p, highlight), y in zip(rows, y_positions):
    y_frac = (y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(1.04, y_frac, txt, transform=ax.transAxes, va='center', ha='left', fontsize=10.5)
    ax.text(1.34, y_frac, p, transform=ax.transAxes, va='center', ha='left', fontsize=10.5)

# Direction labels, anchored on either side of the null line (x = 1 on the log axis)
x_null = np.log(1.0 / XMIN) / np.log(XMAX / XMIN)
ax.text(x_null - 0.02, -0.12, "\u2190 Favors gastroparesis", transform=ax.transAxes,
        fontsize=10, color='gray', ha='right', va='top')
ax.text(x_null + 0.02, -0.12, "Favors comparator \u2192", transform=ax.transAxes,
        fontsize=10, color='gray', ha='left', va='top')

# Vector PDF (preferred) + 300-dpi TIFF backup; PNG kept for pasting into Word
base = os.path.join(OUTPUT_DIR, "Figure_3")
plt.savefig(base + ".pdf", bbox_inches='tight')
plt.savefig(base + ".tiff", dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.savefig(base + ".png", dpi=300, bbox_inches='tight')
print(f"saved: {base}.pdf / .tiff / .png")
