import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Output directory (relative — works on any machine, including the VM)
OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis('off')

boxes = [
    ("All patients with gastroparesis and diabetes\nundergoing bariatric surgery\nTriNetX (2015–2025)\nN = 1,118", 13.7, '#dbe9f7'),
    ("After age ≥18 years restriction\nN = 1,117", 12.1, '#dbe9f7'),
    ("After confirmed K31.84 (gastroparesis diagnosis)\nwithin 1 year before surgery\nN = 907", 10.5, '#dbe9f7'),
    ("After confirmed E10/E11 (diabetes diagnosis)\nwithin 1 year before surgery\nN = 879", 8.9, '#dbe9f7'),
    ("After gastric emptying study (GES)\nconfirming K31.84 diagnosis\nN = 384", 7.3, '#dbe9f7'),
    ("Final gastroparesis cohort\n(single bariatric surgery)\nN = 376", 5.7, '#dbe9f7'),
    ("Complete-case cohort\n(matching-eligible covariates)\nN = 229", 4.1, '#dbe9f7'),
]

excl = [
    ("Excluded: age <18 years\n(n = 1)", 12.9),
    ("Excluded: no gastroparesis diagnosis\nwithin 1 year before surgery\n(n = 210)", 11.3),
    ("Excluded: no diabetes diagnosis\nwithin 1 year before surgery\n(n = 28)", 9.7),
    ("Excluded: no confirmatory gastric\nemptying study\n(n = 495)", 8.1),
    ("Excluded: multiple bariatric surgeries\n(n = 8)", 6.5),
    ("Excluded: missing covariate data\nrequired for PSM\n(n = 147)", 4.9),
]

box_w, box_h = 5.4, 1.1
for text, y, color in boxes:
    rect = mpatches.FancyBboxPatch((0.3, y - box_h/2), box_w, box_h,
                                    boxstyle="round,pad=0.05",
                                    linewidth=1.2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(0.3 + box_w/2, y, text, ha='center', va='center', fontsize=9.5)

excl_w, excl_h = 3.9, 1.0
for text, y in excl:
    rect = mpatches.FancyBboxPatch((6.0, y - excl_h/2), excl_w, excl_h,
                                    boxstyle="round,pad=0.05",
                                    linewidth=1.0, edgecolor='black', facecolor='#fbe1e1')
    ax.add_patch(rect)
    ax.text(6.0 + excl_w/2, y, text, ha='center', va='center', fontsize=8.5)

# vertical arrows between main boxes
main_ys = [y for _, y, _ in boxes]
for i in range(len(main_ys) - 1):
    ax.annotate('', xy=(3.0, main_ys[i+1] + box_h/2), xytext=(3.0, main_ys[i] - box_h/2),
                arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black'))

# horizontal arrows to exclusion boxes
for _, ey in excl:
    ax.annotate('', xy=(6.0, ey), xytext=(3.0, ey),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color='black'))

# final matched cohort box
final_y = 2.2
final_w, final_h = 6.5, 1.6
rect = mpatches.FancyBboxPatch((0.05, final_y - final_h/2), final_w, final_h,
                                boxstyle="round,pad=0.05",
                                linewidth=1.5, edgecolor='black', facecolor='#d9ead3')
ax.add_patch(rect)
ax.text(0.05 + final_w/2, final_y,
        "MATCHED COHORT (Optimal 1:1 PSM)\nGastroparesis: n = 227\nComparator: n = 227\nTotal N = 454\n(from comparator pool N = 2,797)",
        ha='center', va='center', fontsize=10, fontweight='bold')

ax.annotate('', xy=(3.0, final_y + final_h/2), xytext=(3.0, main_ys[-1] - box_h/2),
            arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black'))

plt.tight_layout()

# Vector PDF (preferred) + 300-dpi TIFF backup; PNG kept for pasting into Word
base = os.path.join(OUTPUT_DIR, "Figure_1")
plt.savefig(base + ".pdf", bbox_inches='tight')
plt.savefig(base + ".tiff", dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.savefig(base + ".png", dpi=300, bbox_inches='tight')
print(f"saved: {base}.pdf / .tiff / .png")
