# epm_metric.py
#  Aime Serge Tuyishime
# 4/16/2025
# Purpose: Measure severity of bug by comparing outputs

import matplotlib.pyplot as plt

# ---------------------------
# Clean vs Buggy Outputs
# ---------------------------
clean_output = "Area: 50"
buggy_output = "NameError: name 'calculat_area' is not defined"

# ---------------------------
# iii. Calculate the Butterfly Effect (EPM)
# ---------------------------
def calculate_epm(clean, buggy):
    return len(set(clean.split()) ^ set(buggy.split()))

epm = calculate_epm(clean_output, buggy_output)
print("Calculated Error Propagation Magnitude (EPM):", epm, "differing word units")

# ---------------------------
# iv. Visualize Butterfly Effect on Code Snippets (display only)
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.text(0.05, 0.8, "Clean Code", fontsize=12, fontweight='bold')
ax.text(0.05, 0.6,
        "def calculate_area(length, width):\n    return length * width\n\ndef print_area():\n    l = 10\n    w = 5\n    area = calculate_area(l, w)\n    print('Area:', area)",
        fontsize=9, family='monospace', bbox=dict(facecolor='lightgreen', edgecolor='black'))

ax.text(0.6, 0.8, "Buggy Code (with typo)", fontsize=12, fontweight='bold')
ax.text(0.6, 0.6,
        "def calculate_area(length, width):\n    return length * width\n\ndef print_area():\n    l = 10\n    w = 5\n    area = calculat_area(l, w)\n    print('Area:', area)",
        fontsize=9, family='monospace', bbox=dict(facecolor='lightcoral', edgecolor='black'))

plt.tight_layout()
plt.show()

# ---------------------------
# v. Visualize Program Output Comparison (display only)
# ---------------------------
labels = ['Clean Code Output', 'Buggy Code Output']
values = [50, 0]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, values, color=["green", "red"])
plt.title('Program Output Comparison: Clean vs Buggy')
plt.ylabel('Program Output (units)')
plt.ylim(0, 60)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{height}', ha='center')

plt.tight_layout()
plt.show()
