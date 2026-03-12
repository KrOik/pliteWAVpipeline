#!/usr/bin/env python3
"""
Generate professional benchmark charts from benchmark.json
- Bar chart with error bars
- Box plot
- Violin plot
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Load benchmark data
with open('benchmark.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Extract data
formats = ['original_audio', 'cached_wav', 'mmap']
labels = ['Original Audio', 'Cached WAV', 'Memory Map']
colors = ['#e74c3c', '#f39c12', '#27ae60']

# Get throughput data
throughputs = {k: v['clean_throughputs'] for k, v in results.items()}

# Get statistics
means = {k: v['mean_throughput'] for k, v in results.items()}
medians = {k: v['median_throughput'] for k, v in results.items()}
stds = {k: v['std_throughput'] for k, v in results.items()}
ci_lower = {k: v['ci_95_lower'] for k, v in results.items()}
ci_upper = {k: v['ci_95_upper'] for k, v in results.items()}

error_lower = {k: means[k] - ci_lower[k] for k in formats}
error_upper = {k: ci_upper[k] - means[k] for k in formats}

fig1, ax1 = plt.subplots(figsize=(10, 7))

x = np.arange(len(formats))
y = [means[f] for f in formats]
errors_lower_list = [error_lower[f] for f in formats]
errors_upper_list = [error_upper[f] for f in formats]

bars = ax1.bar(x, y, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

ax1.errorbar(x, y, yerr=[errors_lower_list, errors_upper_list], fmt='none', 
            color='black', capsize=8, capthick=2, linewidth=2)

ci_values_lower = [ci_lower[f] for f in formats]
ci_values_upper = [ci_upper[f] for f in formats]
for i, (bar, mean, ci_l, ci_u) in enumerate(zip(bars, y, ci_values_lower, ci_values_upper)):
    ax1.annotate(f'{mean:.1f}\n[{ci_l:.1f}, {ci_u:.1f}]', 
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 8), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.set_ylabel('Throughput (samples/s)', fontsize=14)
ax1.set_title('Audio Data Loading Performance Comparison\n(Mean ± 95% Confidence Interval)', 
              fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(y) * 1.25)

# Add grid
ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
ax1.set_axisbelow(True)

plt.tight_layout()
plt.savefig('benchmark_bar_chart.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: benchmark_bar_chart.png")

# =============================================================================
# Chart 2: Box Plot
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

box_data = [throughputs[f] for f in formats]

bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, 
                  showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'white', 
                                              'markeredgecolor': 'black', 'markersize': 10})

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add value annotations
for i, (label, d) in enumerate(zip(labels, box_data)):
    median = np.median(d)
    mean = np.mean(d)
    ax2.annotate(f'Median: {median:.1f}\nMean: {mean:.1f}', 
                xy=(i+1, max(d)), xytext=(i+1, max(d)*1.05),
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax2.set_ylabel('Throughput (samples/s)', fontsize=14)
ax2.set_title('Distribution of Throughput Measurements\n(Box Plot: Center line=median, Diamond=mean)', 
              fontsize=14, fontweight='bold')
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig('benchmark_box_plot.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: benchmark_box_plot.png")

# =============================================================================
# Chart 3: Violin Plot
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 7))

parts = ax3.violinplot(box_data, positions=range(1, len(formats)+1), 
                        showmeans=True, showmedians=True)

# Color the violins
for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

# Style the lines
parts['cmeans'].set_color('white')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)

# Add scatter points
for i, d in enumerate(box_data):
    x = np.random.normal(i+1, 0.04, size=len(d))
    ax3.scatter(x, d, alpha=0.4, color=colors[i], s=20, edgecolors='white', linewidths=0.5)

ax3.set_xticks(range(1, len(formats)+1))
ax3.set_xticklabels(labels, fontsize=12)
ax3.set_ylabel('Throughput (samples/s)', fontsize=14)
ax3.set_title('Distribution of Throughput Measurements\n(Violin Plot: White diamond=mean, Black line=median)', 
              fontsize=14, fontweight='bold')
ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig('benchmark_violin_plot.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: benchmark_violin_plot.png")

# =============================================================================
# Chart 4: Combined Chart (Bar + Error Bar + Individual Points)
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(12, 8))

x = np.arange(len(formats))
y = [means[f] for f in formats]
errors_lower_list = [error_lower[f] for f in formats]
errors_upper_list = [error_upper[f] for f in formats]

bars = ax4.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5, 
               label='Mean Throughput', width=0.6)

ax4.errorbar(x, y, yerr=[errors_lower_list, errors_upper_list], fmt='none', 
            color='black', capsize=10, capthick=2, linewidth=2, 
            label='95% Confidence Interval')

for i, f in enumerate(formats):
    data_points = throughputs[f]
    x_jitter = np.random.normal(i, 0.08, size=len(data_points))
    ax4.scatter(x_jitter, data_points, alpha=0.5, color=colors[i], 
               s=40, edgecolors='white', linewidths=0.5, zorder=5)

ci_l_list = [ci_lower[f] for f in formats]
ci_u_list = [ci_upper[f] for f in formats]
for i, (bar, mean, ci_l, ci_u) in enumerate(zip(bars, y, ci_l_list, ci_u_list)):
    ax4.annotate(f'{mean:.1f} samples/s\n95% CI: [{ci_l:.1f}, {ci_u:.1f}]', 
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 12), textcoords='offset points',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor=colors[i], alpha=0.9))

ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax4.set_ylabel('Throughput (samples/s)', fontsize=14, fontweight='bold')
ax4.set_title('Benchmark Results: Audio Data Loading Performance\n' + 
              f'Platform: {data["system_info"]["platform"]["system"]} ' +
              f'{data["system_info"]["platform"]["release"]} | ' +
              f'Python: {data["system_info"]["versions"]["python"]} | ' +
              f'PyTorch: {data["system_info"]["versions"]["torch"]}',
              fontsize=12, fontweight='bold')
ax4.set_ylim(0, max(y) * 1.35)
ax4.yaxis.grid(True, linestyle='--', alpha=0.7)
ax4.set_axisbelow(True)

# Legend
ax4.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('benchmark_combined.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: benchmark_combined.png")

# =============================================================================
# Chart 5: Speedup Comparison (Horizontal Bar)
# =============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 6))

speedups = [
    ('Cached WAV vs Original', 2.30),
    ('Mmap vs Original', 55.54),
    ('Mmap vs Cached WAV', 24.17)
]
speedup_colors = ['#f39c12', '#27ae60', '#3498db']

y_pos = np.arange(len(speedups))
values = [s[1] for s in speedups]
labels_s = [s[0] for s in speedups]

bars = ax5.barh(y_pos, values, color=speedup_colors, alpha=0.8, 
                edgecolor='black', linewidth=1.2, height=0.6)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax5.annotate(f'{val:.1f}x', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                xytext=(8, 0), textcoords='offset points',
                ha='left', va='center', fontsize=14, fontweight='bold')

ax5.set_yticks(y_pos)
ax5.set_yticklabels(labels_s, fontsize=12)
ax5.set_xlabel('Speedup Factor (×)', fontsize=14, fontweight='bold')
ax5.set_title('Performance Speedup Analysis', fontsize=14, fontweight='bold')
ax5.set_xlim(0, max(values) * 1.15)
ax5.xaxis.grid(True, linestyle='--', alpha=0.7)
ax5.set_axisbelow(True)

# Add vertical line at 1x
ax5.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax5.text(1.1, -0.5, 'Baseline (1×)', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('benchmark_speedup.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: benchmark_speedup.png")

print("\n✅ All charts generated successfully!")
print("Generated files:")
print("  - benchmark_bar_chart.png")
print("  - benchmark_box_plot.png")
print("  - benchmark_violin_plot.png")
print("  - benchmark_combined.png")
print("  - benchmark_speedup.png")