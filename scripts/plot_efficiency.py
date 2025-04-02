import matplotlib.pyplot as plt

# Data
methods = [
    "SAM2Act", "ARP+", "3D-LOTUS", "RVT-2", "3D Diffuser Actor",
    "Act3D", "RVT", "PerAct", "PolarNet", "Ours"
]
success_rate = [86.8, 86, 83.1, 81.4, 81.3, 65, 62.9, 49.4, 46.4, 75.5]
training_time = [1.04, 0.5, 0.28, 0.83, 8, 5, 1, 16, 5, 0.24]  # in (V100 x 8 x day)
memory_cost = [160, 64, 32, 128, 240, 128, 128, 128, 64, 16]   # in GB
years = [2025, 2024, 2024, 2024, 2024, 2023, 2023, 2022, 2023, 2025]

# Bubble sizes based on performance
bubble_sizes = [(p - 35) * 50 for p in success_rate]

# Reference value: 3D Diffuser Actor success rate
ref_idx = methods.index("3D Diffuser Actor")
ref_score = success_rate[ref_idx]

# Special labels for only 3 methods
special_methods = {"SAM2Act", "3D Diffuser Actor", "Ours"}
labels = []
for i in range(len(methods)):
    base_label = f"{methods[i]} ({years[i]})"
    if methods[i] in special_methods:
        rel_perf = round(success_rate[i] / ref_score * 100, 1)
        label = f"{base_label}\n{rel_perf}% of 3D Diffuser"
    else:
        label = base_label
    labels.append(label)

# Custom offsets to avoid overlaps (x_multiplier, y_multiplier)
offsets = [
    (1.02, 1.10), (1.05, 0.92), (1.05, 0.88), (1.03, 1.02), (0.98, 1.05),
    (0.95, 0.95), (1.10, 1.08), (0.88, 1.00), (1.12, 0.95), (1.08, 0.85)
]

# Create plot
plt.figure(figsize=(12, 7))
scatter = plt.scatter(training_time, memory_cost, s=bubble_sizes, c=success_rate, cmap='plasma', alpha=0.75)

# Log scale
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel('Training Time (log scale, V100 x 8 x day)', fontsize=14)
plt.ylabel('Memory Cost (log scale, GB)', fontsize=14)
plt.title('Training Efficiency vs Memory Cost vs Performance + Year', fontsize=16)

# Custom y-axis ticks
plt.yticks([10, 20, 50, 100, 200, 300], labels=["10", "20", "50", "100", "200", "300"], fontsize=12)
plt.xticks(fontsize=12)

# Annotate with custom offsets to prevent overlapping
for i in range(len(methods)):
    x_offset = training_time[i] * offsets[i][0]
    y_offset = memory_cost[i] * offsets[i][1]
    plt.annotate(labels[i], (x_offset, y_offset), fontsize=12)

# Color bar
plt.colorbar(scatter, label='Success Rate (%)')

# Grid and layout
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()