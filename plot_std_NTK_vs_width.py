import pickle
import os
import matplotlib.pyplot as pl
import numpy as np

data_dir = "loss_data"
all_widths_data = []

# Load data for all widths
for filename in os.listdir(data_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_widths_data.append(data)

# Sort data by width
all_widths_data.sort(key=lambda x: x['width'])

# Determine the maximum training time across all widths
max_training_time = 0
for data in all_widths_data:
    if data['ntk_record_times_norms']:
        max_training_time = max(max_training_time, max(data['ntk_record_times_norms']))

# Define the four equally spaced training times for plotting
num_plot_times = 4
plot_times = np.linspace(0, max_training_time, num_plot_times).tolist()

# Store standard deviations of NTK norms at plot_times for each width
std_ntk_norms_at_plot_times_across_widths = [[] for _ in range(num_plot_times)]
inverse_widths = []

for data in all_widths_data:
    width = data['width']
    inverse_widths.append(1 / width)
    std_ntk_norms_times = data.get('std_ntk_norms_times')
    ntk_record_times_norms = data.get('ntk_record_times_norms')

    if std_ntk_norms_times and ntk_record_times_norms:
        # Find the closest recorded time for each plot_time and get the corresponding std
        for i, plot_time in enumerate(plot_times):
            # Find the index of the closest recorded time
            closest_time_index = min(range(len(ntk_record_times_norms)), key=lambda j: abs(ntk_record_times_norms[j] - plot_time))
            closest_std = std_ntk_norms_times[closest_time_index]
            std_ntk_norms_at_plot_times_across_widths[i].append(closest_std)
    else:
        # Append None or NaN if data is missing for this width
        for i in range(num_plot_times):
            std_ntk_norms_at_plot_times_across_widths[i].append(np.nan) # Use NaN for missing data

# Plotting
pl.figure(figsize=(10, 6))

colors = ['blue', 'red', 'green', 'purple']
labels = [f'Time ≈ {plot_times[i]:.2f}' for i in range(num_plot_times)]

for i in range(num_plot_times):
    pl.plot(inverse_widths, std_ntk_norms_at_plot_times_across_widths[i], marker='o', linestyle='-', color=colors[i], label=labels[i])

pl.title("Standard Deviation of NTK Norm vs 1/Width at Selected Training Times")
pl.xlabel("1 / width")
pl.ylabel("Standard Deviation of NTK Norm")
pl.grid(True)
pl.legend()

plot_dir = "plots/ntk_analysis"
os.makedirs(plot_dir, exist_ok=True)
plot_filename = os.path.join(plot_dir, "std_ntk_norm_vs_inverse_width_at_times_linear_x.png")
pl.savefig(plot_filename)
pl.close()

print(f"Plot saved to {plot_filename}")