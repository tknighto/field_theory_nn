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
    # Convert the list to a NumPy array before multiplication
    std_devs = np.array(std_ntk_norms_at_plot_times_across_widths[i])
    inverse_widths_np = np.array(inverse_widths) # Convert inverse_widths to numpy array for element-wise multiplication
    pl.plot(inverse_widths_np, std_devs * inverse_widths_np, marker='o', linestyle='-', color=colors[i], label=labels[i])


pl.title("Standard Deviation of NTK Norm * (1/Width) vs 1/Width at Selected Training Times")
pl.xlabel("1 / width")
pl.ylabel("Standard Deviation of NTK Norm * (1/Width)")
pl.grid(True)
pl.legend()

plot_dir = "plots/ntk_analysis"
os.makedirs(plot_dir, exist_ok=True)
plot_filename = os.path.join(plot_dir, "std_ntk_norm_times_inverse_width_at_times.png")
pl.savefig(plot_filename)
pl.close()

print(f"Plot saved to {plot_filename}")

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

# Prepare data for plotting
inverse_widths = []
std_eigenvalues_across_widths = []
times_with_eigenvalue_data = [] # To store the recorded times for eigenvalues

if all_widths_data:
    # Assuming the recorded times for eigenvalues are the same for all widths
    # We will use the times from the first data entry
    times_with_eigenvalue_data = all_widths_data[0].get('ntk_record_times_eigenvalues', [])

    # Initialize lists to hold the standard deviations of eigenvalues for each recorded time, across all widths
    if times_with_eigenvalue_data:
        std_eigenvalues_at_times_across_widths = [[] for _ in range(len(times_with_eigenvalue_data))]

        for data in all_widths_data:
            width = data['width']
            inverse_widths.append(1 / width)
            std_eigenvalue_spectra_times = data.get('std_eigenvalue_spectra_times', [])

            # Check if the number of recorded times for eigenvalues matches the first entry
            if len(std_eigenvalue_spectra_times) != len(times_with_eigenvalue_data):
                print(f"Warning: Number of recorded times for eigenvalues for width {width} is inconsistent. Skipping.")
                # Append NaN for this width at all times if inconsistent
                for time_idx in range(len(times_with_eigenvalue_data)):
                     std_eigenvalues_at_times_across_widths[time_idx].append(np.nan)
                continue


            # For each recorded time, calculate the mean of the standard deviations across the eigenvalue spectrum
            # This aggregates the std of individual eigenvalues at a given time across the ensemble
            # The request is to plot the "std of the eigenvalues vs 1/width".
            # The saved `std_eigenvalue_spectra_times` is a list of lists: [time_1_stds, time_2_stds, ...].
            # Each inner list is the standard deviation of each eigenvalue across the ensemble at that specific time.
            # To plot "std of the eigenvalues vs 1/width", we need a single value representing the std of eigenvalues at each time for each width.
            # A reasonable approach is to take the mean or median of the `std_eigenvalue_spectra_times` list for each time. Let's use the mean.

            mean_std_eigenvalues_at_times = [np.mean(stds) if stds else np.nan for stds in std_eigenvalue_spectra_times]

            for time_idx in range(len(times_with_eigenvalue_data)):
                std_eigenvalues_at_times_across_widths[time_idx].append(mean_std_eigenvalues_at_times[time_idx])

# Plotting
if inverse_widths and std_eigenvalues_at_times_across_widths:
    pl.figure(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'] # More colors if needed

    for i, time in enumerate(times_with_eigenvalue_data):
        if i < len(colors):
            # Multiply std by inverse width and square before plotting
            pl.plot(inverse_widths, (np.array(std_eigenvalues_at_times_across_widths[i]) * np.array(inverse_widths))**2, marker='o', linestyle='-', color=colors[i], label=f'Time ≈ {time:.2f}')
        else:
            # Use default color if not enough colors are defined
            pl.plot(inverse_widths, (np.array(std_eigenvalues_at_times_across_widths[i]) * np.array(inverse_widths))**2, marker='o', linestyle='-', label=f'Time ≈ {time:.2f}')


    pl.title("Standard Deviation of NTK Eigenvalues Squared vs 1/Width at Selected Training Times")
    pl.xlabel("1 / width")
    pl.ylabel("(Standard Deviation of NTK Eigenvalues * (1/Width))^2")
    pl.grid(True)
    pl.legend()

    plot_dir = "plots/ntk_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, "mean_std_eigenvalues_times_inverse_width_squared_at_times.png")
    pl.savefig(plot_filename)
    pl.close()

    print(f"Plot saved to {plot_filename}")
else:
    print("No valid data to plot.")