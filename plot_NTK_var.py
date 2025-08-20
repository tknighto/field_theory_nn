import matplotlib.pyplot as pl
import numpy as np
import os
import pickle

# Define the directory where the saved data is located
data_dir = "loss_data"
final_plot_dir = "plots/final_plots"
os.makedirs(final_plot_dir, exist_ok=True)

# List of widths that were trained
widths = range(25,90,20)

# Define the two target training times for plotting
target_times = [0, 500, 1000, 1500]

# Dictionaries to store the variance and standard error for each width at each target time
variance_at_target_times = {time: [] for time in target_times}
std_error_at_target_times = {time: [] for time in target_times}
widths_for_plotting_at_times = {time: [] for time in target_times}


# Load data for each width and extract the time-dependent variance and SE
for width in widths:
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    if os.path.exists(data_filename):
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        # Extract time-dependent variance, standard error, and recorded times
        variance_first_ntk_entry_times_val = data.get('variance_first_ntk_entry_times')
        std_error_variance_first_ntk_entry_times_val = data.get('std_error_variance_first_ntk_entry_times')
        recorded_ntk_times_first_entry_val = data.get('ntk_record_times_first_entry')

        if variance_first_ntk_entry_times_val is not None and std_error_variance_first_ntk_entry_times_val is not None and recorded_ntk_times_first_entry_val is not None:
            # Find the closest recorded time for each target time
            for target_time in target_times:
                if recorded_ntk_times_first_entry_val:
                     closest_time_index = min(range(len(recorded_ntk_times_first_entry_val)), key=lambda i: abs(recorded_ntk_times_first_entry_val[i] - target_time))
                     closest_recorded_time = recorded_ntk_times_first_entry_val[closest_time_index]

                     # Check if the closest time is reasonably close to the target time
                     if abs(closest_recorded_time - target_time) < 5.0: # Tolerance of 5.0 time units
                          if closest_time_index < len(variance_first_ntk_entry_times_val) and closest_time_index < len(std_error_variance_first_ntk_entry_times_val):
                               variance_at_target_times[target_time].append(variance_first_ntk_entry_times_val[closest_time_index])
                               std_error_at_target_times[target_time].append(std_error_variance_first_ntk_entry_times_val[closest_time_index])
                               widths_for_plotting_at_times[target_time].append(width)
                          else:
                               print(f"Warning: Data index mismatch for width {width} at time {target_time}. Skipping.")
                     else:
                          print(f"Warning: No recorded time close to {target_time:.4f} for width {width}. Closest is {closest_recorded_time:.4f}. Skipping for this target time.")
                else:
                     print(f"Warning: No recorded NTK times found for width {width}. Skipping for all target times.")


# Plotting the variance of the first NTK entry vs Width for the two target times on the same plot
pl.figure(figsize=(10, 6))

colors = ['blue', 'red', 'green'] # Define colors for different time points
markers = ['o', 's', '^'] # Define markers for different time points

for i, target_time in enumerate(target_times):
    if widths_for_plotting_at_times[target_time]:
        pl.errorbar(widths_for_plotting_at_times[target_time], variance_at_target_times[target_time],
                    yerr=std_error_at_target_times[target_time],
                    marker=markers[i % len(markers)], linestyle='-', capsize=5,
                    label=f"Time ~{target_time:.1f} with SE", color=colors[i % len(colors)])
    else:
        print(f"No valid data available to plot for Training Time ~{target_time:.1f}.")


pl.title("Variance of First NTK Entry vs Width at Different Training Times with SE")
pl.xlabel("Width")
pl.ylabel("Variance of First NTK Entry")
pl.grid(True)
pl.legend()

# Save the plot
variance_time_comparison_plot_path = os.path.join(final_plot_dir, "variance_first_ntk_entry_vs_width_time_comparison_with_se.png")
pl.savefig(variance_time_comparison_plot_path)
pl.close()

print(f"Comparison plot of Variance of first NTK entry vs Width with SE at different times saved to {variance_time_comparison_plot_path}")