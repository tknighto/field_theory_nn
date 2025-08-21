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
target_times = [0, 500, 1000, 1490]

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


# Plotting the variance of the first NTK entry vs 1/Width for the two target times on the same plot
pl.figure(figsize=(10, 6))

colors = ['blue', 'red', 'green'] # Define colors for different time points
markers = ['o', 's', '^'] # Define markers for different time points

for i, target_time in enumerate(target_times):
    if widths_for_plotting_at_times[target_time]:
        # Calculate 1/width for plotting
        inverse_widths_for_plotting = [1.0 / w for w in widths_for_plotting_at_times[target_time]]
        pl.errorbar(inverse_widths_for_plotting, variance_at_target_times[target_time],
                    yerr=std_error_at_target_times[target_time],
                    marker=markers[i % len(markers)], linestyle='-', capsize=5,
                    label=f"Time ~{target_time:.1f} with SE", color=colors[i % len(colors)])
    else:
        print(f"No valid data available to plot for Training Time ~{target_time:.1f}.")


pl.title("Variance of First NTK Entry vs 1/Width at Different Training Times with SE")
pl.xlabel("1 / Width")
pl.ylabel("Variance of First NTK Entry")
pl.grid(True)
pl.legend()

# Save the plot
variance_time_comparison_plot_path = os.path.join(final_plot_dir, "variance_first_ntk_entry_vs_inverse_width_time_comparison_with_se.png")
pl.savefig(variance_time_comparison_plot_path)
pl.close()

print(f"Comparison plot of Variance of first NTK entry vs 1/Width with SE at different times saved to {variance_time_comparison_plot_path}")

import pickle
import os
import numpy as np
import matplotlib.pyplot as pl
import math

# Define the directory where the data is saved
data_dir = "loss_data"

# Define the widths that were used
widths = range(25, 90, 20) # Make sure this matches the widths used in training

# Define the target training times for plotting
# We need to select times that are representative and likely to have data across widths.
# Let's choose times that are roughly at the beginning, middle, and end of the recorded times.
# We will find the closest recorded time for each width later.
target_times = [0, 500, 1000, 1490] # These are the times we want to plot against

# Dictionary to store the variance and standard error of the trace for plotting
variance_trace_at_times_across_widths = {target_time: [] for target_time in target_times}
std_error_variance_trace_at_times_across_widths = {target_time: [] for target_time in target_times}
inverse_widths_for_plotting_at_times = {target_time: [] for target_time in target_times}


# Load data for each width and extract the relevant information
for width in widths:
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    if os.path.exists(data_filename):
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        # Extract time-dependent NTK trace data
        std_ntk_traces = data.get('std_ntk_traces_times')
        std_error_std_ntk_traces = data.get('std_error_std_ntk_traces_times')
        recorded_times_traces = data.get('ntk_record_times_traces')

        if std_ntk_traces and std_error_std_ntk_traces and recorded_times_traces:
            # Iterate through target times and find the closest recorded time
            for target_time in target_times:
                if recorded_times_traces: # Ensure there are recorded times
                    closest_time_index = min(range(len(recorded_times_traces)), key=lambda i: abs(recorded_times_traces[i] - target_time))
                    closest_recorded_time = recorded_times_traces[closest_time_index]

                    # Check if the closest time is reasonably close to the target time
                    if abs(closest_recorded_time - target_time) < 10: # Tolerance of 5.0 time units
                         if closest_time_index < len(std_ntk_traces) and closest_time_index < len(std_error_std_ntk_traces):
                              # Calculate the variance of the trace (square of the std dev of the trace)
                              variance_of_trace = std_ntk_traces[closest_time_index]**2

                              # Calculate the standard error of the variance (approx. 2 * std_dev * std_error_of_std_dev)
                              std_error_of_variance = 2 * std_ntk_traces[closest_time_index] * std_error_std_ntk_traces[closest_time_index]

                              # Store the data for plotting
                              variance_trace_at_times_across_widths[target_time].append(variance_of_trace)
                              std_error_variance_trace_at_times_across_widths[target_time].append(std_error_of_variance)
                              inverse_widths_for_plotting_at_times[target_time].append(1 / width)
                         else:
                              print(f"Warning: Data index mismatch for width {width} at time {closest_recorded_time:.4f}. Skipping for target time {target_time:.1f}.")
                    else:
                         print(f"Warning: No recorded time close to {target_time:.4f} for width {width} for trace variance/SE. Closest is {closest_recorded_time:.4f}. Skipping.")
        else:
             print(f"Warning: Missing NTK trace data for width {width}.")

    else:
        print(f"Warning: Data file not found for width {width}: {data_filename}")


# Plot the variance of the NTK trace vs 1/width for each target time on the same graph
plot_dir = "plots/final_plots" # Use the same directory as before
os.makedirs(plot_dir, exist_ok=True)

pl.figure(figsize=(10, 6)) # Create a single figure for all plots

for target_time in target_times:
    variances = variance_trace_at_times_across_widths[target_time]
    std_errors = std_error_variance_trace_at_times_across_widths[target_time]
    inverse_widths = inverse_widths_for_plotting_at_times[target_time]

    if inverse_widths:
        # Sort data by inverse width for plotting
        sorted_indices = np.argsort(inverse_widths)
        sorted_inverse_widths = np.array(inverse_widths)[sorted_indices]
        sorted_variances = np.array(variances)[sorted_indices]
        sorted_std_errors = np.array(std_errors)[sorted_indices]


        pl.errorbar(sorted_inverse_widths, sorted_variances, yerr=sorted_std_errors, marker='o', linestyle='-', capsize=5, label=f"Time ~{target_time:.1f} with SE")
    else:
        print(f"No data available to plot Variance of NTK Trace vs 1/Width at Training Time ~{target_time:.1f}.")

# Add titles and labels to the combined plot
pl.title("Variance of NTK Trace vs 1/Width at Different Training Times with SE")
pl.xlabel("1 / width")
pl.ylabel("Variance of NTK Trace")
pl.grid(True)
pl.legend()

# Save the combined plot
plot_filename = os.path.join(plot_dir, f"variance_ntk_trace_vs_inverse_width_combined_with_se.png")
pl.savefig(plot_filename)
pl.close()
print(f"Combined plot of Variance of NTK Trace vs 1/Width with SE saved to {plot_filename}")


print("\nFinished plotting Variance of NTK Trace vs 1/Width.")