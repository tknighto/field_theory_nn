import pickle
import os
import matplotlib.pyplot as pl
import numpy as np

# Directory where loss data is saved
data_dir = "loss_data"

# List to store loaded data
loaded_data = []

# Iterate through files in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            loaded_data.append(data)

# Sort the loaded data by width
loaded_data.sort(key=lambda x: x['width'])

# Extract relevant data for plotting
widths = [data['width'] for data in loaded_data]
inverse_widths = [1 / width for width in widths]
mean_train_losses_1000_epochs = [data['mean_1000_epoch_losses'] for data in loaded_data]
std_train_losses_1000_epochs = [data['std_1000_epoch_losses'] for data in loaded_data]
mean_test_losses_1000_epochs = [data['mean_1000_epoch_test_losses'] for data in loaded_data]
std_test_losses_1000_epochs = [data['std_1000_epoch_test_losses'] for data in loaded_data]
epochs_recorded_at = [data['epochs_recorded_at'] for data in loaded_data]

print("Loaded data for widths:", widths)

# Plotting training and test losses over epochs for each width
plot_dir_analysis = "plots/analysis_plots"
os.makedirs(plot_dir_analysis, exist_ok=True)

for i, width in enumerate(widths):
    pl.figure(figsize=(10, 6))
    pl.plot(epochs_recorded_at[i], mean_train_losses_1000_epochs[i], label="Mean Training Loss")
    pl.fill_between(epochs_recorded_at[i], np.array(mean_train_losses_1000_epochs[i]) - np.array(std_train_losses_1000_epochs[i]),
                    np.array(mean_train_losses_1000_epochs[i]) + np.array(std_train_losses_1000_epochs[i]), alpha=0.3, label="±1 Std Dev (Train)")

    pl.plot(epochs_recorded_at[i], mean_test_losses_1000_epochs[i], label="Mean Test Loss", linestyle='--')
    pl.fill_between(epochs_recorded_at[i], np.array(mean_test_losses_1000_epochs[i]) - np.array(std_test_losses_1000_epochs[i]),
                    np.array(mean_test_losses_1000_epochs[i]) + np.array(std_test_losses_1000_epochs[i]), alpha=0.3, label="±1 Std Dev (Test)")


    pl.title(f"Training and Test Loss over Epochs (Width: {width})")
    pl.xlabel("Epoch")
    pl.ylabel("MSE")
    pl.yscale('log')
    pl.grid(True)
    pl.legend()
    plot_path = os.path.join(plot_dir_analysis, f"loss_over_epochs_width_{width}.png")
    pl.savefig(plot_path)
    pl.close()
    print(f"Saved loss over epochs plot for width {width} to {plot_path}")

# Plotting mean training loss vs training time for all widths on the same graph
pl.figure(figsize=(10, 6))
for i, data in enumerate(loaded_data):
    width = data['width']
    epochs = data['epochs_recorded_at']
    mean_losses = data['mean_1000_epoch_losses']
    # Assuming LEARNING_RATE is calculated as 0.15 / width
    learning_rate = 0.15 / width
    training_times = [epoch * learning_rate for epoch in epochs]

    pl.plot(training_times, mean_losses, marker='o', linestyle='-', label=f'Width: {width}')

pl.title("Mean Training Loss vs Training Time Across Widths")
pl.xlabel("Training Time (Epochs * Learning Rate)")
pl.ylabel("Mean Training Loss")
pl.yscale('log')
pl.grid(True)
pl.legend()
plot_path_combined_time = os.path.join(plot_dir_analysis, "mean_loss_vs_training_time_all_widths.png")
pl.savefig(plot_path_combined_time)
pl.close()
print(f"Saved combined loss vs training time plot to {plot_path_combined_time}")

# Plotting final training and test losses vs inverse width
final_train_losses = [losses[-1] for losses in mean_train_losses_1000_epochs]
final_train_stds = [stds[-1] for stds in std_train_losses_1000_epochs]
final_test_losses = [losses[-1] for losses in mean_test_losses_1000_epochs]
final_test_stds = [stds[-1] for stds in std_test_losses_1000_epochs]

pl.figure(figsize=(10, 6))
pl.errorbar(inverse_widths, final_train_losses, yerr=final_train_stds, fmt='o-', label="Final Training Loss")
pl.errorbar(inverse_widths, final_test_losses, yerr=final_test_stds, fmt='o-', label="Final Test Loss")

pl.title("Final Training and Test Loss vs 1/Width")
pl.xlabel("1 / width")
pl.ylabel("Final MSE")
pl.yscale('log')
pl.grid(True)
pl.legend()
final_loss_plot_path = os.path.join(plot_dir_analysis, "final_loss_vs_inverse_width.png")
pl.savefig(final_loss_plot_path)
pl.close()
print(f"Saved final loss vs inverse width plot to {final_loss_plot_path}")

import os
import matplotlib.pyplot as pl
import numpy as np

# Plotting training and test loss vs 1/width at 5 equally spaced training times
plot_dir_analysis = "plots/analysis_plots"
os.makedirs(plot_dir_analysis, exist_ok=True)

# Assuming loaded_data is populated from the pickle files
# Data structure in loaded_data: [{'mean_1000_epoch_losses': ..., 'std_1000_epoch_losses': ..., 'mean_1000_epoch_test_losses': ..., 'std_1000_epoch_test_losses': ..., 'width': ..., 'epochs_recorded_at': [...]}, ...]

if loaded_data: # Check if loaded_data is not empty
    # Determine the number of intervals from the first loaded data entry
    # Ensure all data entries have the same number of recorded epochs for intervals
    num_intervals_plotting = len(loaded_data[0]['mean_1000_epoch_losses']) # Using 1000-epoch losses for interval count
    interval_labels = ["0%", "25%", "50%", "75%", "100%"] # These labels are conceptual and might not map directly to 1000-epoch intervals, but we'll use them as before


    pl.figure(figsize=(12, 8))

    # Iterate through each interval (assuming all widths have losses recorded at the same set of 1000-epoch intervals)
    for i in range(num_intervals_plotting):
        inverse_widths = []
        mean_train_losses_at_interval = []
        std_train_losses_at_interval = []
        mean_test_losses_at_interval = []
        std_test_losses_at_interval = []

        # Collect data for the current interval across all widths from loaded_data
        for data in loaded_data:
            width = data['width']
            inverse_widths.append(1 / width)

            # Get the mean and std losses at the current interval index (i)
            mean_train_losses_at_interval.append(data['mean_1000_epoch_losses'][i])
            std_train_losses_at_interval.append(data['std_1000_epoch_losses'][i])
            mean_test_losses_at_interval.append(data['mean_1000_epoch_test_losses'][i])
            std_test_losses_at_interval.append(data['std_1000_epoch_test_losses'][i])


        # At this point, we should have consistent lengths for all lists for this interval

        plottable_inverse_widths = np.array(inverse_widths)
        plottable_mean_train = np.array(mean_train_losses_at_interval)
        plottable_std_train = np.array(std_train_losses_at_interval)
        plottable_mean_test = np.array(mean_test_losses_at_interval)
        plottable_std_test = np.array(std_test_losses_at_interval)


        pl.errorbar(plottable_inverse_widths, plottable_mean_train, yerr=plottable_std_train, fmt='o-', label=f"Train Loss (Interval {i+1})") # Using interval number for clarity
        pl.errorbar(plottable_inverse_widths, plottable_mean_test, yerr=plottable_std_test, fmt='s--', label=f"Test Loss (Interval {i+1})")


    pl.title("Mean Training and Test Loss vs 1/Width at Different Training Stages")
    pl.xlabel("1 / Width")
    pl.ylabel("Mean MSE")
    pl.yscale('log')
    pl.grid(True)
    pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    pl.tight_layout()
    final_combined_loss_plot_path = os.path.join(plot_dir_analysis, "combined_loss_vs_inverse_width_intervals.png")
    pl.savefig(final_combined_loss_plot_path)
    pl.close()
    print(f"Saved combined training and test loss vs inverse width plot to {final_combined_loss_plot_path}")

else:
    print("No data loaded to plot.")