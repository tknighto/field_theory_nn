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

    

    pl.title(f"Training Loss over Epochs (Width: {width})")
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


pl.figure(figsize=(10, 6))
pl.errorbar(inverse_widths, final_train_losses, yerr=final_train_stds, fmt='o-', label="Final Training Loss")


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