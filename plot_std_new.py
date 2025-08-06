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

    pl.title(f"Mean Training and Test Loss vs. Epochs (Width {width})")
    pl.xlabel("Epochs")
    pl.ylabel("Loss")
    pl.yscale('log')
    pl.legend()
    pl.grid(True)
    plot_filename = os.path.join(plot_dir_analysis, f"mean_losses_over_epochs_width_{width}.png")
    pl.savefig(plot_filename)
    pl.close()

print("\nFinished plotting mean losses over epochs with standard deviation.")