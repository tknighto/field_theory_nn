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

import pickle
import os
import numpy as np

data_dir = "loss_data"
widths = range(5, 50, 20) # Use the same widths as in the training code

print("Checking standard deviation data for potential issues...")

for width in widths:
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    if not os.path.exists(data_filename):
        print(f"Data file not found for width {width}: {data_filename}")
        continue

    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    std_train_losses = data.get('std_1000_epoch_losses')
    std_test_losses = data.get('std_1000_epoch_test_losses')
    mean_train_losses = data.get('mean_1000_epoch_losses') # Get mean losses for comparison

    print(f"\n--- Checking Data for Width {width} ---")

    if std_train_losses is not None:
        print(f"Standard deviation of training losses (1000-epoch):")
        print(std_train_losses)
        # Check for zero values
        if np.any(np.array(std_train_losses) == 0):
            print("Warning: Zero values found in standard deviation of training losses.")
        # Check for unusually small values compared to mean loss (optional, requires context)
        # For a rough check, compare to a small fraction of the mean loss (if mean loss > 0)
        if mean_train_losses is not None and len(mean_train_losses) == len(std_train_losses):
             small_threshold = np.mean(mean_train_losses) * 1e-6 # Example threshold
             if np.any(np.array(std_train_losses) < small_threshold):
                  print(f"Note: Some standard deviation values are very small (less than {small_threshold:.2e}) compared to mean training loss.")


    else:
        print("Standard deviation of training losses data not found.")

    if std_test_losses is not None:
        print(f"\nStandard deviation of test losses (1000-epoch):")
        print(std_test_losses)
        # Check for zero values
        if np.any(np.array(std_test_losses) == 0):
            print("Warning: Zero values found in standard deviation of test losses.")
        # Check for unusually small values compared to mean loss (optional)
        if mean_train_losses is not None and len(mean_train_losses) == len(std_test_losses): # Using train mean as rough scale
             small_threshold = np.mean(mean_train_losses) * 1e-6
             if np.any(np.array(std_test_losses) < small_threshold):
                  print(f"Note: Some standard deviation values are very small (less than {small_threshold:.2e}) compared to mean training loss.")
    else:
        print("Standard deviation of test losses data not found.")

print("\nFinished checking standard deviation data.")
