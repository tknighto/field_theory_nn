import os
import matplotlib.pyplot as pl
import numpy as np
# Directory where loss data is saved
data_dir = "loss_data"

# List to store loaded data with NTK properties
loaded_data_with_ntk = []

# Iterate through files in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # Check if NTK properties exist in the loaded data
                if 'ntk_norm' in data and 'ntk_eigenvalues' in data:
                    loaded_data_with_ntk.append(data)
                else:
                    print(f"Skipping file {filename}: NTK properties not found.")
        except Exception as e:
            print(f"Error loading file {filename}: {e}")


# Sort the loaded data by width
loaded_data_with_ntk.sort(key=lambda x: x['width'])

# Extract relevant data for plotting, including NTK properties
widths = [data['width'] for data in loaded_data_with_ntk]
inverse_widths = [1 / width for width in widths]
mean_train_losses_1000_epochs = [data['mean_1000_epoch_losses'] for data in loaded_data_with_ntk]
std_train_losses_1000_epochs = [data['std_1000_epoch_losses'] for data in loaded_data_with_ntk]
mean_test_losses_1000_epochs = [data['mean_1000_epoch_test_losses'] for data in loaded_data_with_ntk]
std_test_losses_1000_epochs = [data['std_1000_epoch_test_losses'] for data in loaded_data_with_ntk]
epochs_recorded_at = [data['epochs_recorded_at'] for data in loaded_data_with_ntk]
ntk_norms = [data['ntk_norm'] for data in loaded_data_with_ntk] # Extract NTK norm
ntk_eigenvalues = [data['ntk_eigenvalues'] for data in loaded_data_with_ntk] # Extract NTK eigenvalues

print("Loaded data for widths (with NTK properties):", widths)
print("Loaded NTK norms:", ntk_norms)
print("Loaded NTK eigenvalues (first 5 for each width):")
for i, eig_list in enumerate(ntk_eigenvalues):
    if eig_list: # Check if the list is not empty
      print(f"Width {widths[i]}: {eig_list[:5]}")
    else:
      print(f"Width {widths[i]}: No eigenvalues recorded.")

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

import os
import matplotlib.pyplot as pl
import numpy as np

# Assuming loaded_data_with_ntk is populated from the pickle files
# Data structure in loaded_data_with_ntk now includes:
# 'mean_eigenvalue_spectra': list of lists (epoch -> mean_spectrum),
# 'std_eigenvalue_spectra': list of lists (epoch -> std_spectrum),
# 'ntk_record_epochs_eigenvalues': list of epochs where eigenvalues were recorded

if loaded_data_with_ntk:

    plot_dir_analysis = "plots/analysis_plots"
    os.makedirs(plot_dir_analysis, exist_ok=True)

    pl.figure(figsize=(10, 6))

    for i, data in enumerate(loaded_data_with_ntk):
        width = data['width']
        epochs = data['ntk_record_epochs_eigenvalues'] # Use epochs specifically for eigenvalues
        mean_eigenvalue_spectra = data['mean_eigenvalue_spectra'] # List of mean spectra (epoch -> mean_spectrum)
        std_eigenvalue_spectra = data['std_eigenvalue_spectra'] # List of std spectra (epoch -> std_spectrum)


        # Extract the largest eigenvalue from the mean spectrum for each epoch
        mean_largest_eigenvalues = []
        std_largest_eigenvalues = [] # This will be the std of the largest eigenvalue across ensemble
        training_times = []

        # Calculate the learning rate for this width
        learning_rate = 0.15 / width

        # Iterate through the recorded epochs for eigenvalues
        for epoch_idx in range(len(epochs)):
            epoch = epochs[epoch_idx]
            if epoch_idx < len(mean_eigenvalue_spectra) and epoch_idx < len(std_eigenvalue_spectra):
                 mean_spectrum = mean_eigenvalue_spectra[epoch_idx] # Mean spectrum (list of mean eigenvalues) for this epoch
                 std_spectrum = std_eigenvalue_spectra[epoch_idx] # Std spectrum (list of std eigenvalues) for this epoch

                 if mean_spectrum is not None and len(mean_spectrum) > 0:
                     # The largest eigenvalue is the last one in the sorted spectrum (assuming eigenvalues are sorted)
                     # However, the 'mean_eigenvalue_spectra' contains the mean of each eigenvalue position across ensemble.
                     # The largest eigenvalue of the MEAN spectrum might not be the mean of the LARGEST eigenvalues.
                     # Let's calculate the mean of the LARGEST eigenvalue across the ensemble from the raw data if needed,
                     # but for now, let's assume mean_eigenvalue_spectra[i] represents the mean of the i-th eigenvalue across the ensemble.
                     # To get the mean of the largest eigenvalue, we need to find the largest value in the mean spectrum.
                     # This is likely what the user intends to visualize from the processed data.

                     mean_largest = np.max(mean_spectrum) # Largest value in the mean spectrum

                     # To get the std dev of the largest eigenvalue across the ensemble,
                     # we would need the original raw eigenvalues.
                     # Since we only saved mean/std of the spectrum, we can't directly get the std of the largest eigenvalue.
                     # Let's plot without error bars for now or find an alternative representation.
                     # Or, if we assume the largest eigenvalue corresponds to a fixed index in the sorted spectrum,
                     # we could use the std at that index. But the order of eigenvalues might change during training.

                     # Let's plot the mean of the largest eigenvalue across the ensemble as calculated before processing.
                     # If we want the std of the largest eigenvalue, we would need the raw ensemble data.
                     # Let's revert to calculating mean/std of the largest eigenvalue from the raw data if necessary,
                     # or just plot the mean of the largest eigenvalue from the mean spectrum.

                     # Let's go back to the previous calculation in a294ae85 before the data saving change:
                     # It iterated through networks and found np.max for each, then calculated mean/std.
                     # With the new data structure, we don't have raw eigenvalues for each network per epoch.

                     # Option 1: Plot the largest value of the mean spectrum. No ensemble std dev of largest eigenvalue.
                     # Option 2: Modify data saving to also save mean/std of LARGEST eigenvalue specifically.

                     # Let's try Option 1 first, as it uses the currently saved data.
                     # We can plot the mean of the largest eigenvalue (which is the largest element in the mean spectrum).
                     # We don't have a direct std dev for this specific value from the saved data.

                     mean_largest_eigenvalues.append(mean_largest)
                     # std_largest_eigenvalues will remain empty or be None
                     training_times.append(epoch * learning_rate) # Calculate training time

                 else:
                     mean_largest_eigenvalues.append(np.nan)
                     training_times.append(epoch * learning_rate)


        # Convert to numpy arrays for plotting and remove NaNs
        training_times_np = np.array(training_times)
        mean_largest_eigenvalues_np = np.array(mean_largest_eigenvalues)


        # Filter out NaNs for plotting
        valid_indices = ~np.isnan(mean_largest_eigenvalues_np)
        plottable_training_times = training_times_np[valid_indices]
        plottable_mean_largest_eigenvalues = mean_largest_eigenvalues_np[valid_indices]


        # Plot mean largest eigenvalue
        if plottable_training_times.size > 0:
            pl.plot(plottable_training_times, plottable_mean_largest_eigenvalues, label=f'Width: {width}')
            # Cannot plot std dev with current saved data structure for largest eigenvalue


    pl.title("Largest Eigenvalue of Mean NTK Spectrum vs Training Time Across Widths") # Updated title
    pl.xlabel("Training Time (Epochs * Learning Rate)")
    pl.ylabel("Largest Eigenvalue of Mean NTK Spectrum (Magnitude)") # Updated ylabel
    pl.yscale('log')
    pl.grid(True)
    pl.legend()
    plot_path_largest_eigenvalue = os.path.join(plot_dir_analysis, "largest_eigenvalue_mean_spectrum_vs_training_time_all_widths.png") # Updated filename
    pl.savefig(plot_path_largest_eigenvalue)
    pl.close()
    print(f"Saved largest eigenvalue of mean NTK spectrum vs training time plot to {plot_path_largest_eigenvalue}")

else:
    print("No data with NTK properties loaded to plot.")