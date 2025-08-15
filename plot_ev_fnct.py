import pickle
import os
import matplotlib.pyplot as pl
import numpy as np

# Directory where the loss data is saved
data_dir = "loss_data"

# Collect data for all widths
all_widths_data = []
for filename in os.listdir(data_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_widths_data.append(data)

# Sort data by width
all_widths_data.sort(key=lambda x: x.get('width', 0))

pl.figure(figsize=(12, 8))

# Iterate through each width's data
for data in all_widths_data:
    width = data.get('width')
    mean_eigenvalue_spectra_times = data.get('mean_eigenvalue_spectra_times')
    ntk_record_times_eigenvalues = data.get('ntk_record_times_eigenvalues')

    if width is None or mean_eigenvalue_spectra_times is None or ntk_record_times_eigenvalues is None or not ntk_record_times_eigenvalues or not mean_eigenvalue_spectra_times:
        print(f"Skipping width {width}: Missing or incomplete required data.")
        continue

    print(f"Processing eigenvalue data for width: {width} for combined plot.")

    # Find the index corresponding to the initial time
    initial_time_index = 0

    # Get the mean eigenvalue spectrum at the initial time
    initial_eigenvalue_spectrum = mean_eigenvalue_spectra_times[initial_time_index]

    if not initial_eigenvalue_spectrum:
         print(f"No initial eigenvalue spectrum found for width {width}. Skipping plotting.")
         continue

    num_eigenvalues = len(initial_eigenvalue_spectrum)

    # Determine the indices of the top 3 largest eigenvalues
    if num_eigenvalues >= 3:
        # Get the indices that would sort the eigenvalues in descending order
        sorted_initial_eigenvalue_indices_desc = np.argsort(initial_eigenvalue_spectrum)[::-1]
        selected_eigenvalue_indices = sorted_initial_eigenvalue_indices_desc[:3]
    elif num_eigenvalues > 0:
         # If less than 3 eigenvalues, select all available
         selected_eigenvalue_indices = np.argsort(initial_eigenvalue_spectrum)[::-1]
         print(f"Warning: Only {num_eigenvalues} eigenvalues available for width {width}. Plotting all.")
    else:
        print(f"No eigenvalues found for width {width}. Skipping plotting.")
        continue


    # Plot f(lambda, t) for the selected eigenvalues
    eigenvalue_labels = {
        selected_eigenvalue_indices[0]: 'Top 1st',
        selected_eigenvalue_indices[1]: 'Top 2nd',
        selected_eigenvalue_indices[2]: 'Top 3rd'
    }


    for eigenvalue_index in selected_eigenvalue_indices:
        f_lambda_t_values = []
        training_times_for_plot = []

        for time_idx, t in enumerate(ntk_record_times_eigenvalues):
            if time_idx < len(mean_eigenvalue_spectra_times) and eigenvalue_index < len(mean_eigenvalue_spectra_times[time_idx]):
                lambda_t = mean_eigenvalue_spectra_times[time_idx][eigenvalue_index]
                lambda_0 = initial_eigenvalue_spectrum[eigenvalue_index]
                f_lambda_t =  (lambda_t - lambda_0)*width
                f_lambda_t_values.append(f_lambda_t)
                training_times_for_plot.append(t)
            else:
                 print(f"Warning: Inconsistent data length for eigenvalue {eigenvalue_index} at time index {time_idx} for width {width}. Stopping for this eigenvalue.")
                 break # Stop processing for this eigenvalue if data is inconsistent

        if training_times_for_plot:
             label = f'Width {width}: {eigenvalue_labels.get(eigenvalue_index, eigenvalue_index+1)}' # Use descriptive label
             pl.plot(training_times_for_plot, f_lambda_t_values, label=label)


pl.title('$f(\lambda, t) = width \times (\lambda(t) - \lambda(0))$ vs. Training Time (Selected Eigenvalues)')
pl.xlabel("Training Time")
pl.ylabel("$f(\lambda, t)$")
pl.legend()
pl.grid(True)

# Save the combined plot
plot_eigenvalues_dir = "plots/eigenvalue_functions"
os.makedirs(plot_eigenvalues_dir, exist_ok=True)
plot_filename = os.path.join(plot_eigenvalues_dir, "combined_eigenvalue_functions_top3.png") # Changed filename
pl.savefig(plot_filename)
pl.close()
print(f"Saved combined eigenvalue function plot to {plot_filename}")

print("\nFinished plotting combined eigenvalue functions.")