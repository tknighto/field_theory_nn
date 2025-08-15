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
    pl.plot(inverse_widths_np, std_devs**2, marker='o', linestyle='-', color=colors[i], label=labels[i])


pl.title("Variance of NTK Norm vs 1/Width at Selected Training Times")
pl.xlabel("1 / width")
pl.ylabel("Variance of NTK Norm ")
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

# Determine the maximum training time across all widths
max_training_time = 0
for data in all_widths_data:
    if data.get('ntk_record_times_eigenvalues'):
        max_training_time = max(max_training_time, max(data['ntk_record_times_eigenvalues']))

# Define the four equally spaced training times for plotting
num_plot_times = 4
plot_times = np.linspace(0, max_training_time, num_plot_times).tolist()

# Store the mean standard deviation of eigenvalues at plot_times for each width
mean_std_eigenvalues_at_plot_times_across_widths = [[] for _ in range(num_plot_times)]
inverse_widths = []

for data in all_widths_data:
    width = data['width']
    inverse_widths.append(1 / width)
    std_eigenvalue_spectra_times = data.get('std_eigenvalue_spectra_times', [])
    ntk_record_times_eigenvalues = data.get('ntk_record_times_eigenvalues', [])

    if std_eigenvalue_spectra_times and ntk_record_times_eigenvalues:
        # For each plot_time, find the closest recorded time and get the corresponding mean std eigenvalue
        for i, plot_time in enumerate(plot_times):
            # Find the index of the closest recorded time
            closest_time_index = min(range(len(ntk_record_times_eigenvalues)), key=lambda j: abs(ntk_record_times_eigenvalues[j] - plot_time))

            # Get the list of standard deviations for all eigenvalues at this closest time
            stds_at_closest_time = std_eigenvalue_spectra_times[closest_time_index]

            # Calculate the mean of these standard deviations
            mean_std_at_closest_time = np.mean(stds_at_closest_time) if stds_at_closest_time else np.nan

            mean_std_eigenvalues_at_plot_times_across_widths[i].append(mean_std_at_closest_time)
    else:
        # Append NaN if data is missing for this width
        for i in range(num_plot_times):
            mean_std_eigenvalues_at_plot_times_across_widths[i].append(np.nan)


# Plotting
if inverse_widths and mean_std_eigenvalues_at_plot_times_across_widths:
    pl.figure(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'purple']
    labels = [f'Time ≈ {plot_times[i]:.2f}' for i in range(num_plot_times)]

    for i in range(num_plot_times):
        # Square mean std before plotting
        pl.plot(inverse_widths, (np.array(mean_std_eigenvalues_at_plot_times_across_widths[i]))**2, marker='o', linestyle='-', color=colors[i], label=labels[i])


    pl.title("Mean Variance of NTK Eigenvalues vs 1/Width at Selected Training Times")
    pl.xlabel("1 / width")
    pl.ylabel("(Mean Variance of NTK Eigenvalues")
    pl.grid(True)
    pl.legend()

    plot_dir = "plots/ntk_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, "mean_std_eigenvalues_times_inverse_width_squared_at_selected_times.png")
    pl.savefig(plot_filename)
    pl.close()

    print(f"Plot saved to {plot_filename}")
else:
    print("No valid data to plot.")

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
    if data.get('ntk_record_times_eigenvalues'):
        max_training_time = max(max_training_time, max(data['ntk_record_times_eigenvalues']))

# Define the four equally spaced training times for plotting
num_plot_times = 4
plot_times = np.linspace(0, max_training_time, num_plot_times).tolist()

# Store the mean variance of eigenvalues and propagated standard errors at plot_times for each width
mean_variance_eigenvalues_at_plot_times_across_widths = [[] for _ in range(num_plot_times)]
propagated_std_error_variance_at_plot_times_across_widths = [[] for _ in range(num_plot_times)]
inverse_widths = []

for data in all_widths_data:
    width = data['width']
    inverse_widths.append(1 / width)
    variance_eigenvalue_spectra_times = data.get('variance_eigenvalue_spectra_times', [])
    std_error_std_eigenvalues_times = data.get('std_error_std_eigenvalues_times', []) # Get standard error of std data
    ntk_record_times_eigenvalues = data.get('ntk_record_times_eigenvalues', [])

    if variance_eigenvalue_spectra_times and ntk_record_times_eigenvalues and std_error_std_eigenvalues_times:
        # For each plot_time, find the closest recorded time and get the corresponding mean variance and std error of std
        for i, plot_time in enumerate(plot_times):
            # Find the index of the closest recorded time
            closest_time_index = min(range(len(ntk_record_times_eigenvalues)), key=lambda j: abs(ntk_record_times_eigenvalues[j] - plot_time))

            # Get the list of variances for all eigenvalues at this closest time
            variances_at_closest_time = variance_eigenvalue_spectra_times[closest_time_index]
            # Get the list of standard errors for all eigenvalues at this closest time
            std_errors_std_at_closest_time = std_error_std_eigenvalues_times[closest_time_index]

            # Calculate the mean of these variances
            mean_variance_at_closest_time = np.mean(variances_at_closest_time) if variances_at_closest_time else np.nan

            # Propagate the error for the variance. Variance is the square of the standard deviation (approx).
            # Error propagation for f(x) = x^2 is approx |f'(x)| * error(x) = |2x| * error(x).
            # Here, x is the standard deviation of an eigenvalue, and error(x) is its standard error of the standard deviation.
            # We need the mean standard deviation at this time point across eigenvalues to use in the formula.
            # We saved `std_eigenvalue_spectra_times` which is std dev of each eigenvalue across ensemble.
            # We need the mean of these std devs across eigenvalues. Let's load it.
            std_eigenvalue_spectra_times = data.get('std_eigenvalue_spectra_times', [])
            stds_at_closest_time = std_eigenvalue_spectra_times[closest_time_index]
            mean_std_at_closest_time = np.mean(stds_at_closest_time) if stds_at_closest_time else np.nan


            # Now calculate the propagated error for the mean variance.
            # This is more complex as we are averaging variances.
            # A simpler approach for visualization might be to propagate the error for the mean standard deviation squared.
            # The value plotted is (mean std * 1/width)^2. The error bar for this is 2 * |mean std * 1/width| * error(mean std * 1/width).
            # error(mean std * 1/width) = error(mean std) * 1/width.
            # We need the standard error of the mean standard deviation across eigenvalues.
            # We have saved `std_error_std_eigenvalues_times` which is a list of std errors for each eigenvalue's std.
            # The standard error of the mean of these stds would be np.std(std_errors_std_at_closest_time) / sqrt(num_eigenvalues).
            # Or, we can directly use the mean of the standard errors of the standard deviations as an approximation of the standard error of the mean standard deviation.
            # Let's use the mean of the standard errors of the standard deviations as the error on the mean standard deviation.
            mean_std_error_std_at_closest_time = np.mean(std_errors_std_at_closest_time) if std_errors_std_at_closest_time else np.nan


            # Propagated error for the mean variance (f(x) = x^2, where x is mean std) is approx 2 * |mean_std| * mean_std_error_std
            propagated_error = 2 * np.abs(mean_std_at_closest_time) * mean_std_error_std_at_closest_time if not np.isnan(mean_std_at_closest_time) and not np.isnan(mean_std_error_std_at_closest_time) else np.nan


            mean_variance_eigenvalues_at_plot_times_across_widths[i].append(mean_variance_at_closest_time)
            propagated_std_error_variance_at_plot_times_across_widths[i].append(propagated_error)

    else:
        # Append NaN if data is missing for this width
        for i in range(num_plot_times):
            mean_variance_eigenvalues_at_plot_times_across_widths[i].append(np.nan)
            propagated_std_error_variance_at_plot_times_across_widths[i].append(np.nan)


# Plotting
if inverse_widths and mean_variance_eigenvalues_at_plot_times_across_widths:
    pl.figure(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'purple']
    labels = [f'Time ≈ {plot_times[i]:.2f}' for i in range(num_plot_times)]

    for i in range(num_plot_times):
        # Convert lists to NumPy arrays
        mean_variances = np.array(mean_variance_eigenvalues_at_plot_times_across_widths[i])
        propagated_errors = np.array(propagated_std_error_variance_at_plot_times_across_widths[i])
        inverse_widths_np = np.array(inverse_widths)

        # Plot the mean variances
        pl.plot(inverse_widths_np, mean_variances, marker='o', linestyle='-', color=colors[i], label=labels[i])

        # Add error bars using the propagated error
        pl.errorbar(inverse_widths_np, mean_variances, yerr=propagated_errors, fmt='none', capsize=3, color=colors[i], alpha=0.5)


    pl.title("Mean Variance of NTK Eigenvalues vs 1/Width with Propagated Std Error at Selected Training Times")
    pl.xlabel("1 / width")
    pl.ylabel("Mean Variance of NTK Eigenvalues") # Label updated
    pl.grid(True)
    pl.legend()

    plot_dir = "plots/ntk_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, "mean_variance_eigenvalues_with_propagated_std_error_at_selected_times.png")
    pl.savefig(plot_filename)
    pl.close()

    print(f"Plot saved to {plot_filename}")
else:
    print("No valid data to plot.")


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
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                all_widths_data.append(data)
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")


# Sort data by width
all_widths_data.sort(key=lambda x: x.get('width', float('inf'))) # Use .get with a default for safety

# Determine the maximum training time across all widths for NTK norms
max_training_time_norms = 0
for data in all_widths_data:
    if data.get('ntk_record_times_norms'):
        max_training_time_norms = max(max_training_time_norms, max(data['ntk_record_times_norms']))

# Define the four equally spaced training times for plotting
num_plot_times = 4
plot_times = np.linspace(0, max_training_time_norms, num_plot_times).tolist()

# Store the standard deviations of NTK norms and their standard errors at plot_times for each width
std_ntk_norms_at_plot_times_across_widths = [[] for _ in range(num_plot_times)]
std_error_std_ntk_norms_at_plot_times_across_widths = [[] for _ in range(num_plot_times)]
inverse_widths = []

for data in all_widths_data:
    width = data.get('width')
    if width is None:
        print(f"Skipping data entry with no 'width' key: {data}")
        continue
    inverse_widths.append(1 / width)

    std_ntk_norms_times = data.get('std_ntk_norms_times')
    std_error_std_ntk_norms_times = data.get('std_error_std_ntk_norms_times')
    ntk_record_times_norms = data.get('ntk_record_times_norms')

    if std_ntk_norms_times and ntk_record_times_norms and std_error_std_ntk_norms_times:
        # Find the closest recorded time for each plot_time and get the corresponding std and std error
        for i, plot_time in enumerate(plot_times):
            # Find the index of the closest recorded time
            closest_time_index = min(range(len(ntk_record_times_norms)), key=lambda j: abs(ntk_record_times_norms[j] - plot_time))
            closest_std_norm = std_ntk_norms_times[closest_time_index]
            closest_std_error_std_norm = std_error_std_ntk_norms_times[closest_time_index]

            std_ntk_norms_at_plot_times_across_widths[i].append(closest_std_norm)
            std_error_std_ntk_norms_at_plot_times_across_widths[i].append(closest_std_error_std_norm)
    else:
        # Append NaN if data is missing for this width
        for i in range(num_plot_times):
            std_ntk_norms_at_plot_times_across_widths[i].append(np.nan)
            std_error_std_ntk_norms_at_plot_times_across_widths[i].append(np.nan)

# Plotting
if inverse_widths and std_ntk_norms_at_plot_times_across_widths:
    pl.figure(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'purple']
    labels = [f'Time ≈ {plot_times[i]:.2f}' for i in range(num_plot_times)]

    for i in range(num_plot_times):
        # Convert lists to NumPy arrays
        std_devs = np.array(std_ntk_norms_at_plot_times_across_widths[i])
        std_errors = np.array(std_error_std_ntk_norms_at_plot_times_across_widths[i])
        inverse_widths_np = np.array(inverse_widths)

        # Plot the standard deviations
        pl.plot(inverse_widths_np, std_devs**2, marker='o', linestyle='-', color=colors[i], label=labels[i])

        # Add error bars
        pl.errorbar(inverse_widths_np, std_devs**2, yerr=std_errors, fmt='none', capsize=3, color=colors[i], alpha=0.5)


    pl.title("Variance of NTK Norm vs 1/Width with Std Error at Selected Training Times")
    pl.xlabel("1 / width")
    pl.ylabel("Variance of NTK Norm")
    pl.grid(True)
    pl.legend()

    plot_dir = "plots/ntk_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, "std_ntk_norm_with_std_error_at_selected_times.png")
    pl.savefig(plot_filename)
    pl.close()

    print(f"Plot saved to {plot_filename}")
else:
    print("No valid data to plot.")