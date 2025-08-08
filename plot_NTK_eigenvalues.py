import pickle
import os
import numpy as np
import matplotlib.pyplot as pl
import torch

data_dir = "loss_data"
widths = range(5, 50, 20) # Use the same widths as in the training code

plot_dir = "plots/eigenvalue_plots"
os.makedirs(plot_dir, exist_ok=True)

for width in widths:
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    if not os.path.exists(data_filename):
        print(f"Data file not found for width {width}: {data_filename}")
        continue

    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    mean_eigenvalue_spectra = data.get('mean_eigenvalue_spectra_times')
    ntk_record_times_eigenvalues = data.get('ntk_record_times_eigenvalues')

    if mean_eigenvalue_spectra is None or ntk_record_times_eigenvalues is None or len(mean_eigenvalue_spectra) == 0:
        print(f"No eigenvalue data available for width {width}.")
        continue

    print(f"\n--- Plotting Eigenvalues for Width {width} ---")

    # --- Plot with Log Scale X-axis (Full Range) ---
    # Keep the log scale plot for the full range as it might still be useful
    pl.figure(figsize=(10, 6))

    # Transpose the list of eigenvalue lists to plot each eigenvalue's trajectory over time
    # mean_eigenvalue_spectra is a list of lists: [[eigs@t1], [eigs@t2], ...]
    # We need: [[eig1@t1, eig1@t2, ...], [eig2@t1, eig2@t2, ...], ...]
    if mean_eigenvalue_spectra and all(len(eig_list) == len(mean_eigenvalue_spectra[0]) for eig_list in mean_eigenvalue_spectra):
        eigenvalues_over_time = list(zip(*mean_eigenvalue_spectra))

        for i, eigenvalue_trajectory in enumerate(eigenvalues_over_time):
            pl.plot(ntk_record_times_eigenvalues, eigenvalue_trajectory, label=f'Eigenvalue {i+1}')

        pl.title(f"Eigenvalues of NTK vs. Training Time (Log Scale X) (Width {width})")
        pl.xlabel("Training Time (epochs * learning rate)")
        pl.ylabel("Eigenvalue Magnitude (Absolute)")
        pl.yscale('log')  # Use log scale for better visualization
        # pl.legend() # Comment out legend if too many eigenvalues
        pl.grid(True)
        plot_filename_log = os.path.join(plot_dir, f"eigenvalues_vs_time_width_{width}.png")
        pl.savefig(plot_filename_log)
        pl.close()
        print(f"Full plot saved for width {width} to {plot_filename_log}")
    else:
        print(f"Skipping full range plotting for width {width} due to inconsistent or missing eigenvalue data.")

    # --- Plot with Linear Scale X-axis (Zoomed-in Early Times) ---
    # Create a new figure specifically for the zoomed-in view
    if mean_eigenvalue_spectra and all(len(eig_list) == len(mean_eigenvalue_spectra[0]) for eig_list in mean_eigenvalue_spectra):
        pl.figure(figsize=(10, 6))

        eigenvalues_over_time = list(zip(*mean_eigenvalue_spectra))

        # Determine a reasonable x-limit for the zoomed-in plot (e.g., first few training time units)
        early_time_limit = 1000 # Example: Focus on the first 5 training time units
        # Adjust the limit based on the actual recorded times if necessary to ensure points are included
        if ntk_record_times_eigenvalues and len(ntk_record_times_eigenvalues) > 0:
             # Find the largest recorded time that is less than or equal to early_time_limit,
             # or take a few points if the first few times are very small.
             valid_early_times_indices = [i for i, t in enumerate(ntk_record_times_eigenvalues) if t <= early_time_limit]
             if valid_early_times_indices:
                  zoom_end_index = valid_early_times_indices[-1] + 1 # Include the point at or below the limit
             elif ntk_record_times_eigenvalues: # If no times <= limit, just take the first point if exists
                 zoom_end_index = 1
             else: # No recorded times
                 zoom_end_index = 0

             # Ensure we have at least two points to plot a line
             if zoom_end_index < 2 and len(ntk_record_times_eigenvalues) >= 2:
                 # If not enough points within the strict limit, take the first two points if available
                 zoom_end_index = 2
             elif zoom_end_index < 1 and len(ntk_record_times_eigenvalues) >= 1:
                 zoom_end_index = 1


        else: # No recorded times at all
             zoom_end_index = 0


        zoomed_times = ntk_record_times_eigenvalues[:zoom_end_index]

        if len(zoomed_times) > 1: # Only plot if there are at least two time points in the zoomed range
            for i, eigenvalue_trajectory in enumerate(eigenvalues_over_time):
                zoomed_eigenvalue_trajectory = eigenvalue_trajectory[:zoom_end_index]
                pl.plot(zoomed_times, zoomed_eigenvalue_trajectory, label=f'Eigenvalue {i+1}')

            pl.title(f"Eigenvalues of NTK vs. Training Time (Zoomed-in Early Times) (Width {width})")
            pl.xlabel("Training Time (epochs * learning rate)")
            pl.ylabel("Eigenvalue Magnitude (Absolute)")
            pl.yscale('log') # Keep y-scale as log if eigenvalues vary greatly
            pl.xscale('linear') # Use linear scale for zoomed-in view
            # Set x-axis limit to the maximum time in the zoomed range
            pl.xlim(0, max(zoomed_times) if zoomed_times else early_time_limit)
            # pl.legend() # Comment out legend if too many eigenvalues
            pl.grid(True)
            plot_filename_linear = os.path.join(plot_dir, f"eigenvalues_vs_time_linear_x_early_width_{width}.png")
            pl.savefig(plot_filename_linear)
            pl.close()
            print(f"Linear scale X (zoomed) plot saved for width {width} to {plot_filename_linear}")
        else:
            print(f"Not enough data points in the early time range ({early_time_limit} or first points) to create a zoomed-in plot for width {width}.")
            pl.close() # Close the empty figure

    else:
        print(f"Skipping zoomed-in plotting for width {width} due to inconsistent or missing eigenvalue data.")


print("\nFinished plotting eigenvalues.")

import pickle
import os
import numpy as np
import matplotlib.pyplot as pl


data_dir = "loss_data"
widths = range(5, 50, 20) # Use the same widths as in the training code

plot_dir = "plots/ntk_change_plots"
os.makedirs(plot_dir, exist_ok=True)

pl.figure(figsize=(10, 6))

for width in widths:
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    if not os.path.exists(data_filename):
        print(f"Data file not found for width {width}: {data_filename}")
        continue

    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    mean_ntk_matrices_times = data.get('mean_ntk_matrices_times')
    ntk_record_times_matrices = data.get('ntk_record_times_matrices')

    if mean_ntk_matrices_times is None or ntk_record_times_matrices is None or len(mean_ntk_matrices_times) < 2:
        print(f"Not enough NTK matrix data available for width {width} to compute change.")
        continue

    relative_norm_changes = []
    times_for_change = []

    # Iterate through recorded times to compute the change
    for i in range(len(mean_ntk_matrices_times) - 1):
        ntk_t = torch.tensor(mean_ntk_matrices_times[i])
        ntk_t_plus_1 = torch.tensor(mean_ntk_matrices_times[i+1])
        time_t = ntk_record_times_matrices[i]

        # Compute the difference and its Frobenius norm
        diff_norm = torch.linalg.norm(ntk_t_plus_1 - ntk_t, ord='fro').item()

        # Compute the Frobenius norm at time t
        norm_t = torch.linalg.norm(ntk_t, ord='fro').item()

        # Compute the relative change
        if norm_t > 0: # Avoid division by zero
            relative_change = diff_norm / norm_t
            relative_norm_changes.append(relative_change)
            times_for_change.append(time_t) # Record the time at the start of the interval

    if relative_norm_changes:
        pl.plot(times_for_change, relative_norm_changes, label=f'Width {width}')
        print(f"Computed and stored {len(relative_norm_changes)} relative norm changes for width {width}.")
    else:
        print(f"No relative norm changes computed for width {width}.")


pl.title("Relative Change in NTK Frobenius Norm vs. Training Time")
pl.xlabel("Training Time (epochs * learning rate)")
pl.ylabel("Relative Change in Norm (||K(t+1)-K(t)|| / ||K(t)||)")
pl.yscale('log') # Use log scale for better visualization of changes
pl.legend()
pl.grid(True)

plot_filename = os.path.join(plot_dir, "relative_ntk_norm_change_vs_time.png")
pl.savefig(plot_filename)
pl.close()

print("\nFinished plotting relative NTK norm change.")

import pickle
import os
import numpy as np
import matplotlib.pyplot as pl
import torch

data_dir = "loss_data"
widths = range(5, 50, 20) # Use the same widths as in the training code

plot_dir = "plots/loss_ntk_combined_plots"
os.makedirs(plot_dir, exist_ok=True)

for width in widths:
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    if not os.path.exists(data_filename):
        print(f"Data file not found for width {width}: {data_filename}")
        continue

    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    # Get loss data (using mean 1000-epoch losses for comparison with NTK times)
    mean_1000_epoch_losses = data.get('mean_1000_epoch_losses')
    epochs_recorded_at = data.get('epochs_recorded_at')
    # Need to convert epochs to training time for the x-axis
    # Assuming LEARNING_RATE is consistent with how training times were recorded for NTK
    # We'll need to retrieve the LEARNING_RATE used for this width.
    # This is not directly saved, so let's assume we can calculate it as 0.15 / width
    learning_rate = 0.15 / width
    loss_training_times = [(epoch + 1) * learning_rate for epoch in epochs_recorded_at]


    # Get NTK matrix data and compute relative norm changes
    mean_ntk_matrices_times = data.get('mean_ntk_matrices_times')
    ntk_record_times_matrices = data.get('ntk_record_times_matrices')

    if mean_1000_epoch_losses is None or loss_training_times is None or \
       mean_ntk_matrices_times is None or ntk_record_times_matrices is None or \
       len(mean_ntk_matrices_times) < 2:
        print(f"Not enough data available for width {width} to plot combined loss and NTK change.")
        continue

    relative_ntk_norm_changes = []
    times_for_ntk_change = []

    # Compute relative change in NTK Frobenius norm
    for i in range(len(mean_ntk_matrices_times) - 1):
        ntk_t = torch.tensor(mean_ntk_matrices_times[i])
        ntk_t_plus_1 = torch.tensor(mean_ntk_matrices_times[i+1])
        time_t = ntk_record_times_matrices[i]

        diff_norm = torch.linalg.norm(ntk_t_plus_1 - ntk_t, ord='fro').item()
        norm_t = torch.linalg.norm(ntk_t, ord='fro').item()

        if norm_t > 0:
            relative_change = diff_norm / norm_t
            relative_ntk_norm_changes.append(relative_change)
            times_for_ntk_change.append(time_t)

    if not relative_ntk_norm_changes:
         print(f"No relative NTK norm changes computed for width {width}.")
         continue

    print(f"\n--- Plotting Loss and NTK Change for Width {width} ---")

    fig, ax1 = pl.subplots(figsize=(10, 6))

    # Plotting Loss on the first y-axis
    ax1.plot(loss_training_times, mean_1000_epoch_losses, 'b-', label='Mean 1000-Epoch Training Loss')
    ax1.set_xlabel("Training Time (epochs * learning rate)")
    ax1.set_ylabel("Loss", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Creating a second y-axis for NTK change
    ax2 = ax1.twinx()
    ax2.plot(times_for_ntk_change, relative_ntk_norm_changes, 'r-', label='Relative NTK Norm Change')
    ax2.set_ylabel("Relative NTK Norm Change", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')


    # Add a title and legends
    pl.title(f"Loss and Relative NTK Norm Change vs. Training Time (Width {width})")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)


    plot_filename = os.path.join(plot_dir, f"loss_ntk_combined_width_{width}.png")
    pl.savefig(plot_filename)
    pl.close(fig) # Close the figure to prevent it from displaying inline

    print(f"Combined plot saved for width {width} to {plot_filename}")

print("\nFinished plotting combined loss and relative NTK norm change.")