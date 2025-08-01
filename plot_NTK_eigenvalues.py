import pickle
import os
import numpy as np
import matplotlib.pyplot as pl

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

    pl.figure(figsize=(10, 6))

    # Transpose the list of eigenvalue lists to plot each eigenvalue's trajectory over time
    # mean_eigenvalue_spectra is a list of lists: [[eigs@t1], [eigs@t2], ...]
    # We need: [[eig1@t1, eig1@t2, ...], [eig2@t1, eig2@t2, ...], ...]
    # Check if mean_eigenvalue_spectra is not empty and has consistent lengths
    if mean_eigenvalue_spectra and all(len(eig_list) == len(mean_eigenvalue_spectra[0]) for eig_list in mean_eigenvalue_spectra):
        eigenvalues_over_time = list(zip(*mean_eigenvalue_spectra))

        for i, eigenvalue_trajectory in enumerate(eigenvalues_over_time):
            pl.plot(ntk_record_times_eigenvalues, eigenvalue_trajectory, label=f'Eigenvalue {i+1}')

        pl.title(f"Eigenvalues of NTK vs. Training Time (Width {width})")
        pl.xlabel("Training Time (epochs * learning rate)")
        pl.ylabel("Eigenvalue Magnitude (Absolute)")
        pl.yscale('log')
        # pl.legend() # Comment out legend if too many eigenvalues
        pl.grid(True)
        plot_filename = os.path.join(plot_dir, f"eigenvalues_vs_time_width_{width}.png")
        pl.savefig(plot_filename)
        pl.close()
        print(f"Plot saved for width {width} to {plot_filename}")
    else:
        print(f"Skipping plotting for width {width} due to inconsistent or missing eigenvalue data.")

print("\nFinished plotting eigenvalues.")

import pickle
import os
import numpy as np
import matplotlib.pyplot as pl
import torch

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