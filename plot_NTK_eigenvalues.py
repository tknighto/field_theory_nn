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