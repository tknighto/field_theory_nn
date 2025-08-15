import pickle
import os
import matplotlib.pyplot as pl
import numpy as np

data_dir = "loss_data"
widths = range(5, 90, 20) # Use the same widths as in the training code

plot_dir = "plots/ntk_norm_plots"
os.makedirs(plot_dir, exist_ok=True)

pl.figure(figsize=(10, 6))

for width in widths:
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    if not os.path.exists(data_filename):
        print(f"Data file not found for width {width}: {data_filename}")
        continue

    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    mean_ntk_norms = data.get('ntk_norms_times')
    ntk_record_times = data.get('ntk_record_times_norms')
    std_ntk_norms = data.get('std_ntk_norms_times')


    if mean_ntk_norms is None or ntk_record_times is None or len(mean_ntk_norms) == 0 or len(ntk_record_times) == 0:
        print(f"No NTK norm data available for width {width}.")
        continue

    print(f"\n--- Plotting NTK Norm for Width {width} ---")

    # Plot the mean NTK norm
    inverse_width = 1 / width
    pl.plot(ntk_record_times, mean_ntk_norms, label=f'Width {width}')

    # Add shaded area for standard deviation if available and lengths match
    if std_ntk_norms is not None and len(std_ntk_norms) == len(mean_ntk_norms):
        mean_ntk_norms_np = np.array(mean_ntk_norms)
        std_ntk_norms_np = np.array(std_ntk_norms)
        # Ensure lower bound is non-negative for log scale plotting if needed later
        lower_bound = np.maximum(0, mean_ntk_norms_np - std_ntk_norms_np)
        pl.fill_between(ntk_record_times, lower_bound, (mean_ntk_norms_np + std_ntk_norms_np), alpha=0.3)
        print(f"Added standard deviation fill for width {width}.")
    elif std_ntk_norms is not None:
        print(f"Warning: Length mismatch between mean and std NTK norms for width {width}. Skipping std fill.")


pl.title("NTK Frobenius Norm vs. Training Time")
pl.xlabel("Training Time (epochs * learning rate)")
pl.ylabel("NTK Frobenius Norm")
pl.yscale('log') # Use log scale for better visualization
pl.legend()
pl.grid(True)

plot_filename = os.path.join(plot_dir, "ntk_norm_vs_time.png")
pl.savefig(plot_filename)
pl.close()

print("\nFinished plotting NTK Frobenius Norm vs. Training Time.")