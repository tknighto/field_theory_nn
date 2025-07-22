import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as pl # Assuming pl is matplotlib.pyplot
import torch # Import torch to check CUDA availability
import os
from Train_model import train_model 

# Set the start method for multiprocessing to 'spawn'
# This is necessary to use CUDA with multiprocessing on systems that default to 'fork'
try:
    mp.set_start_method('spawn', force=True)
    print("Multiprocessing start method set to 'spawn'")
except ValueError:
    print("Multiprocessing start method already set to 'spawn'")


# Define the function to be parallelized (same as before)
def train_model_mp(width):
    # Ensure necessary imports are within the function if train_model relies on them
    # For this specific case, train_model is defined globally in the first cell,
    # so it should be accessible by the child processes.
    # Also, ensure the device is set correctly within the child process
    device = torch.device("cpu")
    print(f"Using device: {device} in child process for width {width}") # Added print
    import multiprocessing as mp
    print(mp.cpu_count())

    mean, std = train_model(width)
    return mean, std, width

# List of widths to iterate over
widths = range(5, 50, 15)

print("Starting parallel execution with multiprocessing...")

# Use a multiprocessing Pool
# The number of processes will be determined by the system's CPU count by default
# Using 'spawn' method explicitly when creating the Pool (optional, but can be clearer)
with mp.Pool(processes=mp.cpu_count()) as pool:
    # Use imap_unordered for potentially better memory usage and progress reporting
    results = list(pool.imap_unordered(train_model_mp, widths), widths)

print("Parallel execution finished.")

# Process the results
loss_tot = []
loss_std = []
neurons = []

# Sort results based on width to ensure correct plotting order
results.sort(key=lambda x: x[2])

for mean, std, width in results:
    loss_tot.append(mean)
    loss_std.append(std)
    neurons.append(1/width)

print("Processing results and plotting...")
# Plot the results (assuming pl is already imported and configured)
pl.figure()
pl.plot(neurons, loss_tot, label="Mean Loss", color="blue")
loss_tot_np = np.array(loss_tot)
loss_std_np = np.array(loss_std)
pl.fill_between(neurons, loss_tot_np - loss_tot_np, loss_tot_np + loss_std_np, alpha=0.3, color="blue", label="±1 Std Dev")
pl.title("Loss vs 1/Width with Std Dev")
pl.xlabel("1 / width")
pl.ylabel("Loss")
pl.grid(True)
pl.legend()
pl.savefig("loss_vs_inverse_width_mp.png")
pl.close()
print("Plotting finished.")