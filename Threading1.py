import threading
import numpy as np
import matplotlib.pyplot as pl # Assuming pl is matplotlib.pyplot
import os
# import tempfile # Import tempfile

# Define a lock for thread-safe appending to lists (since lists are shared)
results_lock = threading.Lock()

# Define the function to be threaded (same as before)
def train_model_thread(width, results_list):
    # Ensure necessary imports are within the function if train_model relies on them
    # For this specific case, train_model is defined globally in the first cell,
    # so it should be accessible by the threads.
    print(f"Starting training for width: {width} in thread.") # Added print
    try:
        mean, std = train_model(width)
        print(f"Finished training for width: {width} in thread.") # Added print
        # Use the lock when appending to the shared list
        with results_lock:
            results_list.append((mean, std, width))
    except Exception as e:
        print(f"Error training for width {width}: {e}")
        # Append an error marker or handle as needed
        with results_lock:
            results_list.append((None, None, width))


# List of widths to iterate over
widths = range(5, 50, 10)
results = []
threads = []

print("Starting threaded execution...")

# Create and start threads
for width in widths:
    thread = threading.Thread(target=train_model_thread, args=(width, results))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("Threaded execution finished.")

# Process the results
loss_tot = []
loss_std = []
neurons = []

# Filter out any potential None results if errors occurred
valid_results = [r for r in results if r[0] is not None]

# Sort results based on width to ensure correct plotting order
valid_results.sort(key=lambda x: x[2])

for mean, std, width in valid_results:
    loss_tot.append(mean)
    loss_std.append(std)
    neurons.append(1/width)

if valid_results:
    print("Processing results and plotting...")
    # Create a directory for the final plot
    final_plot_dir = "plots/final_plots"
    os.makedirs(final_plot_dir, exist_ok=True)

    # Plot the results (assuming pl is already imported and configured)
    pl.figure()
    pl.plot(neurons, loss_tot, label="Mean Loss", color="blue")
    loss_tot_np = np.array(loss_tot)
    loss_std_np = np.array(loss_std)
    pl.fill_between(neurons, loss_tot_np - loss_std_np, loss_tot_np + loss_std_np, alpha=0.3, color="blue", label="±1 Std Dev")
    pl.title("Loss vs 1/Width with Std Dev")
    pl.xlabel("1 / width")
    pl.ylabel("Loss")
    pl.grid(True)
    pl.legend()
    final_plot_path = os.path.join(final_plot_dir, "loss_vs_inverse_width_threaded.png")
    pl.savefig(final_plot_path)
    pl.close()
    print("Plotting finished.")

    # # Upload final plot to Google Drive
    # upload_file_to_drive(final_plot_path, folder_id='1-W3bSzD3SnavtR_mr-OIh0IUEcLQHFCX') # Add your folder_id here if needed: folder_id='your_folder_id'

    # # Clean up temporary directory
    # os.remove(final_plot_path)
    # os.rmdir(final_plot_dir)
    # os.rmdir(temp_dir)

else:
    print("No valid results to plot.")