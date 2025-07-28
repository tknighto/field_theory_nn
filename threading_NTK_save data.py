import torch.func # Keep import just in case needed elsewhere, although not for this NTK method
import torch.autograd as autograd # Import autograd for gradient computation

def compute_ntk(model, x):
    """
    Computes the Neural Tangent Kernel (NTK) matrix for a given model and input data
    by computing gradients individually.

    Args:
        model: The neural network model (torch.nn.Module).
        x: The input data (torch.Tensor).

    Returns:
        The NTK matrix (torch.Tensor).
    """
    # Ensure model is in training mode
    model.train()

    batch_size = x.size(0)
    with torch.no_grad():
         output_dim = model(x[:1]).size(1) # Determine output dimension

    # List to store the gradients for each input
    jacobians_list = []

    # Compute gradients for each input data point individually
    for i in range(batch_size):
        x_i = x[i:i+1] # Get a single input data point (maintain batch dimension)
        output_i = model(x_i) # Get the output for this input

        # Compute gradients of the output with respect to all model parameters
        # output_i should be a scalar or have grad_outputs provided if not scalar
        # For output_dim = 1, output_i is (1, 1), which can be treated as scalar for grad.
        # For output_dim > 1, need to sum or provide grad_outputs.
        # For NTK, we need Jacobian d(output_i)/d(params).
        # grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False)
        # outputs: output_i (shape 1, output_dim)
        # inputs: tuple(model.parameters())
        # grad_outputs: needed if output_dim > 1. Should be a tensor of ones matching output_i shape.

        if output_dim > 1:
             grad_outputs = torch.ones_like(output_i)
        else:
             grad_outputs = None # For scalar output

        # Compute gradients of output_i with respect to all parameters
        # This returns a tuple of gradients, one for each parameter tensor.
        gradients = autograd.grad(outputs=output_i,
                                  inputs=model.parameters(),
                                  grad_outputs=grad_outputs,
                                  retain_graph=True if i < batch_size - 1 else None, # Retain graph for all but the last iteration
                                  create_graph=False)

        # Flatten and concatenate gradients for this input point to form a row (or block of rows if output_dim > 1) of the Jacobian
        flattened_gradients = torch.cat([grad.view(-1) for grad in gradients]) # Shape (total_params,)

        # If output_dim > 1, gradients are computed separately for each output dimension, resulting in (output_dim * total_params).
        # The above `torch.cat` already flattens all gradients. If output_dim > 1, and we compute grad w.r.t parameters for a (1, output_dim) output,
        # the gradients will have shape (output_dim, *parameter_shape). Flattening and concatenating will give (output_dim * total_params).
        # We need the Jacobian to have shape (batch_size * output_dim, total_params).
        # Let's reshape flattened_gradients if output_dim > 1

        if output_dim > 1:
            # Reshape from (output_dim * total_params) to (output_dim, total_params) and then append
             reshaped_gradients = flattened_gradients.view(output_dim, -1)
             jacobians_list.append(reshaped_gradients)
        else: # output_dim == 1
             # Shape is already (total_params,)
             jacobians_list.append(flattened_gradients.unsqueeze(0)) # Add a batch dimension to make it (1, total_params)

    # Stack the gradients for all input points to form the Jacobian matrix J
    # If output_dim > 1, each element in jacobians_list is (output_dim, total_params), stack gives (batch_size * output_dim, total_params)
    # If output_dim == 1, each element is (1, total_params), stack gives (batch_size, total_params)
    J = torch.cat(jacobians_list, dim=0)

    # Compute the NTK matrix: J @ J^T
    ntk_matrix = J @ J.T # Shape (batch_size * output_dim, batch_size * output_dim) or (batch_size, batch_size) if output_dim == 1

    return ntk_matrix

def compute_ntk_properties(ntk_matrix):
    """
    Computes the Frobenius norm and eigenvalues of an NTK matrix.

    Args:
        ntk_matrix: The NTK matrix (torch.Tensor).

    Returns:
        A tuple containing:
            - frobenius_norm: The Frobenius norm of the NTK matrix (float).
            - eigenvalues: The eigenvalues of the NTK matrix (torch.Tensor).
    """
    # Calculate the Frobenius norm
    frobenius_norm = torch.linalg.norm(ntk_matrix, ord='fro').item()

    # Compute the eigenvalues
    eigenvalues = torch.linalg.eigvals(ntk_matrix)

    return frobenius_norm, eigenvalues

    import torch.optim as optim
import time
import torch.nn as nn
import multiprocessing as mp
import torch
import threading
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.special import legendre
import pickle # Import pickle for saving data

# Redefine necessary global variables from the first cell
ENSEMBLE_SIZE = 10
NUM_LAYERS = 1

# Redefine the true function and data
def true_function(x):
    P = legendre(4)
    x_np = x.numpy().squeeze()
    y_np = P(x_np)
    return torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

x_train = torch.linspace(-1, 1, 24).unsqueeze(1)
y_train = true_function(x_train)

# Now split the data
from sklearn.model_selection import train_test_split

x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print("x_train_split shape:", x_train_split.shape)
print("x_test_split shape:", x_test_split.shape)
print("y_train_split shape:", y_train_split.shape)
print("y_test_split shape:", y_test_split.shape)


# Define a lock for thread-safe appending to lists (since lists are shared)
results_lock = threading.Lock()

# Define the function to be threaded (same as before)
def train_model_thread(width, results_list):
    print(f"Starting training for width: {width} in thread.")
    try:
        # train_model now returns mean_interval_losses, std_interval_losses, final_loss, final_std_loss, mean_1000_epoch_losses, std_1000_epoch_losses, mean_1000_epoch_test_losses, std_1000_epoch_test_losses
        mean_interval, std_interval, final_mean, final_std, mean_1000_epoch, std_1000_epoch, mean_1000_epoch_test, std_1000_epoch_test = train_model(width)
        print(f"Finished training for width: {width} in thread.")
        with results_lock:
            # Store all returned values along with the width
            results_list.append((mean_interval, std_interval, final_mean, final_std, mean_1000_epoch, std_1000_epoch, mean_1000_epoch_test, std_1000_epoch_test, width))
    except Exception as e:
        print(f"Error training for width {width}: {e}")
        with results_lock:
            results_list.append((None, None, None, None, None, None, None, None, width))


# Define the train_network function with 1000 epoch loss recording and time-dependent NTK computation
def train_network(model, x_train_split, y_train_split, x_test_split, y_test_split, learning_rate, num_epochs, device, weights):
    # Define the loss functions
    weighted_criterion = nn.MSELoss(reduction='none') # For training loss (unreduced)
    standard_criterion = nn.MSELoss() # For test loss (reduced to mean)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

    # Move model and criteria to the device
    model.to(device)
    weighted_criterion.to(device)
    standard_criterion.to(device)

    # Move data and weights to the device
    x_train_device = x_train_split.to(device)
    y_train_device = y_train_split.to(device)
    x_test_device = x_test_split.to(device)
    y_test_device = y_test_split.to(device)
    weights_device = weights.to(device) # Move weights to device


    train_losses = [] # List to store training losses for each epoch
    test_losses = [] # List to store test losses for each epoch
    interval_losses = [] # Losses at specific intervals (0%, 25%, etc.)
    losses_1000_epochs = [] # Losses at every 1000 epochs
    test_losses_1000_epochs = [] # Test losses at every 1000 epochs
    ntk_norms_epochs = [] # List to store NTK norms at recorded epochs
    ntk_eigenvalues_epochs = [] # List to store NTK eigenvalues (as lists) at recorded epochs


    model.train()  # Set model to training mode

    total_epochs = num_epochs

    record_epochs = [
        0,
        total_epochs // 4,
        total_epochs // 2,
        total_epochs * 3 // 4,
        total_epochs - 1
    ]

    record_epochs = sorted(list(set([max(0, min(total_epochs - 1, ep)) for ep in record_epochs])))

    # Define epochs at which to compute NTK (every 1000 epochs)
    ntk_record_epochs = range(0, total_epochs, 1000)
    # Ensure the last epoch is included if it's not a multiple of 1000
    if (total_epochs - 1) not in ntk_record_epochs:
        ntk_record_epochs = sorted(list(ntk_record_epochs) + [total_epochs - 1])
    else:
         ntk_record_epochs = list(ntk_record_epochs)


    for epoch in range(num_epochs):

        optimizer.zero_grad()
        model.train()  # Set model to training mode

        # Forward pass
        output = model(x_train_device)

        # Apply weighted MSE loss for training
        loss_elements = weighted_criterion(output, y_train_device)
        weighted_loss = (loss_elements * weights_device).mean()  # Element-wise weighting

        # Backward and optimize
        weighted_loss.backward()
        optimizer.step()

        train_losses.append(weighted_loss.item())

        if epoch in record_epochs:
            interval_losses.append(weighted_loss.item())

        if epoch % 1000 == 0:
            losses_1000_epochs.append(weighted_loss.item())

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Loss: {weighted_loss.item():.8f}")

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_output = model(x_test_device)
            test_loss = standard_criterion(test_output, y_test_device)  # Use standard criterion for test loss
            test_losses.append(test_loss.item())

            if epoch % 1000 == 0:
                test_losses_1000_epochs.append(test_loss.item())

        model.train() # Set model back to training mode

        # Compute NTK properties at specified epochs
        if epoch in ntk_record_epochs:
            print(f"Computing NTK at epoch {epoch}...")
            # Ensure model is on CPU for NTK computation if compute_ntk expects CPU tensors
            model.cpu()
            # compute_ntk should take model statedict and input data
            # or take the model and ensure inputs are on the same device
            # Let's assume compute_ntk handles device internally or expects CPU
            ntk_matrix = compute_ntk(model, x_train_split.cpu()) # Pass CPU data for NTK
            ntk_norm, ntk_eigenvalues = compute_ntk_properties(ntk_matrix)
            ntk_norms_epochs.append(ntk_norm)
            ntk_eigenvalues_epochs.append(ntk_eigenvalues.tolist()) # Convert eigenvalues to list for saving
            print(f"Finished computing NTK at epoch {epoch}.")
            model.to(device) # Move model back to original device


        # Optional: print loss every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {weighted_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")


    # Return the final training loss and other recorded data
    return model, interval_losses, train_losses[-1], losses_1000_epochs, test_losses_1000_epochs, ntk_norms_epochs, ntk_eigenvalues_epochs, ntk_record_epochs


# Update train_model to collect and save 1000-epoch losses and time-dependent NTK properties, and calculate mean/std of final losses
def train_model(width):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(mp.cpu_count())

    NUM_EPOCHS = 1000* width
    LEARNING_RATE = 0.15 / width
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    NEURONS_PER_LAYER = width

    plot_dir = f"plots/width_{width}"
    os.makedirs(plot_dir, exist_ok=True)

    class SimpleNet(nn.Module):
        def __init__(self, num_layers, neurons_per_layer):
            super(SimpleNet, self).__init__()
            layers = []

            input_dim = 1
            for _ in range(num_layers):
                layer = nn.Linear(input_dim, neurons_per_layer)
                nn.init.normal_(layer.weight, mean=0.0, std=1.0 / width)
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)
                layers.append(layer)
                layers.append(nn.Tanh())
                input_dim = neurons_per_layer

            final_layer = nn.Linear(input_dim, 1)
            nn.init.normal_(final_layer.weight, mean=0.0, std=1.0 / width)
            nn.init.normal_(final_layer.bias, mean=0.0, std=1.0)
            layers.append(final_layer)

            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # --- NTK Computation and Properties at Initialization ---
    # This part is now for NTK at initialization, computed once.
    # The time-dependent NTK will be computed within train_network.
    # Instantiate the model for initialization NTK
    # init_ntk_model = SimpleNet(NUM_LAYERS, NEURONS_PER_LAYER)
    # init_ntk_model.to(device) # Move model to device for NTK computation
    # print(f"Computing NTK at initialization for width: {width}...")
    # init_ntk_matrix = compute_ntk(init_ntk_model, x_train_split.to(device))
    # print(f"Finished computing NTK at initialization for width: {width}.")
    # init_ntk_norm, init_ntk_eigenvalues = compute_ntk_properties(init_ntk_matrix)
    # print(f"Finished computing NTK properties at initialization for width: {width}.")
    # --- End NTK at Initialization ---


    ensemble_outputs = []
    ensemble_interval_losses = []
    all_ensemble_final_losses = [] # To store final training loss for each network
    all_ensemble_1000_epoch_losses = []
    all_ensemble_1000_epoch_test_losses = []
    all_ensemble_ntk_norms = [] # To store time-dependent NTK norms for the ensemble
    all_ensemble_ntk_eigenvalues = [] # To store time-dependent NTK eigenvalues for the ensemble
    ntk_record_epochs_list = [] # To store the list of epochs where NTK was recorded


    for i in range(ENSEMBLE_SIZE):
        print(f"Training network {i+1}/{ENSEMBLE_SIZE}")
        start_time = time.time()
        model = SimpleNet(NUM_LAYERS, NEURONS_PER_LAYER)

        weights = torch.abs(y_train_split) + 1e-6
        weights = weights / weights.sum()

        test_weights = torch.abs(y_test_split) + 1e-6
        test_weights = test_weights / test_weights.sum()

        # train_network now returns final_loss and time-dependent NTK data
        trained_model, interval_losses, final_loss, losses_1000_epochs, test_losses_1000_epochs, ntk_norms_epochs, ntk_eigenvalues_epochs, ntk_record_epochs = train_network(model, x_train_split, y_train_split, x_test_split, y_test_split, LEARNING_RATE, NUM_EPOCHS, device, weights)

        ensemble_interval_losses.append(interval_losses)
        all_ensemble_final_losses.append(final_loss) # Append final loss for this network
        all_ensemble_1000_epoch_losses.append(losses_1000_epochs)
        all_ensemble_1000_epoch_test_losses.append(test_losses_1000_epochs)
        all_ensemble_ntk_norms.append(ntk_norms_epochs) # Append time-dependent NTK norms
        all_ensemble_ntk_eigenvalues.append(ntk_eigenvalues_epochs) # Append time-dependent NTK eigenvalues
        ntk_record_epochs_list.append(ntk_record_epochs) # Store the recorded epochs list


        with torch.no_grad():
            output = trained_model(x_train.to(device)).cpu()
            ensemble_outputs.append(output.numpy())

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for network {i+1}: {elapsed_time:.2f} seconds")

    if all_ensemble_final_losses: # Check if any networks trained successfully
         mean_final_loss = np.mean(all_ensemble_final_losses) # Calculate mean of final losses
         std_final_loss = np.std(all_ensemble_final_losses) # Calculate std of final losses
    else:
         mean_final_loss = None
         std_final_loss = None


    ensemble_outputs = np.stack(ensemble_outputs, axis=0)

    mean_output = np.mean(ensemble_outputs, axis=0).squeeze()
    std_output = np.std(ensemble_outputs, axis=0).squeeze()

    ensemble_interval_losses_np = np.array(ensemble_interval_losses)
    mean_interval_losses = np.mean(ensemble_interval_losses_np, axis=0)
    std_interval_losses = np.std(ensemble_interval_losses_np, axis=0)

    all_ensemble_1000_epoch_losses_np = np.array(all_ensemble_1000_epoch_losses)
    mean_1000_epoch_losses = np.mean(all_ensemble_1000_epoch_losses_np, axis=0)
    std_1000_epoch_losses = np.std(all_ensemble_1000_epoch_losses_np, axis=0)

    all_ensemble_1000_epoch_test_losses_np = np.array(all_ensemble_1000_epoch_test_losses)
    mean_1000_epoch_test_losses = np.mean(all_ensemble_1000_epoch_test_losses_np, axis=0)
    std_1000_epoch_test_losses = np.std(all_ensemble_1000_epoch_test_losses_np, axis=0)

    # Calculate mean and std for time-dependent NTK norms across the ensemble
    if all_ensemble_ntk_norms:
        # Before calculating mean/std, ensure all lists within all_ensemble_ntk_norms have the same length
        min_norm_epochs = min(len(norm_list) for norm_list in all_ensemble_ntk_norms)
        if not all(len(norm_list) == min_norm_epochs for norm_list in all_ensemble_ntk_norms):
             print(f"Warning: NTK norms lists for width {width} have inconsistent lengths across networks. Truncating to minimum length ({min_norm_epochs}).")
             all_ensemble_ntk_norms = [norm_list[:min_norm_epochs] for norm_list in all_ensemble_ntk_norms]

        mean_ntk_norms = np.mean(np.array(all_ensemble_ntk_norms), axis=0)
        std_ntk_norms = np.std(np.array(all_ensemble_ntk_norms), axis=0)
        recorded_ntk_epochs = ntk_record_epochs_list[0][:min_norm_epochs] # Use the truncated epoch list
    else:
        mean_ntk_norms = None
        std_ntk_norms = None
        recorded_ntk_epochs = None


    # Process NTK eigenvalues across the ensemble for each recorded epoch
    # Calculate mean and std of the eigenvalue spectrum at each recorded epoch across networks
    mean_eigenvalue_spectra = []
    std_eigenvalue_spectra = []
    epochs_with_eigenvalue_data = [] # Track epochs where we have valid eigenvalue data

    if all_ensemble_ntk_eigenvalues:
        # all_ensemble_ntk_eigenvalues is list of lists of lists (network -> epoch -> eigenvalues)
        # We need to iterate through epochs and then networks
        # First, find the minimum number of recorded epochs for eigenvalues across all networks
        min_eigenvalue_epochs = min(len(eig_list_epochs) for eig_list_epochs in all_ensemble_ntk_eigenvalues)
        if not all(len(eig_list_epochs) == min_eigenvalue_epochs for eig_list_epochs in all_ensemble_ntk_eigenvalues):
             print(f"Warning: NTK eigenvalues lists for width {width} have inconsistent epoch counts across networks. Processing up to minimum epochs ({min_eigenvalue_epochs}).")

        # Iterate through each recorded epoch index up to the minimum
        for epoch_idx in range(min_eigenvalue_epochs):
            ensemble_eigenvalues_at_epoch = []
            valid_for_epoch = True
            # Collect eigenvalue lists for this epoch across all networks
            for network_idx in range(ENSEMBLE_SIZE):
                 if epoch_idx < len(all_ensemble_ntk_eigenvalues[network_idx]): # Check if this network has data for this epoch
                      network_eigenvalues = all_ensemble_ntk_eigenvalues[network_idx][epoch_idx]
                      if network_eigenvalues is not None and len(network_eigenvalues) > 0:
                           ensemble_eigenvalues_at_epoch.append(np.abs(network_eigenvalues)) # Use absolute values
                      else:
                           valid_for_epoch = False # Mark epoch as invalid if any network has no eigenvalues
                           break # No need to check other networks for this epoch

                 else:
                     valid_for_epoch = False # Mark epoch as invalid if network doesn't have data for this epoch
                     break # No need to check other networks

            if valid_for_epoch and ensemble_eigenvalues_at_epoch:
                 # Ensure all eigenvalue lists for this epoch have the same length across networks
                 if not all(len(eig_list) == len(ensemble_eigenvalues_at_epoch[0]) for eig_list in ensemble_eigenvalues_at_epoch):
                     print(f"Warning: NTK eigenvalue list lengths inconsistent within epoch {recorded_ntk_epochs[epoch_idx]} for width {width}. Skipping this epoch's eigenvalues.")
                     continue # Skip this epoch

                 # Stack the eigenvalue arrays for this epoch across the ensemble
                 stacked_eigenvalues = np.stack(ensemble_eigenvalues_at_epoch, axis=0) # Shape (ENSEMBLE_SIZE, num_eigenvalues)

                 # Calculate mean and std deviation of the eigenvalue spectrum across the ensemble
                 mean_spectrum = np.mean(stacked_eigenvalues, axis=0).tolist() # Mean over ensemble for each eigenvalue
                 std_spectrum = np.std(stacked_eigenvalues, axis=0).tolist() # Std dev over ensemble for each eigenvalue

                 mean_eigenvalue_spectra.append(mean_spectrum)
                 std_eigenvalue_spectra.append(std_spectrum)
                 epochs_with_eigenvalue_data.append(recorded_ntk_epochs[epoch_idx]) # Record the epoch number

            else:
                 print(f"Warning: Skipping epoch {recorded_ntk_epochs[epoch_idx]} for width {width} due to incomplete or missing eigenvalue data across ensemble.")


    data_dir = "loss_data"
    os.makedirs(data_dir, exist_ok=True)

    data_to_save = {
        'mean_1000_epoch_losses': mean_1000_epoch_losses,
        'std_1000_epoch_losses': std_1000_epoch_losses,
        'mean_1000_epoch_test_losses': mean_1000_epoch_test_losses,
        'std_1000_epoch_test_losses': std_1000_epoch_test_losses, # Corrected std_test_losses_1000_epochs
        'width': width,
        'epochs_recorded_at': [e for e in range(0, NUM_EPOCHS, 1000)], # Epochs for 1000-epoch losses
        'ntk_norms_epochs': mean_ntk_norms.tolist() if mean_ntk_norms is not None else None, # Save mean NTK norms over epochs
        'std_ntk_norms_epochs': std_ntk_norms.tolist() if std_ntk_norms is not None else None, # Save std NTK norms over epochs
        'ntk_record_epochs_norms': recorded_ntk_epochs, # Save the list of epochs for norms
        'mean_eigenvalue_spectra': mean_eigenvalue_spectra, # Save mean eigenvalue spectra over epochs
        'std_eigenvalue_spectra': std_eigenvalue_spectra, # Save std eigenvalue spectra over epochs
        'ntk_record_epochs_eigenvalues': epochs_with_eigenvalue_data # Save the list of epochs for eigenvalues
    }
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    with open(data_filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved loss data and processed time-dependent NTK properties for width {width} to {data_filename}")


    # Return mean/std of final losses and other recorded data
    return mean_interval_losses, std_interval_losses, mean_final_loss, std_final_loss, mean_1000_epoch_losses, std_1000_epoch_losses, mean_1000_epoch_test_losses, std_1000_epoch_test_losses


# List of widths to iterate over
widths = range(5, 50, 20)
results = []
threads = []

print("Starting threaded execution...")

for width in widths:
    thread = threading.Thread(target=train_model_thread, args=(width, results))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("Threaded execution finished.")

# Process results including time-dependent NTK data (if train_model_thread returns it)
# The train_model_thread function needs to be updated to return the new NTK data
# and the processing here needs to extract it.

# Let's update train_model_thread to capture the new return values from train_model
# The current train_model returns 8 items. The train_model_thread expects 8 + 1 (width) = 9 items.
# The train_model function now returns 8 items: mean_interval_losses, std_interval_losses, mean_final_loss, std_final_loss, mean_1000_epoch_losses, std_1000_epoch_losses, mean_1000_epoch_test_losses, std_1000_epoch_test_losses
# The train_model_thread adds the width. So the results tuple will have 9 items.
# This seems consistent with the existing valid_results processing logic.

valid_results = [r for r in results if r[0] is not None]

valid_results.sort(key=lambda x: x[8]) # Adjusted index for width

if valid_results:
    # Assuming mean_interval_losses is the first item (index 0) and has the same length for all valid results
    if valid_results[0] and valid_results[0][0] is not None:
        num_intervals_plotting = len(valid_results[0][0])
        print(f"Number of recorded intervals for plotting: {num_intervals_plotting}")

        interval_means_across_widths = [[] for _ in range(num_intervals_plotting)]
        interval_stds_across_widths = [[] for _ in range(num_intervals_plotting)]
        inverse_widths = []

        # Update the unpacking in the loop to match the return of train_model_thread
        for mean_interval, std_interval, mean_final_loss_val, std_final_loss_val, mean_1000_epoch, std_1000_epoch, mean_1000_epoch_test, std_1000_epoch_test, width in valid_results:
             inverse_widths.append(1 / width)
             for i in range(num_intervals_plotting):
                 interval_means_across_widths[i].append(mean_interval[i])
                 interval_stds_across_widths[i].append(std_interval[i])


        print("Finished collecting interval losses across widths for plotting.")

        final_plot_dir = "plots/final_plots"
        os.makedirs(final_plot_dir, exist_ok=True)

        interval_labels = ["0%", "25%", "50%", "75%", "100%"]

        for i in range(num_intervals_plotting):
            pl.figure()
            pl.plot(inverse_widths, interval_means_across_widths[i], label="Mean Loss", color="blue")
            interval_means_np = np.array(interval_means_across_widths[i])
            interval_stds_np = np.array(interval_stds_across_widths[i])
            pl.fill_between(inverse_widths, interval_means_np - interval_stds_np, interval_means_np + interval_stds_np, alpha=0.3, color="blue", label="±1 Std Dev")
            pl.title(f"Loss vs 1/Width at {interval_labels[i]} Training")
            pl.xlabel("1 / width")
            pl.ylabel("Loss")
            pl.grid(True)
            pl.legend()
            final_plot_path = os.path.join(final_plot_dir, f"loss_vs_inverse_width_interval_{i+1}.png")
            pl.savefig(final_plot_path)
            pl.close()
            print(f"Plotting finished for interval {i+1}.")

        print("Loss data saved to 'loss_data' directory.")

    else:
         print("Valid results list is not empty, but the first result does not contain interval loss data.")


else:
    print("No valid results to plot or save.")