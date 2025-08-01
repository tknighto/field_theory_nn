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
        # train_model now returns 11 items: trained_model, interval_losses, final_loss, losses_1000_epochs, test_losses_1000_epochs, ntk_norms_epochs, ntk_eigenvalues_epochs, ntk_matrices_epochs, ntk_record_times, train_losses, training_times
        (mean_interval_losses, std_interval_losses, mean_final_loss, std_final_loss,
         mean_1000_epoch_losses, std_1000_epoch_losses, mean_1000_epoch_test_losses,
         std_1000_epoch_test_losses, all_ensemble_train_losses, all_ensemble_training_times) = train_model(width)

        print(f"Finished training for width: {width} in thread.")
        with results_lock:
            # Store all returned values along with the width (11 items + 1 width = 12 items)
            results_list.append((mean_interval_losses, std_interval_losses, mean_final_loss,
                                std_final_loss, mean_1000_epoch_losses, std_1000_epoch_losses,
                                mean_1000_epoch_test_losses, std_1000_epoch_test_losses,
                                all_ensemble_train_losses, all_ensemble_training_times, width)) # Adjusted order to match unpacking in main loop
    except Exception as e:
        print(f"Error training for width {width}: {e}")
        with results_lock:
            results_list.append((None, None, None, None, None, None, None, None, None, None, width)) # Ensure 11 None values + width


# Define the train_network function with 1000 epoch loss recording and time-dependent NTK computation
def train_network(model, x_train_split, y_train_split, x_test_split, y_test_split, learning_rate, num_epochs, device, weights):
    # Define the loss functions
    criterion = nn.MSELoss() # Use standard MSE loss

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

    # Move model and criterion to the device
    model.to(device)
    criterion.to(device)

    # Move data to the device
    x_train_device = x_train_split.to(device)
    y_train_device = y_train_split.to(device)
    x_test_device = x_test_split.to(device)
    y_test_device = y_test_split.to(device)


    train_losses = [] # List to store training losses for each epoch
    test_losses = [] # List to store test losses for each epoch
    interval_losses = [] # Losses at specific intervals (0%, 25%, etc.)
    losses_1000_epochs = [] # Losses at every 1000 epochs
    test_losses_1000_epochs = [] # Test losses at every 1000 epochs
    ntk_norms_epochs = [] # List to store NTK norms at recorded epochs
    ntk_eigenvalues_epochs = [] # List to store NTK eigenvalues (as lists) at recorded epochs
    ntk_matrices_epochs = [] # List to store NTK matrices at recorded epochs
    ntk_record_times = [] # List to store the training time when NTK is recorded
    training_times = [] # List to store training time for each epoch


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

    # Define training time intervals at which to compute NTK
    ntk_record_interval = 1.0 # Record NTK every 1 training time unit
    current_ntk_record_time = 0.0
    next_ntk_record_epoch = 0


    for epoch in range(num_epochs):

        optimizer.zero_grad()
        model.train()  # Set model to training mode

        # Forward pass
        output = model(x_train_device)

        # Apply standard MSE loss for training
        loss = criterion(output, y_train_device)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        current_training_time = (epoch + 1) * learning_rate
        training_times.append(current_training_time)


        if epoch in record_epochs:
            interval_losses.append(loss.item())

        if epoch % 1000 == 0:
            losses_1000_epochs.append(loss.item())

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Loss: {loss.item():.8f}")

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_output = model(x_test_device)
            test_loss = criterion(test_output, y_test_device)  # Use standard criterion for test loss
            test_losses.append(test_loss.item())

            if epoch % 1000 == 0:
                test_losses_1000_epochs.append(test_loss.item())

        model.train() # Set model back to training mode

        # Compute and save NTK properties at specified training times
        if current_training_time >= current_ntk_record_time:
             print(f"Computing NTK at training time {current_training_time:.4f} (Epoch {epoch})...")
             # Ensure model is on CPU for NTK computation if compute_ntk expects CPU tensors
             model.cpu()
             ntk_matrix = compute_ntk(model, x_train_split.cpu()) # Pass CPU data for NTK
             ntk_norm, ntk_eigenvalues = compute_ntk_properties(ntk_matrix)

             ntk_norms_epochs.append(ntk_norm)
             ntk_eigenvalues_epochs.append(ntk_eigenvalues.tolist()) # Convert eigenvalues to list for saving
             ntk_matrices_epochs.append(ntk_matrix.tolist()) # Save the full NTK matrix as a list of lists
             ntk_record_times.append(current_training_time) # Record the training time

             print(f"Finished computing and saving NTK at training time {current_training_time:.4f}.")
             model.to(device) # Move model back to original device

             # Update the next time to record NTK
             current_ntk_record_time += ntk_record_interval


        # Optional: print loss every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")


    # Return the final training loss and other recorded data, including training times
    return model, interval_losses, train_losses[-1], losses_1000_epochs, test_losses_1000_epochs, ntk_norms_epochs, ntk_eigenvalues_epochs, ntk_matrices_epochs, ntk_record_times, train_losses, training_times


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
    all_ensemble_ntk_matrices = [] # To store time-dependent NTK matrices for the ensemble
    ntk_record_times_list = [] # To store the list of training times where NTK was recorded
    all_ensemble_train_losses = [] # To store individual training loss trajectories
    all_ensemble_training_times = [] # To store individual training time trajectories


    for i in range(ENSEMBLE_SIZE):
        print(f"Training network {i+1}/{ENSEMBLE_SIZE}")
        start_time = time.time()
        model = SimpleNet(NUM_LAYERS, NEURONS_PER_LAYER)

        weights = torch.abs(y_train_split) + 1e-6
        weights = weights / weights.sum()

        test_weights = torch.abs(y_test_split) + 1e-6
        test_weights = test_weights / test_weights.sum()

        # train_network now returns final_loss and time-dependent NTK data including matrices and record times, and individual loss/time
        trained_model, interval_losses, final_loss, losses_1000_epochs, test_losses_1000_epochs, ntk_norms_epochs, ntk_eigenvalues_epochs, ntk_matrices_epochs, ntk_record_times, train_losses_individual, training_times_individual = train_network(model, x_train_split, y_train_split, x_test_split, y_test_split, LEARNING_RATE, NUM_EPOCHS, device, weights)

        ensemble_interval_losses.append(interval_losses)
        all_ensemble_final_losses.append(final_loss) # Append final loss for this network
        all_ensemble_1000_epoch_losses.append(losses_1000_epochs)
        all_ensemble_1000_epoch_test_losses.append(test_losses_1000_epochs)
        all_ensemble_ntk_norms.append(ntk_norms_epochs) # Append time-dependent NTK norms
        all_ensemble_ntk_eigenvalues.append(ntk_eigenvalues_epochs) # Append time-dependent NTK eigenvalues
        all_ensemble_ntk_matrices.append(ntk_matrices_epochs) # Append time-dependent NTK matrices
        ntk_record_times_list.append(ntk_record_times) # Store the recorded training times list
        all_ensemble_train_losses.append(train_losses_individual) # Append individual training losses
        all_ensemble_training_times.append(training_times_individual) # Append individual training times


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
             ntk_record_times_list = [times_list[:min_norm_epochs] for times_list in ntk_record_times_list]


        mean_ntk_norms = np.mean(np.array(all_ensemble_ntk_norms), axis=0)
        std_ntk_norms = np.std(np.array(all_ensemble_ntk_norms), axis=0)
        recorded_ntk_times = ntk_record_times_list[0][:min_norm_epochs] # Use the truncated time list
    else:
        mean_ntk_norms = None
        std_ntk_norms = None
        recorded_ntk_times = None


    # Process NTK eigenvalues across the ensemble for each recorded time
    # Calculate mean and std of the eigenvalue spectrum at each recorded epoch across networks
    mean_eigenvalue_spectra = []
    std_eigenvalue_spectra = []
    times_with_eigenvalue_data = [] # Track times where we have valid eigenvalue data

    if all_ensemble_ntk_eigenvalues:
        # all_ensemble_ntk_eigenvalues is list of lists of lists (network -> time -> eigenvalues)
        # We need to iterate through times and then networks
        # First, find the minimum number of recorded times for eigenvalues across all networks
        min_eigenvalue_times = min(len(eig_list_times) for eig_list_times in all_ensemble_ntk_eigenvalues)
        if not all(len(eig_list_times) == min_eigenvalue_times for eig_list_times in all_ensemble_ntk_eigenvalues):
             print(f"Warning: NTK eigenvalues lists for width {width} have inconsistent time counts across networks. Processing up to minimum times ({min_eigenvalue_times}).")

        # Iterate through each recorded time index up to the minimum
        for time_idx in range(min_eigenvalue_times):
            ensemble_eigenvalues_at_time = []
            valid_for_time = True
            # Collect eigenvalue lists for this time across all networks
            for network_idx in range(ENSEMBLE_SIZE):
                 if time_idx < len(all_ensemble_ntk_eigenvalues[network_idx]): # Check if this network has data for this time
                      network_eigenvalues = all_ensemble_ntk_eigenvalues[network_idx][time_idx]
                      if network_eigenvalues is not None and len(network_eigenvalues) > 0:
                           ensemble_eigenvalues_at_time.append(np.abs(network_eigenvalues)) # Use absolute values
                      else:
                           valid_for_time = False # Mark time as invalid if any network has no eigenvalues
                           break # No need to check other networks for this time

                 else:
                     valid_for_time = False # Mark time as invalid if network doesn't have data for this time
                     break # No need to check other networks

            if valid_for_time and ensemble_eigenvalues_at_time:
                 # Ensure all eigenvalue lists for this time have the same length across networks
                 if not all(len(eig_list) == len(ensemble_eigenvalues_at_time[0]) for eig_list in ensemble_eigenvalues_at_time):
                     print(f"Warning: NTK eigenvalue list lengths inconsistent within time {recorded_ntk_times[time_idx]:.4f} for width {width}. Skipping this time's eigenvalues.")
                     continue # Skip this time

                 # Stack the eigenvalue arrays for this time across the ensemble
                 stacked_eigenvalues = np.stack(ensemble_eigenvalues_at_time, axis=0) # Shape (ENSEMBLE_SIZE, num_eigenvalues)

                 # Calculate mean and std deviation of the eigenvalue spectrum across the ensemble
                 mean_spectrum = np.mean(stacked_eigenvalues, axis=0).tolist() # Mean over ensemble for each eigenvalue
                 std_spectrum = np.std(stacked_eigenvalues, axis=0).tolist() # Std dev over ensemble for each eigenvalue

                 mean_eigenvalue_spectra.append(mean_spectrum)
                 std_eigenvalue_spectra.append(std_spectrum)
                 times_with_eigenvalue_data.append(recorded_ntk_times[time_idx]) # Record the time


            else:
                 print(f"Warning: Skipping time {recorded_ntk_times[time_idx]:.4f} for width {width} due to incomplete or missing eigenvalue data across ensemble.")

    # Process NTK matrices across the ensemble for each recorded time
    mean_ntk_matrices = []
    std_ntk_matrices = []
    times_with_matrix_data = [] # Track times where we have valid matrix data

    if all_ensemble_ntk_matrices:
        # all_ensemble_ntk_matrices is list of lists of lists of lists (network -> time -> matrix row -> matrix column)
        # We need to iterate through times and then networks
        min_matrix_times = min(len(matrix_list_times) for matrix_list_times in all_ensemble_ntk_matrices)
        if not all(len(matrix_list_times) == min_matrix_times for matrix_list_times in all_ensemble_ntk_matrices):
             print(f"Warning: NTK matrices lists for width {width} have inconsistent time counts across networks. Processing up to minimum times ({min_matrix_times}).")

        for time_idx in range(min_matrix_times):
            ensemble_matrices_at_time = []
            valid_for_time = True
            for network_idx in range(ENSEMBLE_SIZE):
                if time_idx < len(all_ensemble_ntk_matrices[network_idx]):
                    network_matrix = all_ensemble_ntk_matrices[network_idx][time_idx]
                    if network_matrix is not None and len(network_matrix) > 0:
                        ensemble_matrices_at_time.append(np.array(network_matrix)) # Convert list of lists to numpy array
                    else:
                        valid_for_time = False
                        break
                else:
                    valid_for_time = False
                    break

            if valid_for_time and ensemble_matrices_at_time:
                 # Ensure all matrices for this time have the same shape across networks
                 if not all(matrix.shape == ensemble_matrices_at_time[0].shape for matrix in ensemble_matrices_at_time):
                     print(f"Warning: NTK matrix shapes inconsistent within time {recorded_ntk_times[time_idx]:.4f} for width {width}. Skipping this time's matrices.")
                     continue

                 stacked_matrices = np.stack(ensemble_matrices_at_time, axis=0) # Shape (ENSEMBLE_SIZE, matrix_dim, matrix_dim)

                 mean_matrix = np.mean(stacked_matrices, axis=0).tolist() # Mean over ensemble
                 std_matrix = np.std(stacked_matrices, axis=0).tolist() # Std dev over ensemble

                 mean_ntk_matrices.append(mean_matrix)
                 std_ntk_matrices.append(std_matrix)
                 times_with_matrix_data.append(recorded_ntk_times[time_idx])

            else:
                 print(f"Warning: Skipping time {recorded_ntk_times[time_idx]:.4f} for width {width} due to incomplete or missing matrix data across ensemble.")


    data_dir = "loss_data"
    os.makedirs(data_dir, exist_ok=True)

    data_to_save = {
        'mean_1000_epoch_losses': mean_1000_epoch_losses,
        'std_1000_epoch_losses': std_1000_epoch_losses,
        'mean_1000_epoch_test_losses': mean_1000_epoch_test_losses,
        'std_1000_epoch_test_losses': std_1000_epoch_test_losses, # Corrected std_test_losses_1000_epochs
        'width': width,
        'epochs_recorded_at': [e for e in range(0, NUM_EPOCHS, 1000)], # Epochs for 1000-epoch losses
        'ntk_norms_times': mean_ntk_norms.tolist() if mean_ntk_norms is not None else None, # Save mean NTK norms over times
        'std_ntk_norms_times': std_ntk_norms.tolist() if std_ntk_norms is not None else None, # Save std NTK norms over times
        'ntk_record_times_norms': recorded_ntk_times, # Save the list of times for norms
        'mean_eigenvalue_spectra_times': mean_eigenvalue_spectra, # Save mean eigenvalue spectra over times
        'std_eigenvalue_spectra_times': std_eigenvalue_spectra, # Save std eigenvalue spectra over times
        'ntk_record_times_eigenvalues': times_with_eigenvalue_data, # Save the list of times for eigenvalues
        'mean_ntk_matrices_times': mean_ntk_matrices, # Save mean NTK matrices over times
        'std_ntk_matrices_times': std_ntk_matrices, # Save std NTK matrices over times
        'ntk_record_times_matrices': times_with_matrix_data, # Save the list of times for matrices
        'all_ensemble_train_losses': all_ensemble_train_losses, # Save individual training losses
        'all_ensemble_training_times': all_ensemble_training_times # Save individual training times
    }
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    with open(data_filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved loss data and processed time-dependent NTK properties for width {width} to {data_filename}")


    # Return mean/std of final losses and other recorded data
    return mean_interval_losses, std_interval_losses, mean_final_loss, std_final_loss, mean_1000_epoch_losses, std_1000_epoch_losses, mean_1000_epoch_test_losses, std_1000_epoch_test_losses, all_ensemble_train_losses, all_ensemble_training_times


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
# The train_model function now returns 11 items: mean_interval_losses, std_interval_losses, mean_final_loss, std_final_loss, mean_1000_epoch_losses, std_1000_epoch_losses, mean_1000_epoch_test_losses, std_1000_epoch_test_losses, all_ensemble_train_losses, all_ensemble_training_times
# The train_model_thread adds the width. So the results tuple will have 11 items.
# This seems consistent with the existing valid_results processing logic.

valid_results = [r for r in results if r[0] is not None]

# Sort by width (index 10)
valid_results.sort(key=lambda x: x[10])

if valid_results:
    # Assuming mean_interval_losses is the first item (index 0) and has the same length for all valid results
    if valid_results[0] and valid_results[0][0] is not None:
        num_intervals_plotting = len(valid_results[0][0])
        print(f"Number of recorded intervals for plotting: {num_intervals_plotting}")

        interval_means_across_widths = [[] for _ in range(num_intervals_plotting)]
        interval_stds_across_widths = [[] for _ in range(num_intervals_plotting)]
        inverse_widths = []

        # Update the unpacking in the loop to match the return of train_model_thread (now 11 items)
        for mean_interval, std_interval, mean_final_loss_val, std_final_loss_val, mean_1000_epoch, std_1000_epoch, mean_1000_epoch_test, std_1000_epoch_test, all_ensemble_train_losses_individual, all_ensemble_training_times_individual, width in valid_results:
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

        # Plotting individual loss trajectories vs training time
        plot_loss_trajectories_dir = "plots/loss_trajectories"
        os.makedirs(plot_loss_trajectories_dir, exist_ok=True)

        # Since results are sorted by width, iterate through them
        for mean_interval, std_interval, mean_final_loss_val, std_final_loss_val, mean_1000_epoch, std_1000_epoch, mean_1000_epoch_test, std_1000_epoch_test, all_ensemble_train_losses_individual, all_ensemble_training_times_individual, width in valid_results:
             pl.figure(figsize=(10, 6))
             for i in range(ENSEMBLE_SIZE):
                 pl.plot(all_ensemble_training_times_individual[i], all_ensemble_train_losses_individual[i], label=f'Model {i+1}')

             pl.title(f"Training Loss vs. Training Time (Width {width})")
             pl.xlabel("Training Time (epochs * learning rate)")
             pl.ylabel("Training Loss")
             pl.yscale('log')
             # pl.legend() # Comment out legend if too many models
             pl.grid(True)
             plot_filename = os.path.join(plot_loss_trajectories_dir, f"loss_trajectories_width_{width}.png")
             pl.savefig(plot_filename)
             pl.close()
             print(f"Loss trajectories plot saved for width {width} to {plot_filename}")

        print("\nFinished plotting loss trajectories.")

    else:
         print("Valid results list is not empty, but the first result does not contain interval loss data.")


else:
    print("No valid results to plot or save.")