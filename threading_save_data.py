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

x_train = torch.linspace(-1, 1, 20).unsqueeze(1)
y_train = true_function(x_train)


# Define a lock for thread-safe appending to lists (since lists are shared)
results_lock = threading.Lock()

# Define the function to be threaded (same as before)
def train_model_thread(width, results_list):
    print(f"Starting training for width: {width} in thread.")
    try:
        # train_model now returns mean_interval_losses, std_interval_losses, final_loss, final_std_loss, mean_1000_epoch_losses, std_1000_epoch_losses
        mean_interval, std_interval, final_mean, final_std, mean_1000_epoch, std_1000_epoch = train_model(width)
        print(f"Finished training for width: {width} in thread.")
        with results_lock:
            # Store all returned values along with the width
            results_list.append((mean_interval, std_interval, final_mean, final_std, mean_1000_epoch, std_1000_epoch, width))
    except Exception as e:
        print(f"Error training for width {width}: {e}")
        with results_lock:
            results_list.append((None, None, None, None, None, None, width))


# Define the train_network function with 1000 epoch loss recording
# Pass 'device' as an argument
def train_network(model, x_train, y_train, learning_rate, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

    model.to(device)
    criterion.to(device)

    x_train_device = x_train.to(device)
    y_train_device = y_train.to(device)

    model.train()

    interval_losses = []
    epoch_losses = []
    losses_1000_epochs = []

    total_epochs = num_epochs

    record_epochs = [
        0,
        total_epochs // 4,
        total_epochs // 2,
        total_epochs * 3 // 4,
        total_epochs - 1
    ]
    record_epochs = sorted(list(set([max(0, min(total_epochs - 1, ep)) for ep in record_epochs])))


    for epoch in range(total_epochs):
        optimizer.zero_grad()
        output = model(x_train_device)
        loss = criterion(output, y_train_device)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

        if epoch in record_epochs:
            interval_losses.append(loss.item())

        if epoch % 1000 == 0:
            losses_1000_epochs.append(loss.item())

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Loss: {loss.item():.4f}")

    return model, interval_losses, epoch_losses, losses_1000_epochs


# Update train_model to collect and save 1000-epoch losses
def train_model(width):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(mp.cpu_count())

    NUM_EPOCHS = 10000* width
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

    ensemble_outputs = []
    ensemble_interval_losses = []
    ensemble_epoch_losses = []
    all_ensemble_1000_epoch_losses = []


    for i in range(ENSEMBLE_SIZE):
        print(f"Training network {i+1}/{ENSEMBLE_SIZE}")
        start_time = time.time()
        model = SimpleNet(NUM_LAYERS, NEURONS_PER_LAYER)
        # Pass 'device' to train_network
        trained_model, interval_losses, epoch_losses, losses_1000_epochs = train_network(model, x_train, y_train, LEARNING_RATE, NUM_EPOCHS, device)

        ensemble_interval_losses.append(interval_losses)
        ensemble_epoch_losses.append(epoch_losses)
        all_ensemble_1000_epoch_losses.append(losses_1000_epochs)


        with torch.no_grad():
            output = trained_model(x_train.to(device)).cpu()
            ensemble_outputs.append(output.numpy())

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for network {i+1}: {elapsed_time:.2f} seconds")

    if ensemble_epoch_losses:
        pl.figure()
        pl.plot(ensemble_epoch_losses[-1])
        pl.title(f"Training Loss (Width: {width})")
        pl.xlabel("Epoch")
        pl.ylabel("MSE")
        pl.grid(True)
        loss_plot_path = os.path.join(plot_dir, f"training_loss_width_{width}.png")
        pl.savefig(loss_plot_path)
        pl.close()

    ensemble_outputs = np.stack(ensemble_outputs, axis=0)

    mean_output = np.mean(ensemble_outputs, axis=0).squeeze()
    std_output = np.std(ensemble_outputs, axis=0).squeeze()

    ensemble_interval_losses_np = np.array(ensemble_interval_losses)
    mean_interval_losses = np.mean(ensemble_interval_losses_np, axis=0)
    std_interval_losses = np.std(ensemble_interval_losses_np, axis=0)

    all_ensemble_1000_epoch_losses_np = np.array(all_ensemble_1000_epoch_losses)
    mean_1000_epoch_losses = np.mean(all_ensemble_1000_epoch_losses_np, axis=0)
    std_1000_epoch_losses = np.std(all_ensemble_1000_epoch_losses_np, axis=0)

    # Convert x_train and y_train to numpy for plotting
    x_np = x_train.numpy().squeeze()
    y_np = y_train.numpy().squeeze()

    pl.figure(figsize=(10, 6))
    pl.plot(x_np, y_np, label="True Function", color="black")
    pl.plot(x_np, mean_output, label="Ensemble Mean", color="blue")
    pl.fill_between(x_np, mean_output - std_output, mean_output + std_output, color='blue', alpha=0.3, label="Std Dev")
    for i in range(ENSEMBLE_SIZE):
        pl.plot(x_np, ensemble_outputs[i].squeeze(), alpha=0.2, color='gray')
    pl.legend()
    pl.title(f"Ensemble of Neural Networks Learning P4 (Width: {width})")
    pl.xlabel("x")
    pl.ylabel("f(x)")
    pl.grid(True)
    ensemble_plot_path = os.path.join(plot_dir, f"ensemble_plot_width_{width}.png")
    pl.savefig(ensemble_plot_path)
    pl.close()

    final_losses = [loss[-1] for loss in ensemble_epoch_losses]
    loss = np.mean(final_losses)
    std_loss = np.std(final_losses)

    data_dir = "loss_data"
    os.makedirs(data_dir, exist_ok=True)

    data_to_save = {
        'mean_1000_epoch_losses': mean_1000_epoch_losses,
        'std_1000_epoch_losses': std_1000_epoch_losses,
        'width': width,
        'epochs_recorded_at': [e for e in range(0, NUM_EPOCHS, 1000)]
    }
    data_filename = os.path.join(data_dir, f"loss_data_width_{width}.pkl")
    with open(data_filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved loss data for width {width} to {data_filename}")


    return mean_interval_losses, std_interval_losses, loss, std_loss, mean_1000_epoch_losses, std_1000_epoch_losses


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

valid_results = [r for r in results if r[0] is not None]

valid_results.sort(key=lambda x: x[6])

if valid_results:
    num_intervals_plotting = len(valid_results[0][0])
    print(f"Number of recorded intervals for plotting: {num_intervals_plotting}")

    interval_means_across_widths = [[] for _ in range(num_intervals_plotting)]
    interval_stds_across_widths = [[] for _ in range(num_intervals_plotting)]
    inverse_widths = []

    for mean_interval, std_interval, final_mean, final_std, mean_1000_epoch, std_1000_epoch, width in valid_results:
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
    print("No valid results to plot or save.")