# Re-define x_train and y_train as they were not in the current scope
import torch
from scipy.special import legendre

def true_function(x):
    P = legendre(4)
    x_np = x.numpy().squeeze()
    y_np = P(x_np)
    return torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

x_train = torch.linspace(-1, 1, 20).unsqueeze(1)
y_train = true_function(x_train)

# Now split the data
from sklearn.model_selection import train_test_split

x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print("x_train_split shape:", x_train_split.shape)
print("x_test_split shape:", x_test_split.shape)
print("y_train_split shape:", y_train_split.shape)
print("y_test_split shape:", y_test_split.shape)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import legendre
import time
import os
import multiprocessing as mp
# import tempfile # Import tempfile

print(mp.cpu_count())

# Set seed for reproducibility
torch.manual_seed(42)

# === Configurable parameters ===
ENSEMBLE_SIZE = 10       # Number of networks in ensemble


NUM_LAYERS = 1           # Number of hidden layers





def train_model(width):
    # === Check for GPU and set device ===
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(mp.cpu_count())

    # === Configurable parameters ===
    NUM_EPOCHS = 10000*width
    LEARNING_RATE = 0.5/width
    PATIENCE = 200000 # Number of epochs to wait for improvement before stopping
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Patience for early stopping: {PATIENCE}")


    NEURONS_PER_LAYER = width   # Neurons per hidden layer

    # Create a directory for the current width
    plot_dir = f"plots/width_{width}"
    os.makedirs(plot_dir, exist_ok=True)


    # === Define a customizable feedforward network ===
    class SimpleNet(nn.Module):
        def __init__(self, num_layers, neurons_per_layer):
            super(SimpleNet, self).__init__()
            layers = []

            input_dim = 1
            for _ in range(num_layers):
                layer = nn.Linear(input_dim, neurons_per_layer)
                # Scale the standard deviation of weights by 1/width, biases remain with std=1.0
                nn.init.normal_(layer.weight, mean=0.0, std=1.0/width)  # Gaussian weight init
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)    # Gaussian bias init
                layers.append(layer)
                layers.append(nn.Tanh())  # activation
                input_dim = neurons_per_layer

            # Output layer
            final_layer = nn.Linear(input_dim, 1)
            nn.init.normal_(final_layer.weight, mean=0.0, std=1.0/width)
            nn.init.normal_(final_layer.bias, mean=0.0, std=1.0)
            layers.append(final_layer)

            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)




    # === Train a single network ===

    trained_loss = []
    def train_network(model, x_train_split, y_train_split, x_test_split, y_test_split, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, patience=PATIENCE):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

        # Move model and criterion to the device
        model.to(device)
        criterion.to(device)

        # Move data to the device
        x_train_device = x_train_split.to(device)
        y_train_device = y_train_split.to(device)
        x_test_device = x_test_split.to(device)
        y_test_device = y_test_split.to(device)

        train_losses = [] # List to store training losses
        test_losses = [] # List to store test losses
        best_test_loss = float('inf')
        epochs_no_improve = 0

        model.train()  # Set model to training mode

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # Use training data on the device
            output = model(x_train_device)
            loss = criterion(output, y_train_device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Evaluate on test set
            model.eval()  # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculations
                test_output = model(x_test_device)
                test_loss = criterion(test_output, y_test_device)
                test_losses.append(test_loss.item())

            model.train() # Set model back to training mode

            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # Optional: print loss every 10 epochs
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")


        return model, train_losses, test_losses

    # === Train ensemble of networks ===
    ensemble_outputs = []
    ensemble_train_losses = [] # List to store training losses for each network in the ensemble
    ensemble_test_losses = [] # List to store test losses for each network in the ensemble


    for i in range(ENSEMBLE_SIZE):
        print(f"Training network {i+1}/{ENSEMBLE_SIZE}")
        start_time = time.time()
        model = SimpleNet(NUM_LAYERS, NEURONS_PER_LAYER)
        # Pass the split data to the train_network function and get losses
        trained_model, train_losses, test_losses = train_network(model, x_train_split, y_train_split, x_test_split, y_test_split)
        ensemble_train_losses.append(train_losses) # Store training losses for this network
        ensemble_test_losses.append(test_losses) # Store test losses for this network

        with torch.no_grad():
            # Use original x_train for plotting the full range
            output = trained_model(x_train.to(device)).cpu()
            ensemble_outputs.append(output.numpy())

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for network {i+1}: {elapsed_time:.2f} seconds")

# Plotting loss curve
    pl.figure()
    # Plot the training and test loss for one of the networks (e.g., the last one)
    if ensemble_train_losses and ensemble_test_losses:
        pl.plot(ensemble_train_losses[-1], label='Training Loss')
        pl.plot(ensemble_test_losses[-1], label='Test Loss')
    pl.title(f"Training and Test Loss (Width: {width})")
    pl.xlabel("Epoch")
    pl.ylabel("MSE")
    pl.grid(True)
    pl.legend()
    loss_plot_path = os.path.join(plot_dir, f"training_test_loss_width_{width}.png")
    pl.savefig(loss_plot_path)
    pl.close()


    ensemble_outputs = np.stack(ensemble_outputs, axis=0)  # shape: (ensemble, samples, 1)

    # === Compute statistics across ensemble ===
    mean_output = np.mean(ensemble_outputs, axis=0).squeeze()
    std_output = np.std(ensemble_outputs, axis=0).squeeze()

    # === Plot ===
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

    # Calculate mean and std of the final test losses across the ensemble
    final_test_losses = [losses[-1] for losses in ensemble_test_losses]
    loss = np.mean(final_test_losses)
    std_loss = np.std(final_test_losses)

    return loss, std_loss
# Process the results
loss_tot = []
loss_std = []
neurons = []


for i in range(5,50,20):
    mean, std = train_model(i)
    loss_tot.append(mean)
    loss_std.append(std)
    neurons.append(1/i)

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