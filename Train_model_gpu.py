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

print("Number of GPUs available:", torch.cuda.device_count())

# Set seed for reproducibility
torch.manual_seed(42)

# === Configurable parameters ===
ENSEMBLE_SIZE = 10       # Number of networks in ensemble
NUM_EPOCHS = 500000
LEARNING_RATE = 0.03
NUM_LAYERS = 1           # Number of hidden layers

# === Define a simple dataset ===
def true_function(x):
    P = legendre(4)  # 4th-degree Legendre polynomial, change as needed
    x_np = x.numpy().squeeze()  # Convert to NumPy for scipy
    y_np = P(x_np)              # Evaluate polynomial
    return torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)


x_train = torch.linspace(-1, 1, 20).unsqueeze(1)  # shape: (100, 1)
y_train = true_function(x_train)




def train_model(width):
    # === Check for GPU and set device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Number of GPUs available:", torch.cuda.device_count())

    # === Configurable parameters ===

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
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)  # Gaussian weight init
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)    # Gaussian bias init
                layers.append(layer)
                layers.append(nn.Tanh())  # activation
                input_dim = neurons_per_layer

            # Output layer
            final_layer = nn.Linear(input_dim, 1)
            nn.init.normal_(final_layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(final_layer.bias, mean=0.0, std=1.0)
            layers.append(final_layer)

            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)




    # === Train a single network ===

    trained_loss = []
    losses = []
    losses1 = []
    def train_network(model, x_train, y_train, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

        # Move model and criterion to the device
        model.to(device)
        criterion.to(device)

        # Move data to the device
        x_train_device = x_train.to(device)
        y_train_device = y_train.to(device)


        model.train()  # Set model to training mode

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # Use data on the device
            output = model(x_train_device)
            loss = criterion(output, y_train_device)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())


            # Optional: print loss every 10 epochs
            if (epoch + 1) % 10000 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        losses1.append(loss.item())




        return model

    # === Train ensemble of networks ===
    ensemble_outputs = []

    for i in range(ENSEMBLE_SIZE):
        print(f"Training network {i+1}/{ENSEMBLE_SIZE}")
        start_time = time.time()
        model = SimpleNet(NUM_LAYERS, NEURONS_PER_LAYER)
        trained_model = train_network(model, x_train, y_train)
        with torch.no_grad():
            # Move output back to CPU for numpy conversion if needed for plotting later
            output = trained_model(x_train.to(device)).cpu()
            ensemble_outputs.append(output.numpy())

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training time for network {i+1}: {elapsed_time:.2f} seconds")

# Plotting loss curve
    pl.figure()
    pl.plot(losses)
    pl.title(f"Training Loss (Width: {width})")
    pl.xlabel("Epoch")
    pl.ylabel("MSE")
    pl.grid(True)
    loss_plot_path = os.path.join(plot_dir, f"training_loss_width_{width}.png")
    pl.savefig(loss_plot_path)
    pl.close()
    # # Upload loss plot to Google Drive
    # upload_file_to_drive(loss_plot_path, folder_id='your_google_drive_folder_id') # Add your folder_id here if needed: folder_id='your_folder_id'


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
    # # Upload ensemble plot to Google Drive
    # upload_file_to_drive(ensemble_plot_path, folder_id='your_google_drive_folder_id') # Add your folder_id here if needed: folder_id='your_folder_id'

    # Plot losses after training
    loss = np.mean(losses1)

    std_loss = np.std(losses1)

    # # Clean up temporary directory
    # os.remove(loss_plot_path)
    # os.remove(ensemble_plot_path)
    # os.rmdir(plot_dir)
    # os.rmdir(temp_dir)


    return loss, std_loss