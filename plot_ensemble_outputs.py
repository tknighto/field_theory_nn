import pickle
import os
import matplotlib.pyplot as pl
import numpy as np
import torch

# Redefine the true function and data (as they were in the original code)
def true_function(x):
    P = legendre(4)
    x_np = x.numpy().squeeze()
    y_np = P(x_np)
    return torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

x_train = torch.linspace(-1, 1, 24).unsqueeze(1)
y_train = true_function(x_train)

# Directory where the loss data is saved
data_dir = "loss_data"

# Iterate through each saved pickle file
for filename in os.listdir(data_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        width = data['width']
        all_ensemble_outputs = data['all_ensemble_outputs'] # This is a list of lists (ensemble -> data point)
        print(f"Loaded data for width: {width}")

        # Convert the list of outputs to a numpy array
        ensemble_outputs_np = np.array(all_ensemble_outputs) # Shape (ENSEMBLE_SIZE, num_data_points, 1)
        # Squeeze the last dimension if it's 1
        if ensemble_outputs_np.shape[-1] == 1:
             ensemble_outputs_np = ensemble_outputs_np.squeeze(-1) # Shape (ENSEMBLE_SIZE, num_data_points)


        pl.figure(figsize=(10, 6))

        # Plot the true function
        pl.plot(x_train.squeeze().numpy(), y_train.squeeze().numpy(), label='True Function', color='black', linewidth=2)

        # Plot each model's output from the ensemble
        for i in range(ensemble_outputs_np.shape[0]):
            pl.plot(x_train.squeeze().numpy(), ensemble_outputs_np[i, :], label=f'Model {i+1} Output', alpha=0.5)

        pl.title(f"True Function vs. Ensemble Model Outputs (Width {width})")
        pl.xlabel("x")
        pl.ylabel("y")
        # Add legend if the number of models is not too large
        if ensemble_outputs_np.shape[0] <= 10:
             pl.legend()
        pl.grid(True)

        # Save the plot
        plot_outputs_dir = "plots/ensemble_outputs"
        os.makedirs(plot_outputs_dir, exist_ok=True)
        plot_filename = os.path.join(plot_outputs_dir, f"ensemble_outputs_width_{width}.png")
        pl.savefig(plot_filename)
        pl.close()
        print(f"Saved ensemble outputs plot for width {width} to {plot_filename}")

print("\nFinished plotting ensemble outputs.")