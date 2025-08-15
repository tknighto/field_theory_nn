import pickle
import os
import matplotlib.pyplot as pl
import numpy as np
import torch
from scipy.special import legendre

# Redefine the true function and data (as they were in the original code)
def true_function(x):
    P = legendre(5)
    x_np = x.numpy().squeeze()
    y_np = P(x_np)
    return torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

x_train = torch.linspace(-1, 1, 24).unsqueeze(1)
y_train = true_function(x_train)
x_real = torch.linspace(-1, 1, 100).unsqueeze(1)
y_real = true_function(x_real)
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
        pl.plot(x_real.squeeze().numpy(), y_real.squeeze().numpy(), label='True Function', color='black', linewidth=2)

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




import pickle
import os
import matplotlib.pyplot as pl
import numpy as np
import torch # Import torch for true_function and data if needed

# Redefine the true function and data (assuming these are consistent with training)
def true_function(x):
    from scipy.special import legendre
    P = legendre(5)  # 5th degree Legendre polynomial
    x_np = x.numpy().squeeze()
    y_np = P(x_np)
    return torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

x_train = torch.linspace(-1, 1, 24).unsqueeze(1)
y_train = true_function(x_train)

# Assuming x_test_split and y_test_split are needed for plotting test points
# We need to load these if they are not globally available or regenerate them
# Based on the training code, they are generated using train_test_split
from sklearn.model_selection import train_test_split
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


data_dir = "loss_data"
all_widths_data = []

# Load data for all widths
for filename in os.listdir(data_dir):
    if filename.endswith(".pkl"):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                all_widths_data.append(data)
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")

# Sort data by width
all_widths_data.sort(key=lambda x: x.get('width', float('inf')))

# Plotting for each width
if not all_widths_data:
    print("No data files found or loaded from the loss_data directory.")
else:
    for data in all_widths_data:
        width = data.get('width')
        if width is None:
            print(f"Skipping data entry with no 'width' key: {data}")
            continue

        all_ensemble_outputs = data.get('all_ensemble_outputs') # Load the ensemble outputs

        if all_ensemble_outputs:
            print(f"Generating output plot for width: {width}")

            pl.figure(figsize=(10, 6))

            # Plot the true function
            x_true = torch.linspace(-1, 1, 100).unsqueeze(1) # More points for smooth true function
            y_true = true_function(x_true)
            pl.plot(x_true.squeeze().numpy(), y_true.squeeze().numpy(), label='True Function', color='black', linestyle='--')

            # Plot the mean ensemble output
            ensemble_outputs_np = np.array(all_ensemble_outputs) # Shape (ENSEMBLE_SIZE, num_train_points, 1)
            mean_ensemble_output = np.mean(ensemble_outputs_np, axis=0).squeeze() # Shape (num_train_points,)
            std_ensemble_output = np.std(ensemble_outputs_np, axis=0).squeeze() # Shape (num_train_points,)

            # The ensemble outputs are for the training data points (x_train)
            pl.plot(x_train.squeeze().numpy(), mean_ensemble_output, label='Mean Ensemble Output', color='blue')

            # Add shaded region for ±1 standard deviation of the ensemble output
            pl.fill_between(x_train.squeeze().numpy(), mean_ensemble_output - std_ensemble_output, mean_ensemble_output + std_ensemble_output, color='blue', alpha=0.2, label='±1 Std Dev (Ensemble)')


            # Mark the training points
            pl.plot(x_train_split.squeeze().numpy(), y_train_split.squeeze().numpy(), 'ro', markersize=5, label='Training Points')

            # Mark the test points
            pl.plot(x_test_split.squeeze().numpy(), y_test_split.squeeze().numpy(), 'go', markersize=5, label='Test Points')


            pl.title(f"Ensemble Output vs. True Function (Width {width})")
            pl.xlabel("x")
            pl.ylabel("y")
            pl.grid(True)
            pl.legend()

            plot_dir = "plots/ensemble_outputs"
            os.makedirs(plot_dir, exist_ok=True)
            plot_filename = os.path.join(plot_dir, f"ensemble_output_width_{width}.png")
            pl.savefig(plot_filename)
            pl.close()

            print(f"Plot saved to {plot_filename}")

        else:
            print(f"No ensemble output data found for width: {width}")