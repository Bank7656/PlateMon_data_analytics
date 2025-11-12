import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Configuration ---
# Data and Feature Engineering
WINDOW_SIZE = 30
ROLLING_WINDOW_FEATURES = WINDOW_SIZE
TS_COLS = [str(i) for i in range(1, 501)]  # Explicitly define time series columns

# VAE Model Architecture
INPUT_DIM = WINDOW_SIZE              # NEW: We will ONLY feed the raw signal
VAE_HIDDEN_DIM = 512                 # NEW: Increased capacity (see section 3)
VAE_PRE_LATENT_DIM = 256             # NEW: Increased capacity
VAE_LATENT_DIM = 64                  # NEW: Increased capacity

# Training Hyperparameters
N_EPOCHS = 100                       # NEW: Train for longer
BATCH_SIZE = 32
LEARNING_RATE = 1e-4                 # NEW: Slightly higher LR
# OLD: BETA = 0.5
BETA = 0.001                           # NEW: Lower beta to focus more on reconstruction
                                     # (You can even try 0.01)

# Anomaly Detection
THRESHOLD_STD_MULTIPLIER = 3  # Refined: Use Mean + 3*STD for thresholding
CONSECUTIVE_WINDOW_THRESHOLD = 3 # Refined: Require 3 consecutive windows to trigger an anomaly
NORMAL_RUNS_TO_PLOT = 3  # New: Number of normal runs to plot for sanity check

# System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- 2. Data Preparation, Plotting & Anomaly Filtering Functions ---
def reconstruct_full_errors_from_windows(point_error_windows, original_length,
                                         window_size):
    """
    Reconstructs the full point-wise error signal from overlapping
    windows of point-wise errors.
    """
    reconstructed_error = np.zeros(original_length)
    window_counts = np.zeros(original_length)

    # --- THIS IS THE FIX ---
    # Force the incoming array to be float64, solving the dtype('O') error
    point_error_windows_float = point_error_windows.astype(np.float64)
    # --- END OF FIX ---

    # Now iterate over the new float array
    for i, window in enumerate(point_error_windows_float):
        start_index = i
        end_index = start_index + window_size
        if start_index >= original_length:
            break

        current_window_length = min(window_size,
                                  original_length - start_index)
        
        # This operation will now work (float + float)
        reconstructed_error[start_index:end_index] += window[
                                                     :current_window_length]
        window_counts[start_index:end_index] += 1

    # Average the errors for overlapping regions
    window_counts[window_counts == 0] = 1
    return reconstructed_error / window_counts

def add_rolling_features(df, ts_cols, window):
    df_copy = df.copy()
    ts_data = df_copy[ts_cols].values

    rolling_mean_data = np.zeros_like(ts_data)
    rolling_std_data = np.zeros_like(ts_data)

    for i in range(ts_data.shape[0]):
        series = pd.Series(ts_data[i, :])
        rolling_mean_data[i, :] = series.rolling(
            window=window, min_periods=1).mean().values
        rolling_std_data[i, :] = series.rolling(
            window=window, min_periods=1).std().fillna(0).values

    mean_cols = [f"mean_{c}" for c in ts_cols]
    std_cols = [f"std_{c}" for c in ts_cols]

    mean_df = pd.DataFrame(
        rolling_mean_data, columns=mean_cols, index=df_copy.index)
    std_df = pd.DataFrame(
        rolling_std_data, columns=std_cols, index=df_copy.index)

    featured_df = pd.concat([df_copy, mean_df, std_df], axis=1)
    return featured_df, mean_cols, std_cols


def create_windows_from_featured_df(df, ts_cols, mean_cols, std_cols,
                                    window_size):
    windows = []
    for i in range(len(df)):
        raw_series = df.iloc[i][ts_cols].values
        # We still have mean_series and std_series available if we wanted them
        # but we will NOT pass them to the VAE.
        # mean_series = df.iloc[i][mean_cols].values
        # std_series = df.iloc[i][std_cols].values

        for j in range(len(raw_series) - window_size + 1):
            
            # --- THIS IS THE KEY CHANGE ---
            # OLD:
            # combined_window = np.concatenate([
            #     raw_series[j:j + window_size],
            #     mean_series[j:j + window_size],
            #     std_series[j:j + window_size]
            # ])
            # windows.append(combined_window)
            
            # NEW:
            raw_window = raw_series[j:j + window_size]
            windows.append(raw_window)
            # --- END OF CHANGE ---

    return np.array(windows)


def reconstruct_full_signal_from_windows(windows, original_length,
                                         window_size):
    raw_signal_windows = windows[:, :window_size]
    reconstructed_signal = np.zeros(original_length)
    window_counts = np.zeros(original_length)

    for i, window in enumerate(raw_signal_windows):
        start_index = i
        end_index = start_index + window_size
        if start_index >= original_length:
            break

        current_window_length = min(window_size,
                                    original_length - start_index)
        reconstructed_signal[start_index:end_index] += window[
                                                       :current_window_length]
        window_counts[start_index:end_index] += 1

    window_counts[window_counts == 0] = 1
    return reconstructed_signal / window_counts


def filter_consecutive_anomalies(is_anomalous, min_consecutive):
    """Filters a boolean array to only keep sequences of True values
    that are at least `min_consecutive` long."""
    filtered = np.zeros_like(is_anomalous)
    start_idx = -1
    for i, is_a in enumerate(is_anomalous):
        if is_a and start_idx == -1:
            start_idx = i
        elif not is_a and start_idx != -1:
            if (i - start_idx) >= min_consecutive:
                filtered[start_idx:i] = True
            start_idx = -1
    # Check for a sequence at the very end
    if start_idx != -1 and (len(is_anomalous) - start_idx) >= min_consecutive:
        filtered[start_idx:] = True
    return filtered


import matplotlib.pyplot as plt
import numpy as np
# --- ADD THIS IMPORT AT THE TOP OF YOUR SCRIPT ---
from matplotlib.ticker import MaxNLocator

# ... (your other functions) ...

def plot_reconstruction_and_errors(original_signal, run_id,
                                   reconstructed_windows, window_errors,
                                   threshold, anomalous_points_mask,
                                   ground_truth_mask=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    reconstructed_signal = reconstruct_full_signal_from_windows(
        reconstructed_windows, len(original_signal), WINDOW_SIZE)

    # --- Plot 1: Original vs. Reconstructed Signal (No changes) ---
    ax1.plot(original_signal, label='Original Signal', color='blue', alpha=0.8)
    ax1.plot(reconstructed_signal, label='Reconstructed Signal', color='red',
             linestyle='--')
    highlight_signal = np.where(anomalous_points_mask, original_signal,
                                np.nan)
    ax1.plot(highlight_signal, color='orange', linewidth=4,
             label='Detected Anomalous Region')

    if ground_truth_mask is not None:
        highlight_ground_truth = np.where(ground_truth_mask, original_signal,
                                          np.nan)
        ax1.plot(highlight_ground_truth, color='green', linewidth=2,
                 linestyle=':', label='Ground Truth Anomaly')

    ax1.set_title(f'Run ID: {run_id} - Original vs. Reconstruction',
                  fontsize=16)
    ax1.set_ylabel('Conductivity')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Plot 2: Reconstruction Error (MODIFIED) ---
    ax2.plot(window_errors, label='Reconstruction Error per Window',
             color='purple')
    ax2.axhline(y=threshold, color='black', linestyle='-.',
                label=f'Anomaly Threshold ({threshold:.4f})')
    ax2.set_xlabel('Window Start Index')
    ax2.set_ylabel('Mean Squared Error')
    
    # --- HERE ARE THE FIXES ---
    # 1. COMMENT OUT the log scale
    # ax2.set_yscale('log') 
    
    # 2. SET THE BOTTOM LIMIT TO 0
    # (Your code already had this, but now it will work)
    ax2.set_ylim(bottom=0) 
    
    # 3. ADD MAJOR TICKS
    # This will create ~5 major, evenly-spaced linear ticks (e.g., 0, 0.02, 0.04, 0.06...)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower')) 
    # --- END OF FIXES ---

    ax2.legend(loc='upper left')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.suptitle(f'Anomaly Detection Analysis for Run: {run_id}', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- ADD THIS NEW FUNCTION (COPIED FROM plot_anomalies_example.py) ---
def plot_anomalies(original_signal, is_anomalous_mask, run_id, 
                   y_label="Conductivity", 
                   x_label="Time Step (2s interval)"):
    """
    Creates a plot to visualize time series anomalies, similar to the provided image.

    Args:
        original_signal (array-like): 
            The 1D array or list containing the full time series data (e.g., conductivity values).
        
        is_anomalous_mask (array-like): 
            A 1D boolean array or list of the *same length* as original_signal.
            Should be `True` at indices where an anomaly is detected and `False` otherwise.
        
        run_id (str): 
            The ID of the run, used for the plot title (e.g., "JUL_23_5").
        
        y_label (str, optional): 
            The label for the y-axis. Defaults to "Conductivity".
        
        x_label (str, optional): 
            The label for the x-axis. Defaults to "Time Step (2s interval)".
    """
    
    # Ensure data is in NumPy array format for easier manipulation
    original_signal = np.asarray(original_signal)
    is_anomalous_mask = np.asarray(is_anomalous_mask)
    
    if len(original_signal) != len(is_anomalous_mask):
        raise ValueError("original_signal and is_anomalous_mask must have the same length.")
        
    # Create the time-step axis
    time_steps = np.arange(len(original_signal))
    
    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 1. Plot the original signal as a blue line
    #    zorder=1 places it "behind" the anomaly points.
    ax.plot(time_steps, original_signal, color='blue', label='Normal Data', zorder=1)
    
    # 2. Create the anomaly data
    #    We create an array full of np.nan (which don't get plotted)
    #    and then fill in the *actual signal values* only where the mask is True.
    anomalous_points = np.full_like(original_signal, np.nan, dtype=float)
    anomalous_points[is_anomalous_mask] = original_signal[is_anomalous_mask]
    
    # 3. Plot the anomalies as red circles
    #    This will only plot the non-nan values.
    #    zorder=2 places them "on top" of the line.
    ax.scatter(time_steps, anomalous_points, color='red', label='Anomaly', zorder=2)
    
    # --- Set labels and title ---
    ax.set_title(f"Anomalies Detected in Run: {run_id}", fontsize=16)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    
    # --- Add legend and grid ---
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- Display the plot ---
    plt.tight_layout()
    # plt.show() # We will call plt.show() from the main script loop

# --- NEW FUNCTION FOR STITCHED PLOT ---
def plot_anomalies_stitched(original_signal, 
                            predicted_mask, 
                            ground_truth_mask,
                            title,
                            x_label="Time Step"):
    """
    Creates a single plot for a long, stitched time series, showing the signal,
    predicted anomalies (Red X), and ground truth anomalies (Green O).
    """
    
    # Use the style you liked from the example
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Ensure data is in NumPy array format
    original_signal = np.asarray(original_signal)
    predicted_mask = np.asarray(predicted_mask)
    ground_truth_mask = np.asarray(ground_truth_mask)
        
    # Create the time-step axis
    time_steps = np.arange(len(original_signal))
    
    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(25, 8)) # Make it nice and wide
    
    # 1. Plot the original signal as a blue line
    ax.plot(time_steps, original_signal, color='blue', label='Signal', zorder=1, alpha=0.7)
    
    # 2. Create and plot the PREDICTED anomaly data (Red X)
    predicted_points = np.full_like(original_signal, np.nan, dtype=float)
    predicted_points[predicted_mask] = original_signal[predicted_mask]
    
    ax.scatter(time_steps, predicted_points, 
               color='red', 
               label='Predicted Anomaly (VAE)', 
               marker='x', 
               s=20,      
               zorder=3)   
               
    # 3. Create and plot the GROUND TRUTH anomaly data (Green O)
    ground_truth_points = np.full_like(original_signal, np.nan, dtype=float)
    ground_truth_points[ground_truth_mask] = original_signal[ground_truth_mask]
            
    ax.scatter(time_steps, ground_truth_points, 
                color='green', 
                label='Ground Truth Anomaly',
                marker='o',       
                s=50,            
                zorder=2,         
                alpha=0.5)        

    # --- Set labels and title ---
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Conductivity", fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    
    # --- Add legend and grid ---
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    # --- Display the plot ---
    plt.tight_layout()

def plot_anomalies_all_test(all_signals, all_pred_masks, all_gt_masks, all_run_ids, ncols=2):
    """
    Plots all test runs in a single figure with multiple subplots.
    Each subplot shows the signal, predicted anomalies, and ground truth.
    """
    # Use the style you liked from the example
    plt.style.use('seaborn-v0_8-whitegrid')
    
    n_runs = len(all_signals)
    if n_runs == 0:
        print("No data to plot.")
        return

    # Calculate grid size
    nrows = (n_runs + ncols - 1) // ncols
    
    # Create the figure and all subplots (axes)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 6 * nrows), squeeze=False)
    axes = axes.flatten() # Make the 2D grid of axes easy to loop over

    print(f"\nGenerating a single figure with {n_runs} subplots...")

    for i in range(n_runs):
        ax = axes[i] # Get the specific subplot for this run
        
        # Get the data for this run
        original_signal = all_signals[i]
        predicted_mask = all_pred_masks[i]
        ground_truth_mask = all_gt_masks[i]
        run_id = all_run_ids[i]
        time_steps = np.arange(len(original_signal))

        # --- Plotting logic (copied from your friend's style) ---

        # 1. Plot the original signal as a blue line
        ax.plot(time_steps, original_signal, color='blue', label='Signal', zorder=1, alpha=0.7)

        # 2. Plot PREDICTED anomalies (Red X)
        predicted_points = np.full_like(original_signal, np.nan, dtype=float)
        predicted_points[predicted_mask] = original_signal[predicted_mask]
        ax.scatter(time_steps, predicted_points, 
                   color='red', 
                   label='Predicted Anomaly (VAE)', 
                   marker='x', 
                   s=100, 
                   zorder=3)

        # 3. Plot GROUND TRUTH anomalies (Green O)
        if ground_truth_mask is not None:
            ground_truth_mask = np.asarray(ground_truth_mask)
            if len(original_signal) == len(ground_truth_mask):
                ground_truth_points = np.full_like(original_signal, np.nan, dtype=float)
                ground_truth_points[ground_truth_mask] = original_signal[ground_truth_mask]
                ax.scatter(time_steps, ground_truth_points, 
                           color='green', 
                           label='Ground Truth Anomaly', 
                           marker='o', 
                           s=100, 
                           zorder=2, 
                           alpha=0.5)

        # --- Set labels and title for the subplot ---
        ax.set_title(f"Run ID: {run_id}", fontsize=14)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Conductivity")
        
        # Combine legends to avoid duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # --- Finalize the Figure ---
    
    # Hide any unused subplots if the number of runs isn't a perfect multiple of 'ncols'
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("Anomaly Detection Results for All Test Runs", fontsize=20, y=1.03)
    plt.tight_layout()
    plt.show() # Show the single, complete figure

def plot_specific_anomaly_point(original_signal, is_anomalous_mask, anomaly_index, run_id, 
                                context_points=100, y_label="Conductivity", x_label="Time Step (2s interval)"):
    """
    Creates a zoomed-in plot around a specific anomalous point.

    Args:
        original_signal (array-like): The full time series data.
        is_anomalous_mask (array-like): Boolean mask for anomalies.
        anomaly_index (int): The index of the specific anomaly to focus on.
        run_id (str): The ID of the run for the plot title.
        context_points (int, optional): Number of data points to show on either side of the anomaly. Defaults to 100.
        y_label (str, optional): Y-axis label. Defaults to "Conductivity".
        x_label (str, optional): X-axis label. Defaults to "Time Step (2s interval)".
    """
    
    # Ensure data is in NumPy array format
    original_signal = np.asarray(original_signal)
    is_anomalous_mask = np.asarray(is_anomalous_mask)

    if not (0 <= anomaly_index < len(original_signal)):
        print(f"Error: anomaly_index {anomaly_index} is out of bounds for signal of length {len(original_signal)}.")
        return # Exit if index is invalid
    
    if not is_anomalous_mask[anomaly_index]:
        print(f"Warning: The point at index {anomaly_index} is not marked as an anomaly. Plotting context anyway.")

    # Determine the plot range
    start = max(0, anomaly_index - context_points)
    end = min(len(original_signal), anomaly_index + context_points + 1)
    
    time_steps = np.arange(start, end)
    signal_segment = original_signal[start:end]
    mask_segment = is_anomalous_mask[start:end]
    
    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 1. Plot the signal segment
    ax.plot(time_steps, signal_segment, color='blue', label='Signal', zorder=1)
    
    # 2. Highlight anomalous points in the segment
    anomalous_indices_in_segment = np.where(mask_segment)[0]
    if len(anomalous_indices_in_segment) > 0:
        ax.scatter(time_steps[anomalous_indices_in_segment], signal_segment[anomalous_indices_in_segment], 
                   color='red', label='Anomaly', zorder=2, s=50)

    # 3. Highlight the specific point of interest
    ax.axvline(x=anomaly_index, color='purple', linestyle='--', label=f'Focus Point ({anomaly_index})')
    ax.scatter([anomaly_index], [original_signal[anomaly_index]], color='purple', s=200, zorder=3, marker='*',
               label=f'Anomaly at index {anomaly_index}')

    # --- Set labels and title ---
    ax.set_title(f"Zoomed-in View of Anomaly in Run: {run_id}", fontsize=16)
    ax.set_ylabel(y_label)
    ax.set_xlabel(f"{x_label} (Zoomed on Index {anomaly_index})")
    
    # --- Add legend and grid ---
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- Display the plot ---
    plt.tight_layout()
    # We will call plt.show() from the main script loop


# --- 3. Data Loading & Feature Engineering ---
try:
    normal_df_raw = pd.read_csv("data/transform/pivot_normal_conductivity.csv",
                                index_col=False)
    anomaly_df_raw = pd.read_csv(
        "data/transform/pivot_anomaly_conductivity.csv", index_col=False)
except FileNotFoundError as e:
    print(
        f"Error loading data: {e}. Make sure the CSV files are in the correct directory.")
    exit()

normal_df = normal_df_raw[
    normal_df_raw['metric'] == 'conductivity'].reset_index(drop=True)
anomaly_df = anomaly_df_raw[
    anomaly_df_raw['metric'] == 'conductivity'].reset_index(drop=True)

if not all(col in normal_df.columns for col in TS_COLS) or not all(
        col in anomaly_df.columns for col in TS_COLS):
    print(
        "Error: Not all time series columns ('1' to '500') were found in the dataframes.")
    exit()

print("Adding rolling features to normal and anomaly datasets...")
normal_df_featured, mean_cols, std_cols = add_rolling_features(normal_df,
                                                                TS_COLS,
                                                                ROLLING_WINDOW_FEATURES)
anomaly_df_featured, _, _ = add_rolling_features(anomaly_df, TS_COLS,
                                                   ROLLING_WINDOW_FEATURES)
print("Feature engineering complete.")

# --- 4. Create Datasets of Windows ---
print("\nSplitting normal data and creating windows for training...")
normal_train_runs, normal_val_runs = train_test_split(normal_df_featured,
                                                      test_size=0.2,
                                                      random_state=42)

train_windows = create_windows_from_featured_df(normal_train_runs, TS_COLS,
                                                mean_cols, std_cols,
                                                WINDOW_SIZE)
val_windows = create_windows_from_featured_df(normal_val_runs, TS_COLS,
                                              mean_cols, std_cols,
                                              WINDOW_SIZE)
print(
    f"Created {len(train_windows)} training windows from {len(normal_train_runs)} normal runs.")
print(
    f"Created {len(val_windows)} validation windows from {len(normal_val_runs)} normal runs.")

print(normal_val_runs)
# --- 5. Scaling ---
scaler = MinMaxScaler()
scaler.fit(train_windows)
train_windows_scaled = scaler.transform(train_windows)
val_windows_scaled = scaler.transform(val_windows)
print("Data scaling complete.")


# --- 6. PyTorch Datasets and DataLoaders ---

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, windows_data):
        self.data = torch.tensor(windows_data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_dataset = TimeSeriesWindowDataset(train_windows_scaled)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = TimeSeriesWindowDataset(val_windows_scaled)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- 7. VAE Model Definition ---

class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=VAE_HIDDEN_DIM,
                 pre_latent_dim=VAE_PRE_LATENT_DIM,
                 actual_latent_dim=VAE_LATENT_DIM, device=DEVICE):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, pre_latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.mean_layer = nn.Linear(pre_latent_dim, actual_latent_dim)
        self.logvar_layer = nn.Linear(pre_latent_dim, actual_latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(actual_latent_dim, pre_latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(pre_latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        log_var = self.logvar_layer(x)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(DEVICE)
        return mean + torch.exp(log_var / 2) * epsilon

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        return self.decoder(z), mean, log_var


# --- 8. Training Components ---

def vae_loss(recon_x, x, mu, logvar, beta=BETA):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + (beta * kld)) / x.size(0)


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for x in dataloader:
        x = x.to(DEVICE)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(x)
        loss = vae_loss(x_hat, x, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def val_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            x_hat, mu, log_var = model(x)
            loss = vae_loss(x_hat, x, mu, log_var)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 9. Model Training ---
model = VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses, val_losses = [], []

print("\n--- Starting VAE Training (Refined Model) ---")
for epoch in range(N_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = val_epoch(model, val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:03d}/{N_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

print("--- Training Complete ---")
plt.figure(figsize=(10, 6))
plt.plot(range(1, N_EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, N_EPOCHS + 1), val_losses, label='Validation Loss')
plt.title('VAE Training and Validation Loss (Refined)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# --- 10. Anomaly Thresholding ---

def get_reconstructions_and_errors(model, dataloader):
    model.eval()
    all_reconstructions = []
    all_errors = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            errors = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
            all_errors.extend(errors.cpu().numpy())
            all_reconstructions.extend(x_hat.cpu().numpy())
    return np.array(all_reconstructions), np.array(all_errors)

def get_latent_representations(model, dataloader):
    """
    Runs the dataloader through the VAE's encoder and returns all 
    latent mean (mu) vectors.
    """
    model.eval()
    all_latent_mu = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            # Pass data through encoder
            mu, log_var = model.encode(x)
            all_latent_mu.append(mu.cpu().numpy())
    
    return np.concatenate(all_latent_mu, axis=0)

print("Calculating reconstruction errors on normal validation windows to set threshold...")
_, val_errors = get_reconstructions_and_errors(model, val_loader)

# --- NEW: Plot a histogram of the errors ---
plt.figure(figsize=(10, 6))
sns.histplot(val_errors, bins=50, kde=True)
plt.title('Distribution of Validation Reconstruction Errors')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.show()

# --- OPTION 1: Keep your existing method (Good baseline) ---
mean_val_error = np.mean(val_errors)
std_val_error = np.std(val_errors)
threshold_std = mean_val_error + (THRESHOLD_STD_MULTIPLIER * std_val_error)
print(f"Anomaly Threshold (Mean + 3*STD): {threshold_std:.6f}")

# # --- OPTION 2: Use a Quantile (More robust) ---
QUANTILE = 0.97
threshold_quantile = np.quantile(val_errors, QUANTILE)
print(f"Anomaly Threshold ({QUANTILE*100}th Percentile): {threshold_quantile:.6f}")
# --- OPTION 2: Use a Quantile (More robust) ---
# PERCENTILE = 0.95
# threshold_percentile = np.quantile(val_errors, PERCENTILE)
# print(f"Anomaly Threshold ({PERCENTILE*100}th Percentile): {threshold_percentile:.6f}")

# --- Set your final threshold ---
threshold = threshold_quantile # <-- CHOOSE WHICH ONE TO USE HERE
print(f"\nUsing threshold: {threshold:.6f}\n")




# --- 11. Evaluate and Plot Normal Runs (Sanity Check) ---
print("--- Evaluating Normal Runs (Sanity Check) ---")
# Take a sample of normal runs from the validation set to evaluate
sample_normal_runs = normal_val_runs.sample(
    n=min(NORMAL_RUNS_TO_PLOT, len(normal_val_runs)), random_state=42)

for run_id in sample_normal_runs['run_id']:
    normal_run_df = normal_val_runs[normal_val_runs['run_id'] == run_id]
    original_signal = normal_run_df[TS_COLS].values.flatten()
    normal_windows = create_windows_from_featured_df(normal_run_df, TS_COLS,
                                                     mean_cols, std_cols,
                                                     WINDOW_SIZE)
    if len(normal_windows) == 0:
        print(f"Warning: No windows created for normal run '{run_id}'. Skipping.")
        continue

    normal_windows_scaled = scaler.transform(normal_windows)
    normal_dataset = TimeSeriesWindowDataset(normal_windows_scaled)
    normal_loader = DataLoader(normal_dataset, batch_size=BATCH_SIZE,
                               shuffle=False)

    reconstructed_windows_scaled, window_errors = get_reconstructions_and_errors(
        model, normal_loader)
    reconstructed_windows_unscaled = scaler.inverse_transform(
        reconstructed_windows_scaled)

    # --- Window-based anomaly detection logic ---
    # is_window_anomalous = window_errors > threshold
    
    # y_pred_mask_raw = np.zeros(len(original_signal), dtype=bool)
    # for i in range(len(is_window_anomalous)):
    #     anomalies_in_window = is_window_anomalous[i]
    #     y_pred_mask_raw[i : i + WINDOW_SIZE] = np.logical_or(
    #         y_pred_mask_raw[i : i + WINDOW_SIZE],
    #         anomalies_in_window
    #     )
    
    # y_pred_mask = filter_consecutive_anomalies(
    #     y_pred_mask_raw, CONSECUTIVE_WINDOW_THRESHOLD
    # )


    # --- Window-based anomaly detection logic ---
    point_error_windows = (normal_windows - reconstructed_windows_unscaled)**2
    
    # 2. Reconstruct the full point-wise error signal using our new function
    point_errors_full = reconstruct_full_errors_from_windows(
        point_error_windows, len(original_signal), WINDOW_SIZE)
        
    # 3. Create a mask based on point-errors
    #    We'll reuse the window-level `threshold`. This is a starting point,
    #    and you may need a separate, higher threshold for point-errors.
    is_point_anomalous_raw = point_errors_full > threshold 
    
    # 4. Filter for consecutive anomalous *points*
    #    This is now even more important to filter out single-point noise.
    #    You may want to INCREASE the CONSECUTIVE_WINDOW_THRESHOLD to 10 or 15.
    y_pred_mask = filter_consecutive_anomalies(
        is_point_anomalous_raw, 
        CONSECUTIVE_WINDOW_THRESHOLD
    )


    n_filtered_anomalous_points = np.sum(y_pred_mask)
    print(
        f"Normal Run '{run_id}': Found {n_filtered_anomalous_points} anomalous points after filtering ({n_filtered_anomalous_points / len(original_signal) * 100:.1f}%).")

    # For plotting, we use the mean error per window, which is what window_errors contains.
    plot_reconstruction_and_errors(original_signal, run_id,
                                   reconstructed_windows_unscaled,
                                   window_errors, threshold, y_pred_mask)
    
    plt.show() # Show plots for this run
    print(f"Generating simplified anomaly plot for anomaly run '{run_id}'...")
    plot_anomalies(
        original_signal=original_signal,
        is_anomalous_mask=y_pred_mask,
        run_id=run_id
    )
    plt.show() # Show plots for this run
    # And a zoomed-in plot on the first *false positive* anomaly, if any
    # anomalous_indices = np.where(y_pred_mask)[0]
    # if len(anomalous_indices) > 0:
    #     # Pick the first detected anomaly in the sequence to zoom in on
    #     specific_anomaly_index = anomalous_indices[0]
    #     print(f"Generating zoomed-in plot for first *false positive* at index {specific_anomaly_index} in normal run '{run_id}'...")
    #     plot_specific_anomaly_point(
    #     original_signal=original_signal,
    #     is_anomalous_mask=y_pred_mask,
    #     anomaly_index=specific_anomaly_index,
    #     run_id=run_id,
    #     context_points=50 # Show 50 points on each side
    # )
    # plt.show() 




















# --- 12. Evaluate on Anomalous Runs and Calculate Metrics ---
# print("\n--- Evaluating Anomalous Runs (Refined Model) ---")
# all_true_labels, all_predictions, all_scores_for_roc = [], [], []
# unique_run_ids = anomaly_df_featured['run_id'].unique()

# for run_id in unique_run_ids:
#     anomaly_run_df = anomaly_df_featured[
#         anomaly_df_featured['run_id'] == run_id]
#     original_signal = anomaly_run_df[TS_COLS].values.flatten()
#     anomaly_windows = create_windows_from_featured_df(anomaly_run_df, TS_COLS,
#                                                       mean_cols, std_cols,
#                                                       WINDOW_SIZE)
#     if len(anomaly_windows) == 0:
#         print(f"Warning: No windows created for run '{run_id}'. Skipping.")
#         continue

#     anomaly_windows_scaled = scaler.transform(anomaly_windows)
#     anomaly_dataset = TimeSeriesWindowDataset(anomaly_windows_scaled)
#     anomaly_loader = DataLoader(anomaly_dataset, batch_size=BATCH_SIZE,
#                                 shuffle=False)

#     reconstructed_windows_scaled, window_errors = get_reconstructions_and_errors(
#         model, anomaly_loader)
#     reconstructed_windows_unscaled = scaler.inverse_transform(
#         reconstructed_windows_scaled)

#     # --- Window-based anomaly detection logic ---
#     # is_window_anomalous = window_errors > threshold
    
#     # y_pred_mask_raw = np.zeros(len(original_signal), dtype=bool)
#     # for i in range(len(is_window_anomalous)):
#     #     anomalies_in_window = is_window_anomalous[i]
#     #     y_pred_mask_raw[i : i + WINDOW_SIZE] = np.logical_or(
#     #         y_pred_mask_raw[i : i + WINDOW_SIZE],
#     #         anomalies_in_window
#     #     )
    
#     # y_pred_mask = filter_consecutive_anomalies(
#     #     y_pred_mask_raw, CONSECUTIVE_WINDOW_THRESHOLD
#     # )

    
#     # --- NEW: Point-wise anomaly detection logic ---
    
#     # 1. Calculate point-wise squared errors for all windows
#     #    (This is the error for each individual data point)
#     point_error_windows = (anomaly_windows - reconstructed_windows_unscaled)**2
    
#     # 2. Reconstruct the full point-wise error signal using our new function
#     point_errors_full = reconstruct_full_errors_from_windows(
#         point_error_windows, len(original_signal), WINDOW_SIZE)
        
#     # 3. Create a mask based on point-errors
#     #    We'll reuse the window-level `threshold`. 
#     is_point_anomalous_raw = point_errors_full > threshold 
    
#     # 4. Filter for consecutive anomalous *points*
#     y_pred_mask = filter_consecutive_anomalies(
#         is_point_anomalous_raw, 
#         CONSECUTIVE_WINDOW_THRESHOLD
#     )




#     n_filtered_anomalous_points = np.sum(y_pred_mask)
#     print(
#         f"Anomaly Run '{run_id}': Found {n_filtered_anomalous_points} anomalous points after filtering ({n_filtered_anomalous_points / len(original_signal) * 100:.1f}%).")


#     # Create a point-wise score mask from window-wise errors for ROC curve
#     y_score_mask = np.zeros(len(original_signal))
#     for i, window_error in enumerate(window_errors):
#         current_max = y_score_mask[i: i + WINDOW_SIZE]
#         y_score_mask[i: i + WINDOW_SIZE] = np.maximum(current_max, window_error)

#     current_ground_truth_mask = None
#     label_row = anomaly_df_raw[(anomaly_df_raw['run_id'] == run_id) & (
#             anomaly_df_raw['metric'] != 'conductivity')]
#     if not label_row.empty:
#         y_true_mask = (label_row[TS_COLS].values.flatten() == 1)
#         if len(y_true_mask) == len(original_signal):
#             current_ground_truth_mask = y_true_mask
#             all_true_labels.extend(y_true_mask)
#             all_predictions.extend(y_pred_mask)
#             all_scores_for_roc.extend(y_score_mask)
#         else:
#             print(
#                 f"Warning: Length mismatch for run {run_id}. Ground truth: {len(y_true_mask)}, Original Signal: {len(original_signal)}. Skipping ground truth for this plot and overall metrics.")
#     else:
#         print(
#             f"Warning: Could not find ground truth label row for run {run_id}. Skipping ground truth for this plot and overall metrics.")

#     # For plotting, we use the mean error per window, which is what window_errors contains.
#     plot_reconstruction_and_errors(original_signal, run_id,
#                                    reconstructed_windows_unscaled,
#                                    window_errors, threshold, y_pred_mask,
#                                    current_ground_truth_mask)
#     plt.show() # Show the main reconstruction plot

#     # Also generate the simplified anomaly plot for the anomaly run
#     print(f"Generating simplified anomaly plot for anomaly run '{run_id}'...")
#     plot_anomalies(
#         original_signal=original_signal,
#         is_anomalous_mask=y_pred_mask,
#         run_id=run_id
#     )
#     plt.show()


# --- 12. Evaluate on Anomalous Runs and Calculate Metrics ---
print("\n--- Evaluating Anomalous Runs (Refined Model) ---")
all_true_labels, all_predictions, all_scores_for_roc = [], [], []

# --- NEW: Create lists to store results for the big plot ---
all_run_signals = []
all_run_pred_masks = []
all_run_gt_masks = []
all_run_ids_list = []
# ---

unique_run_ids = anomaly_df_featured['run_id'].unique()

for run_id in unique_run_ids:
    anomaly_run_df = anomaly_df_featured[
        anomaly_df_featured['run_id'] == run_id]
    original_signal = anomaly_run_df[TS_COLS].values.flatten()
    anomaly_windows = create_windows_from_featured_df(anomaly_run_df, TS_COLS,
                                                      mean_cols, std_cols,
                                                      WINDOW_SIZE)
    if len(anomaly_windows) == 0:
        print(f"Warning: No windows created for run '{run_id}'. Skipping.")
        continue

    anomaly_windows_scaled = scaler.transform(anomaly_windows)
    anomaly_dataset = TimeSeriesWindowDataset(anomaly_windows_scaled)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=BATCH_SIZE,
                                shuffle=False)

    reconstructed_windows_scaled, window_errors = get_reconstructions_and_errors(
        model, anomaly_loader)
    reconstructed_windows_unscaled = scaler.inverse_transform(
        reconstructed_windows_scaled)

    # --- NEW: Point-wise anomaly detection logic ---
    point_error_windows = (anomaly_windows - reconstructed_windows_unscaled)**2
    point_errors_full = reconstruct_full_errors_from_windows(
        point_error_windows, len(original_signal), WINDOW_SIZE)
    is_point_anomalous_raw = point_errors_full > threshold 
    y_pred_mask = filter_consecutive_anomalies(
        is_point_anomalous_raw, 
        CONSECUTIVE_WINDOW_THRESHOLD
    )

    n_filtered_anomalous_points = np.sum(y_pred_mask)
    print(
        f"Anomaly Run '{run_id}': Found {n_filtered_anomalous_points} anomalous points after filtering ({n_filtered_anomalous_points / len(original_signal) * 100:.1f}%).")

    # --- (This part for ROC/Metrics is unchanged) ---
    y_score_mask = np.zeros(len(original_signal))
    for i, window_error in enumerate(window_errors):
        current_max = y_score_mask[i: i + WINDOW_SIZE]
        y_score_mask[i: i + WINDOW_SIZE] = np.maximum(current_max, window_error)

    current_ground_truth_mask = None
    label_row = anomaly_df_raw[(anomaly_df_raw['run_id'] == run_id) & (
            anomaly_df_raw['metric'] != 'conductivity')]
    if not label_row.empty:
        y_true_mask = (label_row[TS_COLS].values.flatten() == 1)
        if len(y_true_mask) == len(original_signal):
            current_ground_truth_mask = y_true_mask
            all_true_labels.extend(y_true_mask)
            all_predictions.extend(y_pred_mask)
            all_scores_for_roc.extend(y_score_mask)
        else:
            print(
                f"Warning: Length mismatch for run {run_id}. Ground truth: {len(y_true_mask)}, Original Signal: {len(original_signal)}. Skipping ground truth for this plot and overall metrics.")
    else:
        print(
            f"Warning: Could not find ground truth label row for run {run_id}. Skipping ground truth for this plot and overall metrics.")

    # --- (This is where the plotting calls USED to be) ---
    # --- NEW: Instead of plotting, append the results to our lists ---
    all_run_signals.append(original_signal)
    all_run_pred_masks.append(y_pred_mask)
    all_run_gt_masks.append(current_ground_truth_mask)
    all_run_ids_list.append(run_id)

# --- NEW: After the loop, call your new function ONE time ---

# --- NEW: Code to stitch all runs into one graph ---

print("\nConcatenating all test runs into a single graph...")

# 1. Stitch all the signal arrays together
signals_stitched = np.concatenate(all_run_signals)

# 2. Stitch all the prediction mask arrays together
preds_stitched = np.concatenate(all_run_pred_masks)

# 3. Stitch all the ground truth arrays, handling 'None'
# (If a ground truth is 'None', we create a 'False' mask of the same length)
gt_stitched = np.concatenate([
    gt if gt is not None else np.zeros_like(all_run_signals[i], dtype=bool) 
    for i, gt in enumerate(all_run_gt_masks)
])

# 4. Now, call your standard 'plot_anomalies' function on these giant arrays
#    (Make sure you've replaced 'plot_anomalies' with the new version from my previous answer)
plot_anomalies_stitched(
    original_signal=signals_stitched,
    predicted_mask=preds_stitched,
    ground_truth_mask=gt_stitched,
    title="All Test Runs (Stitched Together)",
    x_label="Combined Time Step Index"
)
plt.show()




































def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# --- 13. Overall Confusion Matrix ---
if all_true_labels and all_predictions:
    print("\n--- Overall Point-wise Confusion Matrix (Refined Model) ---")
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Overall Confusion Matrix for All Anomaly Runs (Refined)')
    plt.show()

    # Calculate and print classification metrics
    accuracy, precision, recall, f1 = calculate_metrics(all_true_labels, all_predictions)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
else:
    print(
        "\nCould not generate a confusion matrix because no valid ground truth labels were found.")

# --- 14. Overall ROC Curve ---
if all_true_labels and all_scores_for_roc:
    print("\n--- Overall ROC Curve ---")
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_scores_for_roc)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
else:
    print(
        "\nCould not generate an ROC curve because no valid ground truth labels or scores were found.")

# --- 15. Latent Space Visualization (t-SNE) ---
# print("\n--- Generating Latent Space Visualization (t-SNE) ---")

# # 1. Create a dataloader for ALL anomaly windows (not run-by-run)
# anomaly_all_windows = create_windows_from_featured_df(anomaly_df_featured, TS_COLS,
#                                                       mean_cols, std_cols,
#                                                       WINDOW_SIZE)
# anomaly_all_windows_scaled = scaler.transform(anomaly_all_windows)
# anomaly_all_dataset = TimeSeriesWindowDataset(anomaly_all_windows_scaled)
# # Use a larger batch size for faster inference
# anomaly_all_loader = DataLoader(anomaly_all_dataset, batch_size=BATCH_SIZE * 2, 
#                                 shuffle=False)

# print("Getting latent representations for normal (validation) data...")
# val_latent_vectors = get_latent_representations(model, val_loader)

# print("Getting latent representations for anomalous data...")
# anomaly_latent_vectors = get_latent_representations(model, anomaly_all_loader)

# # 2. Sample data to keep t-SNE fast and balanced
# max_points = 5000 # Max points per class to plot
# n_val = len(val_latent_vectors)
# n_anom = len(anomaly_latent_vectors)

# val_indices = np.random.choice(n_val, size=min(max_points, n_val), replace=False)
# anom_indices = np.random.choice(n_anom, size=min(max_points, n_anom), replace=False)

# val_sample = val_latent_vectors[val_indices]
# anom_sample = anomaly_latent_vectors[anom_indices]

# # 3. Combine vectors and create labels
# all_vectors = np.concatenate([val_sample, anom_sample], axis=0)
# labels = np.array([0] * len(val_sample) + [1] * len(anom_sample))

# print(f"Running t-SNE on {len(all_vectors)} samples ({len(val_sample)} normal, {len(anom_sample)} anomalous)...")
# # Note: perplexity is a key param. 30-50 is a good default.
# tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
# tsne_results = tsne.fit_transform(all_vectors)

# # 4. Plot the results
# print("Plotting t-SNE...")
# val_2d = tsne_results[labels == 0]
# anom_2d = tsne_results[labels == 1]

# plt.figure(figsize=(12, 8))
# plt.scatter(val_2d[:, 0], val_2d[:, 1], c='blue', alpha=0.5, s=10, 
#             label='Normal (Validation)')
# plt.scatter(anom_2d[:, 0], anom_2d[:, 1], c='red', alpha=0.7, s=10,
#             label='Anomaly')

# plt.title(f't-SNE Visualization of VAE Latent Space ({VAE_LATENT_DIM}D -> 2D)')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')

# # Create a custom legend
# blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='Normal (Validation)')
# red_patch = mpatches.Patch(color='red', alpha=0.7, label='Anomaly')
# plt.legend(handles=[blue_patch, red_patch])

# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()





# print("\n--- Analysis Complete ---")