import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, gaussian_kde
from statsmodels.tsa.stattools import acf
import warnings

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore")

# =============================================================================
# 1. DATA SETUP & DATA GENEARATION
# =============================================================================

def get_all_blocks(series_list, block_size):
    """
    Extracts all overlapping blocks from a list of series.
    """
    all_blocks = []
    for series in series_list:
        n = len(series)
        # Get starting indices for this series
        max_start_index = n - block_size + 1
        for start in range(max_start_index):
            all_blocks.append(series[start : start + block_size])
    
    # Return as a NumPy array for efficient sampling
    return np.array(all_blocks)

def generate_pooled_mbb(block_pool, series_length, block_size=10, n_samples=1000):
    """
    Generates a new series by sampling from the pooled blocks.
    """
    generated_series = []
    for num in range(n_samples):
        num_blocks = len(block_pool)
    
        # Calculate how many blocks to sample
        num_blocks_to_sample = int(np.ceil(series_length / block_size))
    
        # Sample indices from the pool
        sampled_indices = np.random.choice(
            np.arange(num_blocks),
            size=num_blocks_to_sample,
            replace=True
        )
    
        # Create the new series
        new_series_list = [block_pool[i] for i in sampled_indices]
        new_series_combined = np.concatenate(new_series_list)
    
        # Trim to the original length
        new_series_combined[:series_length]
        
        # Store the new series
        generated_series.append(new_series_combined)
        
    return generated_series

# =============================================================================
# 2. PRE-PROCESSING FUNCTION
# =============================================================================

def preprocess_data(series_list):
    """
    Applies the same de-meaning and truncating to the original data
    that you used before feeding it to the bootstrap.
    """
    print("Pre-processing original data (de-meaning and truncating)...")
    
    # 1. Find minimum length
    min_len = min(len(s) for s in series_list)
    
    processed_list = []
    for s in series_list:
        # 2. De-mean each series individually
        s_demeaned = s - np.mean(s)
        # 3. Truncate to min_len
        processed_list.append(s_demeaned[:min_len])
        
    return processed_list, min_len

# =============================================================================
# 3. VALIDATION FUNCTIONS
# =============================================================================

def plot_visual_inspection(original_processed, synthetic_data):
    """Plots one original vs. two synthetic series for a quick sanity check."""
    plt.figure(figsize=(15, 5))
    plt.title("Validation 1: Visual Inspection (Sanity Check)")
    
    # Plot the first processed original series
    plt.plot(original_processed[0], label='Original 1 (Processed)', color='black', linewidth=2)
    
    # Plot two synthetic series
    plt.plot(synthetic_data[0], label='Synthetic 1', color='blue', alpha=0.7)
    plt.plot(synthetic_data[1], label='Synthetic 2', color='green', alpha=0.7)
    
    plt.legend()
    plt.show()

def plot_summary_stats(original_processed, synthetic_data):
    """Compares the distribution of standard deviations."""
    plt.figure(figsize=(12, 6))
    plt.title("Validation 2: Distribution of Standard Deviation")
    
    # 1. Get stats for synthetic data
    std_synthetic = [np.std(s) for s in synthetic_data]
    
    # 2. Plot histogram of synthetic stats
    plt.hist(std_synthetic, bins=30, alpha=0.7, label=f'Synthetic Series (n={len(synthetic_data)})', density=True)
    
    # 3. Get stats for original data
    std_original = [np.std(s) for s in original_processed]
    
    # 4. Overlay original stats as vertical lines
    colors = ['red', 'orange', 'yellow', 'pink', 'cyan']
    for i, std_val in enumerate(std_original):
        plt.axvline(
            std_val, 
            color=colors[i % len(colors)], 
            linestyle='--', 
            linewidth=2, 
            label=f'Original Series {i+1} Std Dev'
        )
    
    plt.xlabel("Standard Deviation")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_value_distributions(original_processed, synthetic_data):
    """Compares the KDE of all data points and runs a K-S test."""
    
    # 1. Pool all data points into two big arrays
    pool_original = np.concatenate(original_processed)
    pool_synthetic = np.concatenate(synthetic_data)
    
    # 2. Run the two-sample K-S test
    ks_stat, p_value = ks_2samp(pool_original, pool_synthetic)
    
    # 3. Create KDE plots
    plt.figure(figsize=(12, 6))
    title = f"Validation 3: Value Distribution (KDE)\nK-S Test p-value: {p_value:.4f}"
    if p_value > 0.05:
        title += " (Good - Distributions are similar)"
    else:
        title += " (Warning - Distributions are different)"
    plt.title(title)
    
    # Create a shared x-axis
    x_min = min(pool_original.min(), pool_synthetic.min())
    x_max = max(pool_original.max(), pool_synthetic.max())
    x_grid = np.linspace(x_min, x_max, 500)
    
    # Plot original KDE
    kde_orig = gaussian_kde(pool_original)
    plt.plot(x_grid, kde_orig(x_grid), label='Original Data', color='black', linewidth=2)
    
    # Plot synthetic KDE
    kde_synth = gaussian_kde(pool_synthetic)
    plt.plot(x_grid, kde_synth(x_grid), label='Synthetic Data', color='blue', linestyle='--', linewidth=2)
    
    plt.xlabel("Data Point Value (De-meaned)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def calculate_average_acf(series_list, nlags):
    """Helper function to get the mean ACF across many series."""
    all_acfs = []
    for s in series_list:
        # Calculate ACF for one series
        all_acfs.append(acf(s, nlags=nlags, fft=True))
    
    # Return the average ACF across all series
    return np.mean(all_acfs, axis=0)

def plot_acf_comparison(original_processed, synthetic_data, nlags=400):
    """Compares the average Autocorrelation Functions."""
    
    # 1. Get average ACF for original data
    avg_acf_original = calculate_average_acf(original_processed, nlags)
    
    # 2. Get average ACF for synthetic data
    avg_acf_synthetic = calculate_average_acf(synthetic_data, nlags)
    
    # 3. Plot them
    plt.figure(figsize=(15, 6))
    plt.title("Validation 4: Average Autocorrelation (Pattern) Comparison")
    
    # Plot as stems
    plt.stem(
        avg_acf_original, 
        linefmt='-k', 
        markerfmt='ok', 
        basefmt='k-', 
        label='Average Original ACF'
    )
    plt.stem(
        avg_acf_synthetic + 0.01, # Offset slightly to see both
        linefmt='--b', 
        markerfmt='xb', 
        basefmt='b-', 
        label='Average Synthetic ACF'
    )
    
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.show()