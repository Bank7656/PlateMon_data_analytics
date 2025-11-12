import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_conductivity_fft(df,bath_id,run_ids):
  
  if bath_id != None:
    condition = (df['bath_id'] == bath_id)
    bath_df = df[condition][['run_id','time_total','conductivity']]
  else:
    bath_df = df[['run_id','time_total','conductivity']]
  run_ids = run_ids
  run_dfs = []
  for id in run_ids:
    run_df = bath_df[bath_df['run_id']==id]
    run_df['time_total'] = run_df['time_total'] - run_df['time_total'].iloc[0]
    run_df = run_df[['time_total', 'conductivity']].reset_index(drop=True)
    run_df.rename(columns={'conductivity': id}, inplace=True)
    run_dfs.append(run_df)

  min_len = min(len(rdf) for rdf in run_dfs)
  run_dfs = [rdf.iloc[:min_len] for rdf in run_dfs]  # truncate longer runs
  # Merge all runs by index (fill shorter ones with NaN)
  merged_df = pd.concat(run_dfs, axis=1)
  # Extract time from first run
  if isinstance(merged_df['time_total'], pd.Series):
    time = merged_df['time_total'].values
  else:
    time = merged_df['time_total'].iloc[:, 0].values
  
  dt = np.mean(np.diff(time))
  fs = 1 / dt  # sampling frequency
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))

  # --- Plot Time Domain ---
  for col, color in zip(run_ids, colors):
    axes[0].plot(time, merged_df[col], label=col, color=color)
  axes[0].set_title("Conductivity vs Time")
  axes[0].set_xlabel("Time (s)")
  axes[0].set_ylabel("Conductivity")
  axes[0].grid(True)
  axes[0].legend()

  # --- Plot Frequency Domain ---
  graph_df = None
  for col, color in zip(run_ids, colors):
    signal_data = merged_df[col].dropna().values
    if len(signal_data) < 2:
      continue  # skip if too short
    if np.allclose(signal_data, signal_data[0]):
      continue  # skip constant signals

    # signal_data = signal.detrend(signal_data)

    fft_vals = np.fft.fft(signal_data - np.mean(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), d=dt)
    pos_mask = freqs > 0.01
    axes[1].plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), label=col, color=color)
    graph_df = pd.concat([graph_df, pd.DataFrame({f"freq_{col}":freqs[pos_mask], 
                            f"fft_vals_{col}":np.abs(fft_vals[pos_mask])})], axis=1)
  axes[1].set_title("Conductivity Spectrum (Frequency Domain)")
  axes[1].set_xlabel("Frequency (Hz)")
  axes[1].set_ylabel("Amplitude")
  axes[1].grid(True)
  axes[1].legend()

  plt.tight_layout()
  plt.show()
  return graph_df

def plot_avg_fft_bars(df):
    import re
    """
    Plot overlayed bar charts of average FFT amplitude for each run in a DataFrame.

    Parameters:
    df : pandas.DataFrame
        A dataframe containing columns like:
        'freq_JUL_22_1', 'fft_vals_JUL_22_1', 'freq_JUL_22_2', 'fft_vals_JUL_22_2', etc.
    """

    # Identify runs automatically
    run_ids = sorted(
        list(
            set(
                re.sub(r"^freq_|^fft_vals_", "", c)
                for c in df.columns
                if c.startswith(("freq_", "fft_vals_"))
            )
        )
    )

    bins = np.arange(0.01, 0.26, 0.01)
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    plt.figure(figsize=(10, 5))

    for i, run in enumerate(run_ids):
        freq_col = f"freq_{run}"
        amp_col = f"fft_vals_{run}"

        if freq_col not in df.columns or amp_col not in df.columns:
            continue

        # Select data within the frequency range of interest
        sub = df.loc[
            (df[freq_col] >= 0.01) & (df[freq_col] <= 0.25), [freq_col, amp_col]
        ].copy()
        sub["freq_bin"] = pd.cut(sub[freq_col], bins=bins, right=False)

        # Compute average amplitude in each bin
        avg = (
            sub.groupby("freq_bin")[amp_col]
            .mean()
            .reset_index()
            .dropna()
        )
        avg["freq_mid"] = avg["freq_bin"].apply(lambda x: x.mid)

        # Plot bars with small offset so they’re side-by-side
        plt.bar(
            avg["freq_mid"],
            avg[amp_col],
            width=0.01,
            label=run,
            color=colors[i],
            alpha=0.7,
            edgecolor="black",
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average Amplitude")
    plt.title("Average Amplitude per 0.01 Hz Interval (0.01–0.25 Hz)")
    plt.legend(title="Run ID")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_voltage_fft(df,bath_id,run_ids):
  
  if bath_id != None:
    condition = (df['bath_id'] == bath_id)
    bath_df = df[condition][['run_id','time_total','voltage']]
  else:
    bath_df = df[['run_id','time_total','voltage']]
  run_ids = run_ids
  run_dfs = []
  for id in run_ids:
    run_df = bath_df[bath_df['run_id']==id]
    run_df['time_total'] = run_df['time_total'] - run_df['time_total'].iloc[0]
    run_df = run_df[['time_total', 'voltage']].reset_index(drop=True)
    run_df = run_df[run_df['time_total']>200]
    run_df.rename(columns={'voltage': id}, inplace=True)
    run_dfs.append(run_df)

  min_len = min(len(rdf) for rdf in run_dfs)
  run_dfs = [rdf.iloc[:min_len] for rdf in run_dfs]  # truncate longer runs
  # Merge all runs by index (fill shorter ones with NaN)
  merged_df = pd.concat(run_dfs, axis=1)
  # Extract time from first run
  if isinstance(merged_df['time_total'], pd.Series):
    time = merged_df['time_total'].values
  else:
    time = merged_df['time_total'].iloc[:, 0].values
  
  dt = np.mean(np.diff(time))
  fs = 1 / dt  # sampling frequency

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))

  from scipy import signal
  # --- Plot Time Domain ---
  for col, color in zip(run_ids, colors):
    signal_data = merged_df[col].dropna().values
    if len(signal_data) < 2:
      continue  # skip if too short
    if np.allclose(signal_data, signal_data[0]):
      continue  # skip constant signals

    signal_data = signal.detrend(signal_data)
    axes[0].plot(time, merged_df[col], label=col, color=color)
  axes[0].set_title("Voltage vs Time")
  axes[0].set_xlabel("Time (s)")
  axes[0].set_ylabel("Conductivity")
  axes[0].grid(True)
  axes[0].legend()

  # --- Plot Frequency Domain ---
  graph_df = None
  for col, color in zip(run_ids, colors):
    signal_data = merged_df[col].dropna().values
    if len(signal_data) < 2:
      continue  # skip if too short
    if np.allclose(signal_data, signal_data[0]):
      continue  # skip constant signals

    # signal_data = signal.detrend(signal_data)

    fft_vals = np.fft.fft(signal_data - np.mean(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), d=dt)
    pos_mask = freqs > 0.01
    axes[1].plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), label=col, color=color)
    
    graph_df = pd.concat([graph_df, pd.DataFrame({f"freq_{col}":freqs[pos_mask], f"fft_vals_{col}":np.abs(fft_vals[pos_mask])})], axis=1)
    

  axes[1].set_title("Voltage Spectrum (Frequency Domain)")
  axes[1].set_xlabel("Frequency (Hz)")
  axes[1].set_ylabel("Amplitude")
  axes[1].grid(True)
  axes[1].legend()

  plt.tight_layout()
  plt.show()
  return graph_df

def store_separated_runs(df, run_ids, param):
    # Fixed time base: 0 to 1200 s, step 2 → 601 rows
    fixed_time = np.arange(0, 1201, 2)

    # Container for aligned runs
    aligned_runs = []

    for run_id in run_ids:
        # Extract data for this run
        run_df = df[df['run_id'] == run_id][['time_total', param]].copy()

        # Normalize time to start at 0
        if not run_df.empty:
            run_df['time_total'] = run_df['time_total'] - run_df['time_total'].iloc[0]
        else:
            # Empty run — just create NaN series
            run_df = pd.DataFrame({'time_total': fixed_time, param: np.nan})

        # Reindex (interpolate or fill NaN as needed)
        run_df = run_df.set_index('time_total').reindex(fixed_time).reset_index()
        run_df.rename(columns={'index': 'time_total', param: run_id}, inplace=True)
        aligned_runs.append(run_df)

    # Merge all runs on the same time base
    merged_df = aligned_runs[0][['time_total']].copy()
    for run_df in aligned_runs:
        merged_df = merged_df.merge(run_df[['time_total', run_df.columns[1]]],
                                    on='time_total', how='outer')

    # Sort by time (just in case)
    merged_df = merged_df.sort_values('time_total').reset_index(drop=True)

    return merged_df