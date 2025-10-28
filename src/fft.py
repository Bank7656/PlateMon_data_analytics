import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_conductivity_fft(df,bath_id,run_ids):
  
  condition = (df['bath_id'] == bath_id)
  bath_df = df[condition][['run_id','time_total','conductivity']]
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

    fft_vals = np.fft.fft(signal_data - np.mean(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), d=dt)
    pos_mask = freqs > 0.005
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


def plot_voltage_fft(df,bath_id,run_ids):
  
  condition = (df['bath_id'] == bath_id)
  bath_df = df[condition][['run_id','time_total','voltage']]
  run_ids = run_ids
  run_dfs = []
  for id in run_ids:
    run_df = bath_df[bath_df['run_id']==id]
    run_df['time_total'] = run_df['time_total'] - run_df['time_total'].iloc[0]
    run_df = run_df[['time_total', 'voltage']].reset_index(drop=True)
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

  # --- Plot Time Domain ---
  for col, color in zip(run_ids, colors):
    axes[0].plot(time, merged_df[col], label=col, color=color)
  axes[0].set_title("Voltage vs Time")
  axes[0].set_xlabel("Time (s)")
  axes[0].set_ylabel("Voltage")
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

    fft_vals = np.fft.fft(signal_data - np.mean(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), d=dt)
    pos_mask = freqs > 0.005
    axes[1].plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), label=col, color=color)
    graph_df = pd.concat([graph_df, pd.DataFrame({f"freq_{col}":freqs[pos_mask], 
                            f"fft_vals_{col}":np.abs(fft_vals[pos_mask])})], axis=1)
  axes[1].set_title("Voltage Spectrum (Frequency Domain)")
  axes[1].set_xlabel("Frequency (Hz)")
  axes[1].set_ylabel("Amplitude")
  axes[1].grid(True)
  axes[1].legend()

  plt.tight_layout()
  plt.show()
  return graph_df

def plot_pH_fft(df,bath_id,run_ids):
  
  condition = (df['bath_id'] == bath_id)
  bath_df = df[condition][['run_id','time_total','pH']]
  run_ids = run_ids
  run_dfs = []
  for id in run_ids:
    run_df = bath_df[bath_df['run_id']==id]
    run_df['time_total'] = run_df['time_total'] - run_df['time_total'].iloc[0]
    run_df = run_df[['time_total', 'pH']].reset_index(drop=True)
    run_df.rename(columns={'pH': id}, inplace=True)
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
  axes[0].set_title("pH vs Time")
  axes[0].set_xlabel("Time (s)")
  axes[0].set_ylabel("pH")
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

    fft_vals = np.fft.fft(signal_data - np.mean(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), d=dt)
    pos_mask = freqs > 0.005
    axes[1].plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), label=col, color=color)
    graph_df = pd.concat([graph_df, pd.DataFrame({f"freq_{col}":freqs[pos_mask], 
                            f"fft_vals_{col}":np.abs(fft_vals[pos_mask])})], axis=1)
  axes[1].set_title("pH Spectrum (Frequency Domain)")
  axes[1].set_xlabel("Frequency (Hz)")
  axes[1].set_ylabel("Amplitude")
  axes[1].grid(True)
  axes[1].legend()

  plt.tight_layout()
  plt.show()
  return graph_df

def plot_temperature_fft(df,bath_id,run_ids):
  
  condition = (df['bath_id'] == bath_id)
  bath_df = df[condition][['run_id','time_total','temperature']]
  run_ids = run_ids
  run_dfs = []
  for id in run_ids:
    run_df = bath_df[bath_df['run_id']==id]
    run_df['time_total'] = run_df['time_total'] - run_df['time_total'].iloc[0]
    run_df = run_df[['time_total', 'temperature']].reset_index(drop=True)
    run_df.rename(columns={'temperature': id}, inplace=True)
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
  axes[0].set_title("temperature vs Time")
  axes[0].set_xlabel("Time (s)")
  axes[0].set_ylabel("temperature")
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

    fft_vals = np.fft.fft(signal_data - np.mean(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), d=dt)
    pos_mask = freqs > 0.005
    axes[1].plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), label=col, color=color)
    graph_df = pd.concat([graph_df, pd.DataFrame({f"freq_{col}":freqs[pos_mask], 
                            f"fft_vals_{col}":np.abs(fft_vals[pos_mask])})], axis=1)
  axes[1].set_title("temperature Spectrum (Frequency Domain)")
  axes[1].set_xlabel("Frequency (Hz)")
  axes[1].set_ylabel("Amplitude")
  axes[1].grid(True)
  axes[1].legend()

  plt.tight_layout()
  plt.show()
  return graph_df