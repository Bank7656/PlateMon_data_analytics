import os
from tqdm import tqdm_notebook as tqdm
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
# import plotly.express as px
# import plotly.subplots as psp
# import plotly.graph_objs as go

GREEN = "\033[92m"
RESET = "\033[00m"

BATHS_DIR = "baths"
RUNS_DIR = "runs"
OUTPUT_DIR_NAME = "monitoring_data"
SAVE_MODE = "save"



PARAMETERS = ['pH' ,'voltage', 'current', 'conductivity', 'temperature']

UNITS = {
    'pH': '',
    'voltage': '(V)',
    'current': '(A)',
    'conductivity': '(mS/cm)',
    'temperature': '(°C)'
}


current_row_index = 2

sns.set_context('notebook')
sns.set_theme(style="darkgrid", 
                rc={"figure.dpi":300, 'savefig.dpi':300, 'lines.linewidth':1.5})

def plot_single_bath(df, bath_ids, param, y_pos: list[int] = []) -> None:
    """Plots a single electroplating bath parameter over time.

    Visualizes a specified parameter for given 'run_id's within a single bath.
    Plots lines for each run, colored by 'run_id'.

    Args:
        df (pandas.DataFrame): Electroplating data with 'run_id', 'time_total', and `param` columns.
        bath_ids (list[str]): List of 'run_id's to plot.
        param (str): The parameter column name to plot (e.g., 'pH', 'voltage').
        y_pos (list[float], optional): `[min, max]` for y-axis limits. Defaults to auto-scale.

    Returns:
        None: Displays the plot.

    Raises:
        This function does not raise any exceptions.
    
    Examples:
        Y_LIM = [[0, 5], [0, 5], [0, 3], [40, 80], [40, 80]]
        BATH = ['18.1', '18.2', '18.3', '18.4', '18.5', '19.1', '19.2']
        for param, y_pos in zip(PARAMETERS, Y_LIM):
            plot_single_bath(two_side_plate, BATH, param, y_pos)
    """
    plot_data = df[df['run_id'].isin(bath_ids)]
    g = sns.relplot(
        data=plot_data, kind='line',
        x='time_total', y=param,
        hue='run_id', aspect=2.
    )
    g.set_axis_labels("Plating Time (s)", param)
    if y_pos:
        g.ax.set_ylim(y_pos[0], y_pos[1])
    g.figure.suptitle(f'{param} vs. Plating Time')
    g.figure.subplots_adjust(top=0.9)
    return None

def plot_all_params(df, run_ids, variables, mode=None, directory=None) -> plt.Figure:
    """ Ploting every measured electroplating bath parameter in time series.

    This function will create subplot for each parameters then iterate to create 
    line plot, title, label, and limits the value for each parameters.
    Last subplot will be for the legend of each electroplating experiment "run_id" 
    that will catagorize by color gradient. 

    This function also have an optional argument "mode" that can change into saving mode "save".
    It will not show the graph to the interface but instead save it into .png file depends on
    "run_ids". Save mode will automatically create "OUTPUT_DIR_NAME" directory
    then create sub-directory as "BATH_DIR" and "RUN_DIR" depends on a length of "baths_ids" list.

        - If "baths_ids" has length of 1. It will create file name with "run_id" 
        otherwise it will create with "Bath_id" instead.
        - If "baths_ids" has length of 1. It will also create sub_directory for each day 
        depends on the "run_id"

    Args:
        df (pandas.DataFrame): The electroplating monitoring dataset.
        run_ids (list[str]): A list of 'run_id' strings representing specific
        electroplating experiments to plot. These do not need to belong to
        the same electroplating bath.
        variables (dict[str, list[float]]): dictionary of paramenters. Name as a key and a list of [min, max] of the 
            plot axis of each parameters. Right now fixed 5 parameters as
            ['conductivity', 'pH', 'temperature', 'voltage', 'current'] can be switching places.
        mode (str, optional): Controls the function's behavior.
            - Set to "save" to save the plots as PNG files to disk instead of
            displaying them.
            - Defaults to displaying the plots interactively.
        directory (str, optional): **[INTERNAL USE ONLY]** The base directory
            where plots will be saved when `mode` is set to `"save"`.
    
    Returns:
        matplotlib.figure.Figure: The Matplotlib Figure object containing the plots.
            This is returned regardless of whether the plots are displayed or saved.
    
    Raises:
        This function does not raise any exceptions.
    
    Example:

        PARAMS = {
            'pH': [0, 5],
            'voltage': [0, 5],
            'current': [0, 5],
            'conductivity': [45, 60],
            'temperature': [45, 60]
        }

        condition = (internal_df['bath_id'] == 'Bath_1')
        full_bath_1 = list(internal_df[condition]['run_id'].unique())
        plot_all_params(internal_df, full_bath_1, PARAMS)

        This will plot whole bath graph for every "run_id" consecutively
        If you want to plot individualy you can use this instead
            plot_all_params(internal_df, [JUL_1_2], PARAMS)
    """
    row = 2
    count = 0
    params = list(variables.keys())
    num_groups = len(variables)
    col = int((num_groups + 1) / 2)
    legend_col = int(len(run_ids) / 15) + 1
    fig, axes = plt.subplots(row, col, figsize=(5 * num_groups, 10), squeeze=False)
    plt.suptitle("Parameters Monitoring Data", fontsize=24)
    plt.subplots_adjust(hspace=0.35)
    plot_data = df[df['run_id'].isin(run_ids)]
    for i in range(row):
        for j in range(col):
            if (i == row - 1 and j == col - 1):
                handles, labels = axes[0][j].get_legend_handles_labels()
                legend_bbox = axes[i][j].get_position()
                fig.legend(handles, labels,
                        loc='center', # Place the legend centered within its "subplot" area
                        ncol=legend_col, 
                        bbox_to_anchor=(legend_bbox.x0 + legend_bbox.width / 2,
                                        legend_bbox.y0 + legend_bbox.height / 2),
                        bbox_transform=fig.transFigure,
                        frameon=True, # Display a frame around the legend
                        title="Run IDs" # Optional: Add a title to the legend
                        )
                axes[i][j].remove()
                break
            if plot_data.empty:
                axes[i][j].set_title(f'No data for: {", ".join(run_ids)}')
                continue                                
            ax = sns.lineplot(
                data=plot_data,
                x='time_total', y=params[count],
                hue='run_id', ax=axes[i][j]
            )
            if (params[count] in ['temperature']):
                minor_locator = ticker.AutoMinorLocator(2)
                ax.yaxis.set_minor_locator(minor_locator)
                ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:.1f}"))
            elif (params[count] in ['pH', 'current']):
                minor_locator = ticker.AutoMinorLocator(2)
                ax.yaxis.set_minor_locator(minor_locator)
                ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:.2f}"))
            elif (params[count] in ['conductivity']):
                minor_locator = ticker.AutoMinorLocator(4)
                ax.yaxis.set_minor_locator(minor_locator)
                ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:.2f}"))
            
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
            ax.set_xlabel('Plating Time (sec)', fontsize=16)
            if variables[params[count]]:
                ax.set_ylim(variables[params[count]][0], variables[params[count]][1])
            name = params[count]
            if params[count] != "pH":
                name = list(name)
                name[0] = name[0].upper()
                name = "".join(name)
            y_label = f"{name} {UNITS[params[count]]}"
            ax.set_ylabel(y_label, fontsize=16)
            # 5. Set the title for each individual subplot
            axes[i][j].set_title(f'{name} vs Plating time', fontsize=20)
            axes[i][j].legend_.remove() # Ensure each subplot has its own legend  
            count += 1
    if mode == "save":
        if len(run_ids) == 1:
            save_graph(fig, run_ids[0], directory)
        else:
            condition = df['run_id'] == run_ids[0]
            bath = df[condition]['bath_id'].unique()[0]
            save_graph(fig, bath, directory)
    plt.close(fig)
    return fig

def create_dir(dir_name) -> None:
    """ Create directory

    Makes the mkdir() system call to the operating system kernel 
    to perform the directory creation.

    Args:
        dir_name (str): Name that you want to use to create a directory
    
    Return:
        None
    
    Raises:
        OSError: If system call was not successfully
    """    
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        # print(f"Directory {dir_name} already exists.")
        pass
    except OSError as e:
        raise OSError(f"Error creating directory: {e}")
    return

def get_date_dir(run_id):
    """Generates a directory name based on the format of a run ID.

    This function parses a `run_id` string and extracts relevant date information
    to construct a standardized directory name. The format of the `run_id`
    determines the output:
    - If `run_id` contains a '.', it assumes a 'MM.DD' format (e.g., '06.15')
      and returns 'JUN_DD'.
    - If `run_id` contains a '_', it assumes a 'MMM_DD' format (e.g., 'JUL_29')
      and returns 'MMM_DD'.
    - If neither character is present, the `run_id` itself is returned.

    Args:
        run_id (str): The unique identifier for an experiment run,
            expected in formats like 'DD.MM' (e.g., '15.06'), 'MMM_DD'
            (e.g., 'JUL_29'), or a plain string.

    Returns:
        str: A standardized directory name based on the `run_id` format.
    """
    if "." in run_id:
        date_lst = run_id.split(".")
        dir_name = f"JUN_{date_lst[0]}"
        return dir_name
    elif "_" in run_id:
        date_lst = run_id.split("_")
        dir_name = f"{date_lst[0]}_{date_lst[1]}"
        return dir_name
    else:
        return run_id


def save_individual_run(df, params):
    """Generates and saves time series plots for each individual experiment run.

    This function iterates through all unique 'run_id's.
    For each run, it creates a dedicated subdirectory within `OUTPUT_DIR_NAME/RUNS_DIR`
    whose name is derived from the `run_id` (e.g., 'JUN_15' for '15.06', 'JUL_29' for 'JUL_29').
    It then calls `plot_all_params` in 'save' mode to generate and save plots
    for that specific run within its designated daily directory.

    Args:
        df (pandas.DataFrame): The main electroplating monitoring dataset.
        params (dict[str, list[float]]): A dictionary defining the parameters
            to plot and their y-axis display limits.
            - Keys: Parameter names (e.g., 'pH', 'voltage').
            - Values: `[min_value, max_value]` for the y-axis range.

    Returns:
        None: The function performs file saving as a side effect and does not
            return any value.
    """
    create_dir(OUTPUT_DIR_NAME)
    sub_dir = f"./{OUTPUT_DIR_NAME}/{RUNS_DIR}"
    create_dir(sub_dir)
    runs_id = df['run_id'].unique()
    for id in tqdm(runs_id, unit='File', colour='green', smoothing=1):
        data_dir =  sub_dir + "/" +  get_date_dir(id) 
        create_dir(data_dir)
        plot_all_params(df, [id], params, SAVE_MODE, data_dir)
    print(f"{GREEN}[Save Completed >_<]{RESET}")
    return


def save_bath(df, bath_ids, params):
    """Generates and saves time series plots for multiple electroplating baths.

    This function automates the process of creating and saving time series plots
    for all parameters within specified electroplating baths. It iterates through 
    each `bath_id` provided, calls `plot_all_params` in 'save' mode to generate 
    and save the plots for all runs within that specific bath.

    Args:
        df (pandas.DataFrame): The main electroplating monitoring dataset.
        bath_ids (list[str]): A list of unique 'bath_id' strings for which
            plots should be generated and saved.
        params (dict[str, list[float]]): A dictionary defining the parameters
            to plot and their y-axis display limits.
            - Keys: Parameter names (e.g., 'pH', 'voltage', 'current').
            - Values: `[min_value, max_value]` for the y-axis range.

    Returns:
        None: The function performs file saving as a side effect and does not
            return any value.
    """
    create_dir(OUTPUT_DIR_NAME)
    sub_dir = f"./{OUTPUT_DIR_NAME}/{BATHS_DIR}"
    create_dir(sub_dir)
    for bath in tqdm(bath_ids, unit='File', colour='green', smoothing=1):
        condition = (df['bath_id'] == bath)
        single_bath = list(df[condition]['run_id'].unique())
        plot_all_params(df, single_bath, params, SAVE_MODE, sub_dir)
    print(f"{GREEN}[Save Completed >_<]{RESET}")
    return


def save_graph(fig, name, dir_name):
    """Saves a Matplotlib figure to a specified directory.

    Constructs a filename and filepath, then saves the provided Matplotlib
    Figure object as a PNG image. Prints a success message upon completion.

    Args:
        fig (matplotlib.figure.Figure): The Matplotlib Figure object to be saved.
        name (str): A descriptive name for the graph, used in the filename
            (e.g., 'pH_vs_time').
        dir_name (str): The path to the directory where the graph will be saved.
            The directory will be created if it does not exist.

    Returns:
        None: The function saves the file as a side effect and does not return a value.

    Raises:
        OSError: If the `dir_name` is invalid or there are permission issues
            preventing file saving.
        TypeError: If `fig` is not a valid Matplotlib Figure object.
    """
    # Ensure the directory exists before saving
    os.makedirs(dir_name, exist_ok=True)

    filename = f"monitoring_{name}.png"
    filepath = os.path.join(dir_name, filename) # Use os.path.join for cross-platform compatibility

    try:
        fig.savefig(filepath)
        print(f'{GREEN}[status]{RESET} {filename} was saved successfully at {filepath}')
    except Exception as e:
        # Catching a broad exception for demonstration; in real code, be more specific.
        raise OSError(f"Failed to save graph to {filepath}: {e}") from e





# def plot_all_params(df, bath_ids, variables) -> None:

#     row = 2
#     count = 0
#     params = list(variables.keys())
#     num_groups = len(variables)
#     col = int((num_groups + 1) / 2)
#     traces = []
#     plot_data = df[df['run_id'].isin(bath_ids)]
#     print(plot_data.shape[0])
#     for region, geo_region in plot_data.groupby('run_id'):
#         traces.append(
#             go.Scatter(
#                 x=geo_region.time_total,
#                 y=geo_region.pH,
#                 name=region,
#                 mode='lines'
#             )
#         )
#         # fig.add_scatter(
#         #     x=geo_region.time_total,
#         #     y=geo_region.pH,
#         #     name=region,
#         #     mode='line'
#         # )
#         # fig = px.line(
#         #     data_frame=plot_data, 
#         #     x='time_total', 
#         #     y=params[0],
#         # )
#     fig = go.Figure(data=traces)
#     config = {'scrollZoom': True}
#     fig.update_layout(height=500, width=1000, 
#                         xaxis=dict(
#                             minallowed=0, 
#                             maxallowed=plot_data.shape[0]
#                         ),
#                         yaxis=dict(
#                             minallowed=0,
#                             maxallowed=plot_data.shape[1]
#                         ),
#     )
#     fig.show(config=config)

# def open_excel(filepath):
#     try:
#         df = pd.read_excel(filepath)
#     except FileNotFoundError:
#         print("File not found")
#         return None
#     return df


# def clean_data(df):
#     df_filtered = df[(df['Voltage'] != '-') & (df['Current'] != '-')]
#     return df_filtered


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
  print(merged_df.head(5))
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
  axes[0].set_ylabel("Conductivity")
  axes[0].grid(True)
  axes[0].legend()

  from scipy import signal
  # --- Plot Frequency Domain ---
  graph_df = None
  for col, color in zip(run_ids, colors):
    signal_data = merged_df[col].dropna().values
    if len(signal_data) < 2:
      continue  # skip if too short
    if np.allclose(signal_data, signal_data[0]):
      continue  # skip constant signals

    signal_data = signal.detrend(signal_data)

    fft_vals = np.fft.fft(signal_data - np.mean(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), d=dt)
    pos_mask = freqs > 0.005
    axes[1].plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), label=col, color=color)
    
    graph_df = pd.concat([graph_df, pd.DataFrame({f"freq_{col}":freqs[pos_mask], f"fft_vals_{col}":np.abs(fft_vals[pos_mask])})], axis=1)
    

  axes[1].set_title("Voltage Spectrum (Frequency Domain)")
  axes[1].set_xlabel("Frequency (Hz)")
  axes[1].set_ylabel("Amplitude")
  axes[1].grid(True)
  axes[1].legend()

  print(graph_df)
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