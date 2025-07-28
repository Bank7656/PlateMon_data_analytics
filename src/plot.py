import os
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
# import plotly.express as px
# import plotly.subplots as psp
# import plotly.graph_objs as go

GREEN = "\033[92m"
RESET = "\033[00m"

OUTPUT_DIR_NAME = "output"
SAVE_MODE = "save"

current_row_index = 2

sns.set_context('notebook')
sns.set_theme(style="darkgrid", 
                rc={"figure.dpi":300, 'savefig.dpi':300, 'lines.linewidth':1.5})

def create_output_dir() -> None:
    try:
        os.mkdir(OUTPUT_DIR_NAME)
    except FileExistsError:
        print(f"Directory {OUTPUT_DIR_NAME} already exists.")
    except OSError as e:
        print(f"Error creating directory: {e}")
    return

def plot_bath(df, bath_ids, param):
    sns.set_context('notebook')
    sns.set_theme(style="darkgrid", 
                  rc={"figure.dpi":150, 'savefig.dpi':150, 'lines.linewidth':1.5})

    # 1. Create a figure with a subplot for each group in the list
    num_groups = len(bath_ids)
    fig, axes = plt.subplots(1, num_groups, figsize=(6 * num_groups, 5), squeeze=False)
    axes = axes.flatten()  # Ensure axes is always a 1D array for easy looping

    for i, bath in enumerate(bath_ids):
        plot_data = df[df['run_id'].isin(bath)]

        # If no data is found for this group, skip to avoid errors
        if plot_data.empty:
            axes[i].set_title(f'No data for: {", ".join(bath)}')
            continue
        sns.lineplot(
            data=plot_data,
            x='cumulative_time', y=param,
            hue='run_id', ax=axes[i]
        )

        # 5. Set the title for each individual subplot
        axes[i].set_title(f'Runs: {", ".join(map(str, bath))}')
        axes[i].legend() # Ensure each subplot has its own legend

    plt.show()
    return None

def plot_all_params(df, bath_ids, variables, mode=None) -> plt.Figure:

    row = 2
    count = 0
    params = list(variables.keys())
    num_groups = len(variables)
    col = int((num_groups + 1) / 2)
    # if  (nu)
    legend_col = int(len(bath_ids) / 15) + 1
    fig, axes = plt.subplots(row, col, figsize=(5 * num_groups, 10), squeeze=False)
    plt.suptitle("Parameters Monitoring Data", fontsize=24)
    plt.subplots_adjust(hspace=0.35)
    plot_data = df[df['run_id'].isin(bath_ids)]
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
                axes[i][j].set_title(f'No data for: {", ".join(bath_ids)}')
                continue
            ax = sns.lineplot(
                data=plot_data,
                x='time_total', y=params[count],
                hue='run_id', ax=axes[i][j]
            )
            ax.set_xlabel('Plating Time (sec)', fontsize=16)
            ax.set_ylabel(params[count], fontsize=16)
            if variables[params[count]]:
                ax.set_ylim(variables[params[count]][0], variables[params[count]][1])
            # 5. Set the title for each individual subplot
            axes[i][j].set_title(f'{params[count]} vs Plating time', fontsize=20)
            axes[i][j].legend_.remove() # Ensure each subplot has its own legend  
            count += 1
    if mode == "save":
        if len(bath_ids) == 1:
            save_graph(fig, bath_ids[0])
        else:
            condition = df['run_id'] == bath_ids[0]
            bath = df[condition]['bath_id'].unique()[0]
            save_graph(fig, bath)
        plt.close(fig)
    else:
        print("[Normal Mode]") 
    return fig

def save_graph(fig, name):

    filename = f"monitoring_{name}.png"
    filepath = f"./{OUTPUT_DIR_NAME}/{filename}"
    fig.savefig(filepath)
    print(f'{GREEN}[status]{RESET} {filename} was saved successfully at {filepath}')
    return 


def save_all_graph(df, params):
    create_output_dir()
    print(f"{GREEN}[Save Mode]{RESET}")
    runs_id = df['run_id'].unique()
    for id in tqdm(runs_id, unit='File', colour='green', smoothing=1):
        plot_all_params(df, [id], params, SAVE_MODE)
    print(f"{GREEN}[Save Completed >_<]{RESET}")
    return

def save_bath_graph(df, bath_ids, params):
    create_output_dir()
    print(f"{GREEN}[Save Mode]{RESET}")
    for bath in tqdm(bath_ids, unit='File', colour='green', smoothing=1):
        condition = (df['bath_id'] == bath)
        single_bath = list(df[condition]['run_id'].unique())
        plot_all_params(df, single_bath, params, SAVE_MODE)
    print(f"{GREEN}[Save Completed >_<]{RESET}")
    return


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

def plot_single_bath(df, bath_ids, param, y_pos: list[int] = [], mode=None) -> None:

    plot_data = df[df['run_id'].isin(bath_ids)]
    g = sns.relplot(
        data=plot_data, kind='line',
        x='time_total', y=param,
        hue='run_id', aspect=1.5
    )
    g.set_axis_labels("Plating Time (s)", param)
    if y_pos:
        g.ax.set_ylim(y_pos[0], y_pos[1])
    g.figure.suptitle(f'{param} vs. Plating Time')
    g.figure.subplots_adjust(top=0.9) # Adjust the top of the plot to make space for the title
    return None


def open_excel(filepath):
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print("File not found")
        return None
    
    return df


def clean_data(df):
    df_filtered = df[(df['Voltage'] != '-') & (df['Current'] != '-')]
    return df_filtered


