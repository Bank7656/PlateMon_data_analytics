import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

sns.set_context('notebook')
sns.set_theme(style="darkgrid", 
                rc={"figure.dpi":300, 'savefig.dpi':300, 'lines.linewidth':1.5})

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

def plot_all_params(df, bath_ids, parameters, y_pos: list[list[int]] = []) -> None:

    # 1. Create a figure with a subplot for each group in the list
    num_groups = len(parameters)
    row = 2
    col = int((num_groups + 1) / 2)
    fig, axes = plt.subplots(row, col, figsize=(5 * num_groups, 10), squeeze=False)
    plt.suptitle("Parameters Monitoring Data", fontsize=24)
    plt.subplots_adjust(hspace=0.35)
    count = 0
    plot_data = df[df['run_id'].isin(bath_ids)]
    for i in range(row):
        for j in range(col):
            if (i == row - 1 and j == col - 1):
                handles, labels = axes[0][j].get_legend_handles_labels()
                legend_bbox = axes[i][j].get_position()
                fig.legend(handles, labels,
                        loc='center', # Place the legend centered within its "subplot" area
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
                x='time_total', y=parameters[count],
                hue='run_id', ax=axes[i][j]
            )
            ax.set_xlabel('Plating Time (sec)', fontsize=16)
            ax.set_ylabel(parameters[count], fontsize=16)
            if y_pos:
                ax.set_ylim(y_pos[count][0], y_pos[count][1])
            # 5. Set the title for each individual subplot
            axes[i][j].set_title(f'{parameters[count]} vs Plating time', fontsize=20)
            axes[i][j].legend_.remove() # Ensure each subplot has its own legend  
            count += 1         
    return None


def plot_single_bath(df, bath_ids, param, y_pos: list[int] = []) -> None:

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


