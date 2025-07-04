import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns



def plot_bath(df, bath_ids, param):
    sns.set_context('notebook')
    sns.set_theme(style="darkgrid", 
                  rc={"figure.dpi":300, 'savefig.dpi':300, 'lines.linewidth':1.5})
    
    # 1. Create a figure with a subplot for each group in the list
    num_groups = len(bath_ids)
    fig, axes = plt.subplots(1, num_groups, figsize=(6 * num_groups, 5), squeeze=False)
    axes = axes.flatten()  # Ensure axes is always a 1D array for easy looping

    for i, bath in enumerate(bath_ids):
        plot_data = df[df['Run_id'].isin(bath)]

        # If no data is found for this group, skip to avoid errors
        if plot_data.empty:
            axes[i].set_title(f'No data for: {", ".join(bath)}')
            continue

        sns.lineplot(
            data=plot_data,
            x='Cumulative_Time', y=param,
            hue='Run_id', ax=axes[i]
        )

        # 5. Set the title for each individual subplot
        axes[i].set_title(f'Runs: {", ".join(map(str, bath))}')
        axes[i].legend() # Ensure each subplot has its own legend

    plt.show()

    return None



def plot_single_bath(df, bath_ids, param):
    sns.set_context('notebook')
    sns.set_theme(style="darkgrid", 
                  rc={"figure.dpi":300, 'savefig.dpi':300, 'lines.linewidth':1.5})

    plot_data = df[df['Run_id'].isin(bath_ids)]
    g = sns.relplot(
        data=plot_data, kind='line',
        x='Cumulative_Time', y=param,
        hue='Run_id', aspect=1.5
    )
    # sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0), ncol=len(bath_ids), frameon=True)
    g.set_axis_labels("Cumulative Time (s)", param)
    # The modern and more robust way to set a main title
    g.figure.suptitle(f'{param} vs. Cumulative Time')
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


