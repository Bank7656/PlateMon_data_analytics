import pandas as pd

from src.utils import save_df_to_csv
from src.config import EXTRACT_FILE, LABEL_TRANSFORM_FILE, NORMAL_RUN, ANOMALY_RUN, PARAM_LIST, ANOMALY_LIST
from src.color import *



def transform():
    print("  Transform State  ".center(80, "="))
    extract_file_path = f"./data/extract/{EXTRACT_FILE}"
    df = pd.read_csv(extract_file_path, index_col=False)
    print(f"{OK} Extracted data was load successfully")
    print(f"{INFO} Transform anomaly label")
    label_df = get_label_run(df)
    print(f"{INFO} Get normal run data")
    normal_df = get_normal_run(label_df)
    print(f"{INFO} Get Anomaly run data")
    anomaly_df = get_anomaly_run(label_df)
    print(f"{INFO} Get pivoted normal run data")
    get_pivot_df(normal_df, "normal")
    get_pivot_df(anomaly_df, "anomaly")
    return 

def get_pivot_df(df, state):
    for parameter, anomaly in zip(PARAM_LIST, ANOMALY_LIST):
        melted_df = df.melt(
            id_vars=['run_id', 'index'], 
            value_vars=[parameter, anomaly],
            var_name='metric',
            value_name='value'
        )
        pivot_df = melted_df.pivot_table(
            index=['run_id', 'metric'],
            columns='index', 
            values='value'
        )
        pivot_df = pivot_df.sort_values(
            by=['run_id', 'metric'],
            ascending=[True, False]
        )
        pivot_df = pivot_df.reset_index()
        pivot_df.columns.name = None

        value_cols = [col for col in pivot_df.columns if col not in ['run_id', 'metric']]
        pivot_df[value_cols] = pivot_df[value_cols].apply(pd.to_numeric, errors='coerce')

        is_param = (pivot_df['metric'] == parameter)
        param_means = pivot_df.loc[is_param, value_cols].mean()
        pivot_df.loc[is_param, value_cols] = pivot_df.loc[is_param, value_cols].fillna(param_means)
        save_df_to_csv(pivot_df, "./data/transform", f"pivot_{state}_{parameter}.csv")

def get_anomaly_run(df):
    anomaly_df = df[df['run_id'].isin(ANOMALY_RUN)].copy()
    save_df_to_csv(anomaly_df, "./data/transform", f"anomaly_run.csv")
    for parameter, anomaly in zip(PARAM_LIST, ANOMALY_LIST):
        anomaly_single_param_df = anomaly_df[['index', 'run_id', parameter, anomaly, 'bath_id']].reset_index(drop=True)
        save_df_to_csv(anomaly_single_param_df, "./data/transform", f"anomaly_{parameter}.csv")
    return anomaly_df

def get_normal_run(df):
    normal_df = df[df['run_id'].isin(NORMAL_RUN)].copy()
    save_df_to_csv(normal_df, "./data/transform", f"normal_run.csv")
    for parameter, anomaly in zip(PARAM_LIST, ANOMALY_LIST):
        normal_single_param_df = normal_df[['index', 'run_id', parameter, anomaly, 'bath_id']].reset_index(drop=True)
        save_df_to_csv(normal_single_param_df, "./data/transform", f"normal_{parameter}.csv")
    return normal_df

def get_label_run(df):
    label_df = df[
        (df['Anomaly C'] != "Unknown") &
        (df['Anomaly P'] != "Unknown") &
        (df['Anomaly T'] != "Unknown") &
        (df['Anomaly V'] != "Unknown")
    ].copy()
    label_df.loc[:, "Anomaly C"] = (label_df["Anomaly C"] != 'N').astype(int)
    label_df.loc[:, "Anomaly P"] = (label_df["Anomaly P"] != 'N').astype(int)
    label_df.loc[:, "Anomaly T"] = (label_df["Anomaly T"] != 'N').astype(int)
    label_df.loc[:, "Anomaly V"] = (label_df["Anomaly V"] != 'N').astype(int)

    label_df.loc[:, 'index'] = label_df.groupby('run_id').cumcount() + 1
    time_index_series = label_df.pop('index')
    label_df.insert(0, 'index', time_index_series)
    save_df_to_csv(label_df, "./data/transform", LABEL_TRANSFORM_FILE)
    return label_df