#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from src.utils import save_df_to_csv, load_config
from src.color import *


def transform():
	try:
		config = load_config("config.toml")
	except:
		print(f"{KO} Can't import config.toml")
	global_config(config)
	print("  Transform State  ".center(80, "="))
	extract_file_path = f"./data/extract/{EXTRACT_FILE}"
	df = pd.read_csv(extract_file_path, index_col=False)
	print(f"{OK} Extracted data was load successfully")
	print(f"{INFO} Filter last 500 datapoint for every run")
	filtered_df = filter_time(df)
	print(f"{INFO} Transform anomaly label")
	label_df, voltage_df = get_label_run(filtered_df)
	print(f"{INFO} Get normal run data")
	get_automatic_anomaly_run(label_df, voltage_df)
	print(f"{INFO} Get run that automatic choosing from lab label")
	# normal_df = get_normal_run(label_df)
	# print(f"{INFO} Get Anomaly run data")
	# anomaly_df = get_anomaly_run(label_df)
	# print(f"{INFO} Get pivoted normal run data")
	# get_pivot_df(normal_df, "normal")
	# get_pivot_df(anomaly_df, "anomaly")
	return 

def filter_time(df):
	filtered_df = df.groupby('run_id').tail(500).reset_index(drop=True).copy()
	filtered_df = filtered_df.sort_values(by=['run_id', 'timestamp'])
	filtered_df['index'] = filtered_df.groupby('run_id').cumcount()
	save_df_to_csv(filtered_df, "./data/transform", f"filtered_run.csv")
	return filtered_df

def get_pivot_df(df, state, filename, mode="multi"):
	full_df_lst = []
	for parameter, anomaly in zip(PARAM_LIST, ANOMALY_LIST):
		pivot_df = get_single_pivot_df(df, parameter, anomaly)
		full_df_lst.append(pivot_df)
		if (mode == "multi"):
			save_df_to_csv(pivot_df, "./data/transform", f"pivot_{state}_{parameter}.csv")
	full_df = pd.concat(full_df_lst).sort_values(by=['run_id', 'metric'], ascending=[True, False])
	save_df_to_csv(full_df, "./data/transform", f"{filename}_{state}.csv")

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

def get_automatic_anomaly_run(df, voltage_df):
	normal_dfs = {}  # will store the result for each anomaly
	anomaly_dfs = {}  # will store the result for each anomaly

	run_is_all_zero = (
		df.groupby('run_id')[ANOMALY_LIST]
		.apply(lambda g: (g == 0).all().all())
	)

	# Run IDs where ALL anomaly columns are zero
	all_zero_runs = run_is_all_zero[run_is_all_zero].index
	# 1. DF with runs where all anomalies are zero
	all_normal_df = df[df['run_id'].isin(all_zero_runs)].copy()

	# 2. DF with all other runs
	others_df = voltage_df[~voltage_df['run_id'].isin(all_zero_runs)].copy()
	save_df_to_csv(all_normal_df, "./data/transform", f"normal_notpivot_auto.csv")
	save_df_to_csv(others_df, "./data/transform", f"anomaly_notpivot_auto.csv")
	get_pivot_df(all_normal_df, 'normal', 'multi_auto', mode="single")
	get_pivot_df(others_df, 'anomaly', 'multi_auto', mode="single")
	# save_df_to_csv(others_df, "./data/transform", f"anomaly_all_auto.csv")
	for parameter, anomaly in zip(PARAM_LIST, ANOMALY_LIST):
		# Find run_ids where this specific anomaly column is ALL zeros
		if parameter == 'voltage':
			normal_runs = (
				voltage_df.groupby('run_id')[anomaly]
				.apply(lambda x: (x == 0).all())
			)
			anomaly_runs = (
				voltage_df.groupby('run_id')[anomaly]
				.apply(lambda x: (x == 1).any())
			)
			normal_runs = normal_runs[normal_runs].index  # only True ones
			anomaly_runs = anomaly_runs[anomaly_runs].index
			# Store filtered dataframe for this anomaly
			normal_dfs[anomaly] = voltage_df[voltage_df['run_id'].isin(normal_runs)].copy()
			anomaly_dfs[anomaly] = voltage_df[voltage_df['run_id'].isin(anomaly_runs)].copy()
		else:
			normal_runs = (
				df.groupby('run_id')[anomaly]
				.apply(lambda x: (x == 0).all())
			)
			anomaly_runs = (
				df.groupby('run_id')[anomaly]
				.apply(lambda x: (x == 1).any())
			)
			normal_runs = normal_runs[normal_runs].index  # only True ones
			anomaly_runs = anomaly_runs[anomaly_runs].index
			# Store filtered dataframe for this anomaly
			normal_dfs[anomaly] = df[df['run_id'].isin(normal_runs)].copy()
			anomaly_dfs[anomaly] = df[df['run_id'].isin(anomaly_runs)].copy()

		normal_single_param_df = normal_dfs[anomaly][['index', 'run_id', parameter, anomaly, 'bath_id']].reset_index(drop=True)
		anomaly_single_param_df = anomaly_dfs[anomaly][['index', 'run_id', parameter, anomaly, 'bath_id']].reset_index(drop=True)
		save_df_to_csv(normal_single_param_df, "./data/transform", f"normal_{parameter}_auto.csv")
		save_df_to_csv(anomaly_single_param_df, "./data/transform", f"anomaly_{parameter}_auto.csv")
		normal_pivot_df = get_single_pivot_df(normal_single_param_df, parameter, anomaly)
		anomaly_pivot_df = get_single_pivot_df(anomaly_single_param_df, parameter, anomaly)
		save_df_to_csv(normal_pivot_df, "./data/transform", f"pivot_normal_{parameter}_auto.csv")
		save_df_to_csv(anomaly_pivot_df, "./data/transform", f"pivot_anomaly_{parameter}_auto.csv")
	return


def get_single_pivot_df(df, parameter, anomaly):
	melted_df = df.melt(
		id_vars=['run_id', 'index'], 
		value_vars=[parameter, anomaly],
		var_name='metric',
		value_name='value'
	)
	melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')
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
	# pivot_df[value_cols] = pivot_df[value_cols].apply(pd.to_numeric, errors='coerce')

	# Select the rows you want to modify
	is_param = (pivot_df['metric'] == parameter)

	# Apply the row-wise fillna operation
	# .apply(axis=1) iterates over each row in the selected DataFrame slice
	filled_values = pivot_df.loc[is_param, value_cols].apply(
		lambda row: row.fillna(row.mean()), 
		axis=1
	)

	# Assign the new, filled values back to the original DataFrame
	pivot_df.loc[is_param, value_cols] = filled_values
	return pivot_df

def get_label_run(df):
	label_df = df.copy()
	# labeled_run_ids = (
	# 	label_df[~label_df[ANOMALY_LIST].eq('Unknown').all(axis=1)]
	# 	['run_id']
	# 	.unique()
	# )
	is_fully_unknown_row = label_df[['Anomaly C', 'Anomaly T', 'Anomaly P']].eq('Unknown').any(axis=1)


	# 2. Find the run_ids that contain at least one of these bad rows
	dirty_run_ids = label_df.loc[is_fully_unknown_row, 'run_id'].unique()
	# 3. Get the list of ALL unique run_ids
	all_run_ids = label_df['run_id'].unique()

	# 4. Subtract the "dirty" runs from the "all" runs
	# Result: IDs where every single row has at least one valid label


	clean_run_ids = np.setdiff1d(all_run_ids, dirty_run_ids)

	# For volage only
	voltage_row = label_df[['Anomaly C', 'Anomaly T', 'Anomaly P', 'Anomaly V']].eq('Unknown').any(axis=1)
	voltage_dirty_run_ids= label_df.loc[voltage_row, 'run_id'].unique()
	voltage_run_ids = np.setdiff1d(all_run_ids, voltage_dirty_run_ids)
	voltage_df = label_df.copy()

	voltage_df = voltage_df[voltage_df['run_id'].isin(voltage_run_ids)]
	voltage_df.loc[:, "Anomaly C"] = (~voltage_df["Anomaly C"].str.startswith("N")).astype(int)
	voltage_df.loc[:, "Anomaly P"] = (~voltage_df["Anomaly P"].str.startswith("N")).astype(int)
	voltage_df.loc[:, "Anomaly T"] = (~voltage_df["Anomaly T"].str.startswith("N")).astype(int)
	voltage_df.loc[:, "Anomaly V"] = (~voltage_df["Anomaly V"].str.startswith("N")).astype(int)
	voltage_df.loc[:, 'index'] = voltage_df.groupby('run_id').cumcount() + 1
	time_index_series = voltage_df.pop('index')
	voltage_df.insert(0, 'index', time_index_series)

	# Other parameter

	label_df = label_df[label_df['run_id'].isin(clean_run_ids)]
	label_df.loc[:, "Anomaly C"] = (~label_df["Anomaly C"].str.startswith("N")).astype(int)
	label_df.loc[:, "Anomaly P"] = (~label_df["Anomaly P"].str.startswith("N")).astype(int)
	label_df.loc[:, "Anomaly T"] = (~label_df["Anomaly T"].str.startswith("N")).astype(int)
	label_df.loc[:, "Anomaly V"] = (~label_df["Anomaly V"].str.startswith("N")).astype(int)
	label_df.loc[:, 'index'] = label_df.groupby('run_id').cumcount() + 1
	time_index_series = label_df.pop('index')
	label_df.insert(0, 'index', time_index_series)
	save_df_to_csv(label_df, "./data/transform", LABEL_TRANSFORM_FILE)
	return label_df, voltage_df

def global_config(cf):
	global NORMAL_RUN
	global ANOMALY_RUN
	global PARAM_LIST
	global ANOMALY_LIST
	global EXTRACT_FILE
	global LABEL_TRANSFORM_FILE
	NORMAL_RUN = cf['multivariate']['NORMAL_RUN']
	ANOMALY_RUN = cf['multivariate']['ANOMALY_RUN']
	PARAM_LIST = cf['list']['PARAM_LIST']
	ANOMALY_LIST = cf['list']['ANOMALY_LIST']
	EXTRACT_FILE = cf['file']['EXTRACT_FILE']
	LABEL_TRANSFORM_FILE = cf['file']['LABEL_TRANSFORM_FILE']

if __name__ == "__main__":
	if os.path.basename(os.getcwd()) == 'src':
		os.chdir("..")
	print(f"Current Working Directory: {os.getcwd()}")
	RED = "\x1b[1;31m"
	BLUE = "\x1b[1;34m"
	GREEN = "\x1b[1;32m"
	RESET = "\x1b[0m"
	INFO = f"{BLUE}[INFO]{RESET}"
	OK = f"{GREEN}[OK]{RESET}"
	KO = f"{RED}[KO]{RESET}"
	transform()
	sys.exit(0)