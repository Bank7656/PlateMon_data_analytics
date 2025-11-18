import gspread as gc
import numpy as np
import pandas as pd
import sys
import os
from gspread.exceptions import APIError, WorksheetNotFound
from gspread_dataframe import set_with_dataframe

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.color import *
from src.utils import save_df_to_csv, load_config



googleclient = gc.service_account()

ANOMALY_DETECTION_PARAM = {
    'run_id' : object, 
    'timestamp': 'datetime64[ns]', 
    'time' : int, 
    'time_total' : int, 
    'area' : int , 
    'cathode' : object, 
    'anode' : object,
    'mass_SLS' : float, 
    'mass_NISO4' : float, 
    'mass_NICL2' : float, 
    'current_density' : float,
    'conductivity' : float, 
    'Anomaly C' : object, 
    'pH' : float, 
    'Anomaly P' : object, 
    'temperature' : float,
    'Anomaly T' : object, 
    'voltage' : float, 
    'Anomaly V' : object, 
    'current' : float, 
    'amp_hour' : float,
    'deposition_rate' : float, 
    'bath_id' : object
}

def extract():

    try:
        config = load_config("config.toml")
    except:
        print(f"{KO} Can't import config.toml")
    NORMAL_RUN = config['experiments']['NORMAL_RUN']
    ANOMALY_RUN = config['experiments']['ANOMALY_RUN']
    PARAM_LIST = config['list']['PARAM_LIST']
    ANOMALY_LIST = config['list']['ANOMALY_LIST']
    EXTRACT_FILE = config['file']['EXTRACT_FILE']
    LABEL_TRANSFORM_FILE = config['file']['LABEL_TRANSFORM_FILE']

    print("  Extract State  ".center(80, "="))
    print(f"{INFO} Extracting data")
    jun_jul_df, sheet1 = load_sheet("Electroplate Experiments Data JUN_JUL", "vary_internal_table")
    august_df, sheet3 = load_sheet("Electroplating Experiments Data August", "Sheet1")
    september_df, sheet4 = load_sheet("Electroplating Experiments Data Sep-Oct", "Sheet1")
    november_df, sheet1 = load_sheet("Electroplating Experiments Data November", "Sheet1")
    print(f"{OK} All data was loaded successfully")

    print(f"{INFO} Filtering columns")
    jun_jul_df = clean_sheet_with_label("Electroplate Experiments Data JUN_JUL", jun_jul_df)
    august_df = clean_sheet_with_label("Electroplating Experiments Data August", august_df)
    september_df = clean_sheet_with_label("Electroplating Experiments Data Sep-Oct", september_df)
    november_df = clean_sheet_with_label("Electroplating Experiments Data November", november_df)
    print(f"{OK} Filtering complete")

    print(f"{INFO} Merging data")
    df = pd.concat([jun_jul_df, august_df, september_df, november_df])
    print(f"{OK} data were merged sucessfully")

    df['Anomaly C'] = df['Anomaly C'].fillna("Unknown").replace("", "Unknown").astype(object)
    df['Anomaly P'] = df['Anomaly P'].fillna("Unknown").replace("", "Unknown").astype(object)
    df['Anomaly T'] = df['Anomaly T'].fillna("Unknown").replace("", "Unknown").astype(object)
    df['Anomaly V'] = df['Anomaly V'].fillna("Unknown").replace("", "Unknown").astype(object)

    save_df_to_csv(df, "./data/extract", EXTRACT_FILE)
    print(f"{OK} Finishing Extract state")
    return
	

def load_sheet(file_name, prompt):

	def open_sheet(gc):
		try:
			sheet = gc.open(file_name)
			return (sheet)
		except:
			print(f"{KO} Sheet does not exists")
			sys.exit(1)

	sheet = open_sheet(googleclient)
	worksheet = sheet.worksheet(prompt)
	data = worksheet.get_all_values()
	if not data:
		print(f"{KO} No data found in the sheet.")
		sys.exit(1)

	headers = data.pop(0)
	df = pd.DataFrame(data, columns=headers)
	print(f"{OK} {file_name} was load successfully")
	return df, sheet


def clean_sheet_with_label(name, df) -> pd.DataFrame:
	filtered_df = pd.DataFrame()
	clean_old_df = df[(df['voltage'] != '-') & (df['current'] != '-')].copy()
	for col, datatype in ANOMALY_DETECTION_PARAM.items():
		if col not in clean_old_df.columns:
			filtered_df[col] = "Unknown"
		else:
			try:
				filtered_df[col] = clean_old_df[col].astype(datatype)
			except Exception as e:
				filtered_df[col] = clean_old_df[col]
	print(f"{OK} {name} was filtered successfully")
	return (filtered_df)


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
    extract()
    sys.exit(0)