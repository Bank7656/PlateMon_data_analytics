import gspread as gc
import numpy as np
import pandas as pd
import sys
from gspread.exceptions import APIError, WorksheetNotFound
from gspread_dataframe import set_with_dataframe

from src.config import ANOMALY_DETECTION_PARAM, EXTRACT_FILE
from src.color import *
from src.utils import save_df_to_csv

googleclient = gc.service_account()

def extract():
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
    november_df = clean_sheet_with_label("Electroplating Experiments Data Sep-Oct", november_df)
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