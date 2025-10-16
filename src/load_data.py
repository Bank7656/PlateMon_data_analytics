import gspread as gc
import sys
from gspread.exceptions import APIError, WorksheetNotFound
from gspread_dataframe import set_with_dataframe
import pandas as pd

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



def load_sheet(file_name, prompt):

	def open_sheet(gc):
		try:
			sheet = gc.open(file_name)
			return (sheet)
		except:
			print("Sheet does not exists")
			sys.exit(0)

	sheet = open_sheet(googleclient)
	worksheet = sheet.worksheet(prompt)
	data = worksheet.get_all_values()
	headers = data.pop(0)
	df = pd.DataFrame(data, columns=headers)
	return df, sheet


def clean_sheet_with_label(df) -> pd.DataFrame:
	filtered_df = pd.DataFrame()
	clean_old_df = df[(df['voltage'] != '-') & (df['current'] != '-')].copy()
	for col, datatype in ANOMALY_DETECTION_PARAM.items():
		if col not in clean_old_df.columns:
			filtered_df[col] = None
		else:
			filtered_df[col] = clean_old_df[col].astype(datatype)
	return (filtered_df)



def seperate_table(sheet, df):
	print(f"You can view the sheet here: {sheet.url}")
	run_id = df.run_id.unique()
	worksheet_lst = sheet.worksheets()
	test = "hello"
	print("Clean sheet")
	for id in run_id:
		try:
			worksheet = sheet.worksheet(id)
			sheet.del_worksheet(worksheet)
			print(f"[OK] Worksheet {id} was deleted.")
		except WorksheetNotFound:
			print(f"[OK] Worksheet {id} does not exist.")
	print("All worksheet are cleaned")
	for id in run_id:
		single_run_df = df[df['run_id'] == id]
		worksheet = None
		try:
			worksheet = sheet.add_worksheet(title=id, 
								   rows=single_run_df.shape[0], 
								   cols=single_run_df.shape[1]
								)
			print(f"[OK] Worksheet {id} was successfully created.")
		except APIError:
			print(f"[OK] Sheet {id} already exists")
			worksheet = sheet.worksheet(id)
		finally:
			if worksheet is not None:
				set_with_dataframe(worksheet, single_run_df)
				print("[OK] DataFrame successfully written to Google Sheet.")
			else:
				print(f"[FAIL] Worksheet {id} does not exist.")

	return 