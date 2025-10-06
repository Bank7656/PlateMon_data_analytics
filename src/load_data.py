import gspread as gc
import sys
from gspread.exceptions import APIError, WorksheetNotFound
from gspread_dataframe import set_with_dataframe
import pandas as pd

googleclient = gc.service_account()

FILTERED_COL = [
	"run_id",
	"time_total",
	"conductivity",
	"temperature",
	"pH",
	"voltage",
	"current",
	"amp_hour",
	"bath_id"
]

FILTERED_INTERNAL_COL = [
	"run_id",
	"mass_SLS",
	"mass_NISO4",
	"mass_NICL2",
	"time_total",
	"conductivity",
	"temperature",
	"pH",
	"voltage",
	"current",
	"amp_hour",
	# "deposition_rate",
	"bath_id"
]

CONVERT_COL = {
	"run_id" : object,
	"time_total" : int,
	"conductivity" : float,
	"temperature" : float,
	"pH" : float,
	"voltage" : float,
	"current" : float,
	"amp_hour" : float,
	# "deposition_rate": float,
	"bath_id" : object
}

CONVERT_INTERNAL_COL = {
	"mass_SLS" : float,
	"mass_NISO4" : float,
	"mass_NICL2" : float
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



def clean_sheet(df) -> pd.DataFrame:

	def convert_type_sheet(df):
		df = df.astype(CONVERT_COL)
		try:
			return df.astype(CONVERT_INTERNAL_COL)
		except Exception as e:
			return df

	def filter_sheet(df): 
		df = df[(df['voltage'] != '-') & (df['current'] != '-')]
		try:
			return df[FILTERED_INTERNAL_COL]
		except Exception as e:
			return df[FILTERED_COL]
	
	df = filter_sheet(df)
	df = convert_type_sheet(df)
	return df



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