import gspread as gc
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
	"bath_id" : object
}

CONVERT_INTERNAL_COL = {
	"mass_SLS" : float,
	"mass_NISO4" : float,
	"mass_NICL2" : float
}


def load_sheet(prompt) -> pd.DataFrame:

	def open_sheet(gc):
		try:
			sheet = gc.open("Electroplate Experiments Data")
			return (sheet)
		except:
			print("Sheet does not exists")
			exit(0)

	sheet = open_sheet(googleclient)
	worksheet = sheet.worksheet(prompt)
	data = worksheet.get_all_values()
	headers = data.pop(0)
	df = pd.DataFrame(data, columns=headers)
	return df



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