import os

from src.color import *


def save_df_to_csv(df, dir, file_name):
	
	if os.path.exists(dir):
		# print(f"{OK} Directory is already exist")
		pass
	else:
		print(f"{INFO} create {dir} directory.")
		os.makedirs(dir)
	file_path = os.path.join(dir, file_name)
	df.to_csv(file_path, index=False)
	print(f"{OK} Data was saved sucessfully at {file_path}")