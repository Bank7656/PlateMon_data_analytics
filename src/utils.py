import sys
import os
import tomllib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
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

def load_config(file_path):
    """Loads a TOML configuration file."""
    with open(file_path, 'rb') as f: # Open in binary read mode for tomllib
        config = tomllib.load(f)
    return config