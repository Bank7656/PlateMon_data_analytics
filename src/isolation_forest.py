import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


np.random.seed(42)

# ---------------------------------------------
# 1. Custom Transformer: Select Specific Columns
# ---------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
	def __init__(self, sensor_cols, lag_steps, rolling_window, rolling_min_periods=1):
		self.sensor_cols = sensor_cols
		self.lag_steps = lag_steps
		self.rolling_window = rolling_window
		self.rolling_min_periods = rolling_min_periods
		self.group_col = 'run_id'
		self.drop_cols = ['timestamp', 'index', self.group_col]
		self.retained_index_ = None
	
	def fit(self, X, y=None):
		return self
	
	def _generate_lag_features(self, df):
		for col in self.sensor_cols:
			for lag in self.lag_steps:
				new_col_name = f'{col}_Lag_{lag}'
				df[new_col_name] = df.groupby(self.group_col)[col].shift(lag)
		return df

	def _generate_rolling_statistics(self, df):
		"""Generates rolling mean and standard deviation, grouped by Experiment_ID."""
		for col in self.sensor_cols:
			# 1. Rolling Mean
			new_mean_name = f'{col}_RollMean_{self.rolling_window}'
			df[new_mean_name] = df.groupby(self.group_col)[col].rolling(
				window=self.rolling_window, 
				min_periods=self.rolling_min_periods
			).mean().reset_index(level=0, drop=True)

			# 2. Rolling Standard Deviation
			new_std_name = f'{col}_RollStd_{self.rolling_window}'
			df[new_std_name] = df.groupby(self.group_col)[col].rolling(
				window=self.rolling_window, 
				min_periods=self.rolling_min_periods
			).std().reset_index(level=0, drop=True)
			
			# Fill NaN from rolling std (which occurs only when min_periods=1) with 0
			df[new_std_name] = df[new_std_name].fillna(0)
		
		return df

	def transform(self, X):
		df_working = X.copy()

		# df_working = self._generate_lag_features(df_working)
		# df_working = self._generate_rolling_statistics(df_working)

		index_retained = df_working.dropna().index
		df_working.dropna(inplace=True)

		# 4. Drop metadata columns
		X_final = df_working.drop(columns=self.drop_cols, errors='ignore')

		# Store the retained index as an attribute for external use (like mapping scores back)
		self.retained_index_ = index_retained

		return X_final.reset_index(drop=True)

def score_and_map_anomalies(pipeline, X_raw):
    """
    Applies feature engineering, generates anomaly scores, and maps them back
    to the original DataFrame, aligned by the retained index.
    """
    
    # --- Step 1: Pre-process the data (Feature Generation & Index Tracking) ---
    feature_generator = pipeline.named_steps['feature_generator']
    
    # Apply FeatureSelector to the raw data
    X_features = feature_generator.transform(X_raw.copy()) # Use transform
    
    # Get the index of rows that were NOT dropped by the FeatureSelector
    retained_index = feature_generator.retained_index_

    # --- Step 2: Scale the Features ---
    scaler = pipeline.named_steps['scaler']
    X_scaled = scaler.transform(X_features)

    # --- Step 3: Get Anomaly Scores ---
    iso_forest_model = pipeline.named_steps['model']
    
    # Calculate the raw anomaly scores (decision_function)
    anomaly_scores = iso_forest_model.decision_function(X_scaled)

    # Calculate the binary prediction (-1 for Anomaly, 1 for Normal)
    anomaly_labels = iso_forest_model.predict(X_scaled)
    
    # --- Step 4: Map Scores Back to Original Data ---
    
    # Create a result DataFrame with scores and labels
    results_df = pd.DataFrame({
        'anomaly_score': anomaly_scores,
        'is_anomaly': anomaly_labels
    })

    # The scores/labels correspond to the retained rows. Set their index.
    results_df.index = retained_index 
    
    # Join the results back to the original raw data
    X_scored = X_raw.join(results_df, how='left')
    
    # Fill rows that were dropped by FeatureSelector with NaN for scores/labels
    # (These are typically the first few rows of each 'run_id')
    
    return X_scored

def convert_labels_to_binary(series):
    """
    Converts multi-class anomaly labels (e.g., 'N', 'C1', 'C2') into a 
    binary numeric format (0 for Normal, 1 for Anomaly).
    """
    # Define the mapping: 'N' -> 0, anything else -> 1
    # We assume 'N' is the only normal class.
    return (series != 'N').astype(int)

def create_pipeline(param=["conductivity", "temperature", "pH",	"voltage"]):
	# --- Configuration ---
	ROLLING_WINDOW = 60
	CONTAMINATION_RATE = 0.1
	# SENSOR_FEATURES = ["conductivity", "temperature", "pH",	"voltage"]
	LAG_STEPS = [1, 2 ,5] 

	feature_generator = FeatureSelector(
		sensor_cols=param,
		lag_steps=LAG_STEPS,
		rolling_window=ROLLING_WINDOW
	)

	scaler = StandardScaler()

	# B. Define the Model
	iso_forest_model = IsolationForest(
		n_estimators=100, 
		contamination=CONTAMINATION_RATE, 
		random_state=42, 
		n_jobs=-1
	)
	
	anomaly_pipeline = Pipeline([
		('feature_generator', feature_generator),
		('scaler', scaler),
		('model', iso_forest_model)
	])

	return anomaly_pipeline

def tuning_contamination_rate(pipeline, contamination_rate, df_data):
    """Helper function to run the full analysis for a given contamination rate."""
    
    # 1. Set the new contamination rate in the pipeline
    pipeline.set_params(model__contamination=contamination_rate)
    
    # 2. Fit and Predict
    pipeline.fit(df_data)
    predictions = pipeline.predict(df_data)
    
    # 3. Score Alignment (Mapping results back to original data)
    retained_index = pipeline.named_steps['feature_generator'].retained_index_
    results = pd.Series(1, index=df_data.index)
    results.loc[retained_index] = predictions
    
    # 4. Report
    anomaly_count = (results == -1).sum()
    
    return anomaly_count, results


def plot_sub_dataset(df, param):
	fig, axes = plt.subplots(4, 1, figsize=(10, 12))
	for i, var in enumerate(param):
		for run_id, group in df.groupby('run_id'):
			axes[i].plot(group['index'], group[var], label=f'{run_id}')
		axes[i].grid(True)
		axes[i].set_ylim(param[var])
		axes[i].set_xlabel("Time")
		axes[i].set_ylabel(f"{var}")
		axes[i].set_title(f"{var} versus time")
	plt.legend()
	plt.tight_layout()
	plt.show()
# plt.scatter(df_test[df_test['is_anomaly']]['time_total'], df_test[df_test['is_anomaly']]['voltage'],
#             color='red', label='Anomalies', s=10)


def plot_anomaly_detection(X_score_df):
	# Assuming X_scored_df is already prepared as per previous steps.
	# If 'timestamp' is a column, you might want to use it instead of index for X-axis.
	# If not, X_scored_df.index is a good proxy for 'time_step'.
	X_score_df['time_step'] = X_score_df.index 

	# Define the sensor features you want to plot
	SENSOR_FEATURES = ["conductivity", "temperature", "pH", "voltage"]

	# Create a figure and a set of subplots
	# We'll arrange them in 2 rows and 2 columns
	fig, axes = plt.subplots(nrows=len(SENSOR_FEATURES)//2, ncols=2, 
							figsize=(20, 12), sharex=True) # sharex=True aligns x-axes

	# Flatten the axes array for easier iteration if it's 2D
	axes = axes.flatten()

	# Loop through each sensor and create a subplot
	for i, sensor in enumerate(SENSOR_FEATURES):
		ax = axes[i] # Get the current axis for plotting

		if (len(X_score_df['run_id'].unique()) < 20):
			sns.lineplot(x='time_step', y=sensor, 
						data=X_score_df, 
						hue='run_id',
						color='blue', alpha=0.6, ax=ax)
		else:
			sns.lineplot(x='time_step', y=sensor, 
						data=X_score_df, 
						hue='run_id', legend=False,
						color='blue', alpha=0.6, ax=ax)
		# Overlay anomaly points (is_anomaly == -1)
		sns.scatterplot(x='time_step', y=sensor, 
						data=X_score_df[X_score_df['is_anomaly'] == -1], 
						label='Anomaly', color='red', marker='X', s=100, ax=ax)

		ax.set_title(f'Time Series of {sensor}')
		ax.set_xlabel('Time Step')
		ax.set_ylabel(sensor)
		
		ax.grid(True, linestyle='--', alpha=0.7) # Add a grid for better readability

	# Adjust layout to prevent overlap and display the plot
	plt.tight_layout()
	plt.suptitle('Time Series Analysis of Sensor Data with Isolation Forest Anomalies', 
				y=1.02, fontsize=16) # Add a main title for the entire figure
	plt.show()

def plot_anomaly_detection_each_run(X_score_df):
	# Assuming X_scored_df is already prepared as per previous steps.
	# If 'timestamp' is a column, you might want to use it instead of index for X-axis.
	# If not, X_scored_df.index is a good proxy for 'time_step'.
	X_score_df['time_step'] = X_score_df.index 

	# Define the sensor features you want to plot
	SENSOR_FEATURES = ["conductivity", "temperature", "pH", "voltage"]

	# Create a figure and a set of subplots
	# We'll arrange them in 2 rows and 2 columns
	fig, axes = plt.subplots(nrows=len(SENSOR_FEATURES)//2, ncols=2, 
							figsize=(20, 12), sharex=True) # sharex=True aligns x-axes

	# Flatten the axes array for easier iteration if it's 2D
	axes = axes.flatten()

	# Loop through each sensor and create a subplot
	for i, sensor in enumerate(SENSOR_FEATURES):
		ax = axes[i] # Get the current axis for plotting

		if (len(X_score_df['run_id'].unique()) < 20):
			sns.lineplot(x='index', y=sensor, 
						data=X_score_df, 
						hue='run_id',
						color='blue', alpha=0.6, ax=ax)
		else:
			sns.lineplot(x='index', y=sensor, 
						data=X_score_df, 
						hue='run_id', legend=False,
						color='blue', alpha=0.6, ax=ax)
		# Overlay anomaly points (is_anomaly == -1)
		sns.scatterplot(x='index', y=sensor, 
						data=X_score_df[X_score_df['is_anomaly'] == -1], 
						label='Anomaly', color='red', marker='X', s=100, ax=ax)

		ax.set_title(f'Time Series of {sensor}')
		ax.set_xlabel('Time Step')
		ax.set_ylabel(sensor)
		
		ax.grid(True, linestyle='--', alpha=0.7) # Add a grid for better readability

	# Adjust layout to prevent overlap and display the plot
	plt.tight_layout()
	plt.suptitle('Time Series Analysis of Sensor Data with Isolation Forest Anomalies', 
				y=1.02, fontsize=16) # Add a main title for the entire figure
	plt.show()