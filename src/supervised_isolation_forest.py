import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest 


ANOMALY_LABEL_COL = 'Anomaly C' # The new column containing 0 (Normal) or 1 (Anomaly)
BINARY_LABEL_COL = 'True_Anomaly_C'

def convert_labels_to_binary(series):
    """
    Converts multi-class anomaly labels (e.g., 'N', 'C1', 'C2') into a 
    binary numeric format (0 for Normal, 1 for Anomaly).
    """
    # Define the mapping: 'N' -> 0, anything else -> 1
    # We assume 'N' is the only normal class.
    return (series != 'N').astype(int)

class TimeSeriesFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A custom Scikit-learn transformer to generate Lag Features and Rolling Statistics
    on time series data, grouped by a unique experiment ID.
    
    The 'retained_index_' attribute stores the original indices of rows that 
    were kept after dropping NaNs, allowing scores to be mapped back correctly.
    """
    def __init__(self, sensor_cols, lag_steps, rolling_window, rolling_min_periods=1):
        self.sensor_cols = sensor_cols
        self.lag_steps = lag_steps
        self.rolling_window = rolling_window
        self.rolling_min_periods = rolling_min_periods
        # IMPORTANT: 'Timestamp', 'Time_Index', and 'Experiment_ID' are dropped from X
        # We must also drop the ANOMALY_LABEL_COL from the feature matrix X
        self.group_col = 'run_id'
        self.drop_cols = ['timestamp', 'index', self.group_col, ANOMALY_LABEL_COL] 
        self.retained_index_ = None 

    def fit(self, X, y=None):
        return self

    def _generate_lag_features(self, df):
        """Generates lag features, ensuring shift operation is grouped by Experiment_ID."""
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
            
            df[new_std_name] = df[new_std_name].fillna(0)
        
        return df

    def transform(self, X):
        df_working = X.copy()
        
        df_working = self._generate_lag_features(df_working)
        df_working = self._generate_rolling_statistics(df_working)
        
        # 3. Handle Missing Values (NaN)
        # Keep the index of the retained rows
        index_retained = df_working.dropna().index
        df_working.dropna(inplace=True)
        
        # 4. Drop metadata columns (including the True_Anomaly_Label column)
        X_final = df_working.drop(columns=self.drop_cols, errors='ignore')
        
        self.retained_index_ = index_retained
        
        # The output must be a clean NumPy array for the next step (StandardScaler)
        return X_final.values



def create_pipeline():
	# --- Configuration ---
	ROLLING_WINDOW = 1
	SENSOR_FEATURES = ["conductivity", "temperature", "pH",	"voltage"]
	LAG_STEPS = [1, 2 ,5, 10]

	feature_generator = TimeSeriesFeatureGenerator(
		sensor_cols=SENSOR_FEATURES,
		lag_steps=LAG_STEPS,
		rolling_window=ROLLING_WINDOW
	)

	scaler = StandardScaler()

	# B. Define the Model
	iso_forest_model = IsolationForest(
		n_estimators=100, 
		contamination='auto', 
		random_state=42, 
		n_jobs=-1
	)
	
	anomaly_pipeline = Pipeline([
		('feature_generator', feature_generator),
		('scaler', scaler),
		('model', iso_forest_model)
	])

	return anomaly_pipeline