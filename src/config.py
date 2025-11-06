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

EXTRACT_FILE = "electroplating_extracted.csv"
LABEL_TRANSFORM_FILE = "transform_with_label.csv"
NORMAL_RUN = ['JUL_22_4', 'JUL_22_5', 'JUL_22_7']
ANOMALY_RUN = ['JUL_22_3', 'JUL_22_9', 'JUL_23_1', 'JUL_23_2', 'JUL_23_3', 'JUL_23_4', 'JUL_23_5', 'JUL_24_3']
PARAM_LIST = ['voltage', 'conductivity', 'pH', 'temperature']
ANOMALY_LIST = ['Anomaly V', 'Anomaly C', 'Anomaly P', 'Anomaly T']