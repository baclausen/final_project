import os
import numpy as np
import pandas as pd
import random
from mgrs import MGRS

# --- Configuration ---
NUM_SYSTEMS = 10

# Define a range to create imbalance
MIN_PULSES = 100
MAX_PULSES = 2000

# Lat/Lon bounding box for the region:
LAT_RANGE = (24, 40)
LON_RANGE = (118, 130)

# Time range for the whole dataset
START_DATE = pd.to_datetime('2025-11-19 00:00:00')
END_DATE = pd.to_datetime('2025-12-19 23:59:59')

# Initialize the MGRS converter object
m = MGRS()

# Radar Function Definitions
EMITTER_FUNCTION_MAP = {
    1: 'Medium-Range Surveillance',
    2: 'Short-Range Fire Control/Tracking',
    3: 'Naval/Ground Surveillance',
    4: 'Target Acquisition/Tracking',
    5: 'Air Traffic Control',
    6: 'Early Warning', 
    7: 'Target Illumination', 
    8: 'Medium-Range Search/Ground Mapping', 
    9: 'Coastal Defense', 
    10: 'Long-Range Tracking/Weather', 
}

# System Modulation Parameters
SYSTEM_MODULATION_PARAMS = {
    # RF means are much closer and distributions will blend (e.g., 1 & 6, 2 & 7)
    # Amplitude means are also closer (e.g., 3 & 9, 4 & 8)
    1: {'RF_MHz': 9200.0, 'Amplitude_dB': 15.0, 'DOA_deg': 45.0, 'PRI': {'type': 'fixed', 'mean': 200.0, 'std': 5.0}, 'PW': {'type': 'fixed', 'mean': 3.5, 'std': 0.5}}, # Increased PRI/PW std
    2: {'RF_MHz': 10500.0, 'Amplitude_dB': 8.0, 'DOA_deg': 110.0, 'PRI': {'type': 'staggered', 'values': [50.0, 75.0, 100.0], 'std': 2.0}, 'PW': {'type': 'fixed', 'mean': 0.8, 'std': 0.5}}, # Increased PRI/PW std
    3: {'RF_MHz': 8100.0, 'Amplitude_dB': 12.0, 'DOA_deg': 270.0, 'PRI': {'type': 'jittered', 'min': 300.0, 'max': 500.0, 'std': 5.0}, 'PW': {'type': 'fixed', 'mean': 1.5, 'std': 0.5}}, # Increased PRI/PW std
    4: {'RF_MHz': 9800.0, 'Amplitude_dB': 18.0, 'DOA_deg': 20.0, 'PRI': {'type': 'fixed', 'mean': 120.0, 'std': 5.0}, 'PW': {'type': 'staggered', 'values': [1.0, 2.0, 3.0, 4.0], 'std': 0.5}}, # Increased PRI/PW std
    5: {'RF_MHz': 11200.0, 'Amplitude_dB': 5.0, 'DOA_deg': 315.0, 'PRI': {'type': 'jittered', 'min': 750.0, 'max': 850.0, 'std': 5.0}, 'PW': {'type': 'staggered', 'values': [0.2, 0.4], 'std': 0.5}}, # Increased PRI/PW std
    6: {'RF_MHz': 9250.0, 'Amplitude_dB': 16.0, 'DOA_deg': 10.0, 'PRI': {'type': 'staggered', 'values': [400.0, 450.0], 'std': 5.0}, 'PW': {'type': 'fixed', 'mean': 5.0, 'std': 0.5}}, # Very close to Emitter 1 RF/Amp
    7: {'RF_MHz': 10450.0, 'Amplitude_dB': 7.0, 'DOA_deg': 190.0, 'PRI': {'type': 'fixed', 'mean': 50.0, 'std': 5.0}, 'PW': {'type': 'jittered', 'min': 1.5, 'max': 2.5, 'std': 0.5}}, # Very close to Emitter 2 RF/Amp
    8: {'RF_MHz': 9750.0, 'Amplitude_dB': 17.0, 'DOA_deg': 90.0, 'PRI': {'type': 'staggered', 'values': [350.0, 400.0, 450.0], 'std': 5.0}, 'PW': {'type': 'fixed', 'mean': 2.5, 'std': 0.5}}, # Very close to Emitter 4 RF/Amp
    9: {'RF_MHz': 8050.0, 'Amplitude_dB': 11.0, 'DOA_deg': 220.0, 'PRI': {'type': 'staggered', 'values': [110.0, 130.0], 'std': 5.0}, 'PW': {'type': 'fixed', 'mean': 1.8, 'std': 0.5}}, # Very close to Emitter 3 RF/Amp
    10: {'RF_MHz': 11150.0, 'Amplitude_dB': 6.0, 'DOA_deg': 340.0, 'PRI': {'type': 'fixed', 'mean': 800.0, 'std': 5.0}, 'PW': {'type': 'jittered', 'min': 0.5, 'max': 1.5, 'std': 0.5}} # Very close to Emitter 5 RF/Amp
}

# STDs for non-modulated features (RF/Amp/DOA)
STDS_NON_MODULATED = np.array([50.0, 5.0, 15.0]) 

# Utility Functions
def generate_fixed_params(n, params):
    mean = params['mean']
    std = params.get('std', 0.1)
    return np.maximum(np.random.normal(loc=mean, scale=std, size=n), 0.01)

def generate_staggered_params(n, params):
    values = params['values']
    std = params.get('std', 0.1)
    base_values = np.array([values[i % len(values)] for i in range(n)])
    jitter = np.random.normal(loc=0.0, scale=std, size=n)
    return np.maximum(base_values + jitter, 0.01)

def generate_jittered_params(n, params):
    min_val = params['min']
    max_val = params['max']
    std = params.get('std', 0.1)
    base_values = np.random.uniform(min_val, max_val, n)
    jitter = np.random.normal(loc=0.0, scale=std, size=n)
    return np.maximum(base_values + jitter, 0.01)

MODULATION_FUNCTIONS = {
    'fixed': generate_fixed_params,
    'staggered': generate_staggered_params,
    'jittered': generate_jittered_params
}

def generate_cluster_timestamp(n, overall_start, overall_end, activity_duration_hours=1):
    total_span_seconds = (overall_end - overall_start).total_seconds()
    max_start_offset = total_span_seconds - (activity_duration_hours * 3600)
    random_offset_seconds = np.random.uniform(0, max_start_offset)
    burst_start = overall_start + pd.Timedelta(seconds=random_offset_seconds)
    burst_end = burst_start + pd.Timedelta(hours=activity_duration_hours)
    
    start_u = burst_start.value // 10**9
    end_u = burst_end.value // 10**9
    
    random_timestamps = np.random.randint(start_u, end_u, n)
    return pd.to_datetime(random_timestamps, unit='s')

def generate_single_mgrs_point(lat_range, lon_range, precision=3):
    lat = np.random.uniform(lat_range[0], lat_range[1])
    lon = np.random.uniform(lon_range[0], lon_range[1])
    full_mgrs = m.toMGRS(lat, lon)
    mgrs_6_digit = full_mgrs[:5] + full_mgrs[5:5+precision] + full_mgrs[10:10+precision]
    return mgrs_6_digit

# Core Data Generation
emitter_locations = {}
for system_id in range(1, NUM_SYSTEMS + 1):
    emitter_locations[system_id] = generate_single_mgrs_point(LAT_RANGE, LON_RANGE, precision=3)

all_data = []

print("Generating Data:")

for system_id, params in SYSTEM_MODULATION_PARAMS.items():
    
    # 1. Randomly determine the number of pulses for each emitter 
    n_pulses = np.random.randint(MIN_PULSES, MAX_PULSES)
    
    print(f"  > Emitter {system_id}: Generating {n_pulses} rows.")

    # 2. Generate core PDW parameters
    data = {
        'RF_MHz': np.random.normal(loc=params['RF_MHz'], scale=STDS_NON_MODULATED[0], size=n_pulses),
        'Amplitude_dB': np.random.normal(loc=params['Amplitude_dB'], scale=STDS_NON_MODULATED[1], size=n_pulses),
        'DOA_deg': np.random.normal(loc=params['DOA_deg'], scale=STDS_NON_MODULATED[2], size=n_pulses),
        'Emitter_ID': np.full(n_pulses, system_id)}

    # 3. Generate Modulated PRI and PW
    pri_func = MODULATION_FUNCTIONS[params['PRI']['type']]
    data['PRI_us'] = pri_func(n_pulses, params['PRI'])
    pw_func = MODULATION_FUNCTIONS[params['PW']['type']]
    data['PW_us'] = pw_func(n_pulses, params['PW'])

    df = pd.DataFrame(data)

    # 4. Add the Emitter Function
    df['Radar_Function'] = df['Emitter_ID'].map(EMITTER_FUNCTION_MAP)

    # 5. Add the clustered Emission Timestamp
    df['Timestamp'] = generate_cluster_timestamp(
        n_pulses, 
        overall_start=START_DATE, 
        overall_end=END_DATE, 
        activity_duration_hours=random.choice([0.5, 1, 2]))
    
    # 6. Add the fixed Location_MGRS
    df['Location_MGRS'] = emitter_locations[system_id]

    # 7. Compile features
    all_data.append(df)

pdw_data = pd.concat(all_data, ignore_index=True)

# Introduce some errors
DUPLICATE_RATE = 0.01
n_duplicates = int(len(pdw_data) * DUPLICATE_RATE)

print(f"Injecting {n_duplicates} duplicate rows to mimic multipath...")
duplicates = pdw_data.sample(n=n_duplicates, replace=False)
pdw_data = pd.concat([pdw_data, duplicates], ignore_index=True)

MISSING_RATE = 0.02
n_missing_rows = int(len(pdw_data) * MISSING_RATE)

print(f"Corrupting {n_missing_rows} rows with NaN values to mimic sensor dropouts...")
missing_indices = np.random.choice(pdw_data.index, size=n_missing_rows, replace=False)
cols_to_corrupt = ['RF_MHz', 'PW_us', 'PRI_us', 'Amplitude_dB', 'DOA_deg'] 
random_cols = np.random.choice(cols_to_corrupt, size=n_missing_rows)
col_indices = [pdw_data.columns.get_loc(c) for c in random_cols]

for row_idx, col_name in zip(missing_indices, random_cols):
    pdw_data.at[row_idx, col_name] = np.nan

# Convert RF_MHz from float to string
pdw_data['RF_MHz'] = pdw_data['RF_MHz'].astype(str)
pdw_data['RF_MHz'] = pdw_data['RF_MHz'].replace('nan', ' ')
pdw_data['RF_MHz'] = pdw_data['RF_MHz'].fillna(' ')
print("Converted 'RF_MHz' column from float to string type.")

# Shuffle and Save
pdw_data = pdw_data[['Timestamp', 'Emitter_ID', 'Radar_Function', 'RF_MHz', 'PW_us', 'PRI_us', 'Amplitude_dB', 'DOA_deg', 'Location_MGRS']]
pdw_data = pdw_data.sample(frac=1).reset_index(drop=True)

script_dir = os.path.dirname(__file__)
FILE_NAME = 'pdw_dataset.csv'
csv_path = os.path.join(script_dir, 'pdw_dataset.csv')

# Update your actual save path here if needed
pdw_data.to_csv(csv_path, index=False) 

# pdw_data.to_csv(FILE_NAME, index=False) 

print(f"Final dataset contains {len(pdw_data)} rows.")
print(f"  - {n_missing_rows} rows contain at least one NaN value.")
print(f"File saved as {FILE_NAME}")