from pathlib import Path
import re 
from datetime import datetime
from functools import partial
import numpy as np
def is_path_excluded(path: str, exclude_words: str = "Trash") -> bool:
    """
    Determines whether a file path should be excluded based on the presence of any excluded words.

    Args:
        path (str): The file path to check.
        exclude_words (List[str], optional): A list of words; if any of these words are present in the path, the path is excluded.
            Defaults to ["Trash"].

    Returns:
        bool: 
            - `True` if the path contains any of the excluded words (i.e., should be excluded).
            - `False` otherwise.

    Examples:
        >>> is_path_excluded("/user/data/Trash/file1.txt")
        False 
        >>> is_path_excluded("/user/data/file2.txt")
        True
        >>> is_path_excluded("/user/temp/file3.txt", exclude_words=["trash", "temp"])
        True
        >>> is_path_excluded("/user/data/archive/file4.txt", exclude_words=["trash", "temp"])
        False
    """
    # check if the Trash sequence is in the path
    return not exclude_words.lower() in path.lower()


def is_path_included(path: str, include_words: list[str] = []) -> bool:
    """
    Determines whether a file path should be included based on the presence of any inclusion words.

    Args:
        path (str): The file path to check.
        include_words (List[str], optional): A list of words; if any of these words are present in the path, the path is included.
            Defaults to an empty list, which means no inclusion filtering is applied.

    Returns:
        bool: 
            - `True` if the path contains at least one of the inclusion words.
            - `False` otherwise.
            - If `include_words` is empty, returns `True` by default, meaning all paths are included unless excluded.

    Examples:
        >>> is_path_included("/user/data/Important/file1.txt", include_words=["Important", "Critical"])
        True
        >>> is_path_included("/user/data/file2.txt", include_words=["Important", "Critical"])
        False
        >>> is_path_included("/user/data/file2.txt")
        True
    """

    path_lower = path.lower()
    return any(word.lower() in path_lower for word in include_words)
                                    
def extract_turbine_name(input_string,pattern = r'/NRT[^/]+/'):
    match = re.search(pattern, input_string, re.IGNORECASE)
    if match:
        turbine_name = match.group(0).strip('/')
        return turbine_name
    else:
        return None
    
def extract_timestamp(path: str) -> str| None :
    """ 
    Extracts and formats the timestamp from a given file path.

    Example:
        Input:
            '/media/owilab/7759343A6EC07E1C/data_primary_norther/NRTA01/TDD/TDD_IOT/2023/01/05/df-20230105_210000.parquet.gzip'
        Output:
            '2023-01-05 21:00:00'

    Args:
        path (str): The file path containing the timestamp.

    Returns:
        Optional[str]: The extracted timestamp in 'YYYY-MM-DD HH:MM:SS' format if parsing is successful; otherwise, `None`.

    Examples:
        >>> parse_timestamp_from_path("/data/df-20230105_210000.parquet.gzip")
        '2023-01-05 21:00:00'
        >>> parse_timestamp_from_path("/data/df-20231305_210000.parquet.gzip")
        None
    """
    try:
        filename = path.split('/')[-1]
        timestamp_str = filename.split('-')[1].split('.')[0]  # Extract '20230105_210000'
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        return timestamp.isoformat(' ')
    except (IndexError, ValueError) as e:
        # Log the error if logging is set up; for now, we'll just return None
        return None

def extract_turbine_name(input_string, pattern = r'/NRT[^/]+/'):
    match = re.search(pattern, input_string, re.IGNORECASE)
    if match:
        turbine_name = match.group(0).strip('/')
        return turbine_name
    raise ValueError("No turbine name found in the input string")

def add_interval_start(df):
    df['interval_start'] = df.index.floor('10min')
    return df

def add_interval_start_dask(dask_df):
    dask_df = dask_df.map_partitions(add_interval_start)
    
    return dask_df

def extract_fs(data):
    period = data.index[1] - data.index[0]
    fs = 1/period.total_seconds()
    return fs


def filter_based_on_len(df, required_len):
    return len(df) == required_len


def collate_data(data_list):
    num_samples = len(data_list)
    timestamps, turbine_names, sensors_names, signals = [], [], [], []
    for data in data_list:
        timestamps.append(data['timestamp'])
        turbine_names.append(data['turbine_name'])
        sensors_names.append(data['sensor_name'])
        signals.append(data['signal'])
    return {
        'timestamp': np.stack(timestamps),
        'turbine_name': np.stack(turbine_names),
        'sensor_name': np.stack(sensors_names),
        'signal': np.stack(signals),
    }
        
        
def structure_data(group,turbine_name):
    signal = group[['X','Y','Z']].values
    signal = signal.T # shape (3, len(signal))
    timestamp = group['interval_start'].iloc[0].isoformat(' ') 
    
    return {
        'signal': signal,
        'timestamp': timestamp,
        'turbine_name': turbine_name,
        'sensor_name': np.array(['X','Y','Z']).T,
    } 
def group_data(data):
    df, turbine_name = data
    structure_data_p = partial(structure_data, turbine_name=turbine_name)
    res = df.groupby('interval_start').apply(structure_data_p,include_groups=True)
    res = collate_data(res.to_list())
    return res

def convert_dict_to_tuples(data,order_keys):    
    sorted_data = {key: data[key] for key in order_keys if key in data}
    return list(zip(*sorted_data.values()))
