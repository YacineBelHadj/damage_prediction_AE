import sqlite3
import pandas as pd
def load_all_from_db(path_to_db: str, table_name: str) -> pd.DataFrame:
    """
    Load all data from a specified table in an SQLite database into a pandas DataFrame.
    
    Parameters:
        path_to_db (str): The path to the SQLite database file.
        table_name (str): The name of the table to load data from.
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing all data from the specified table.
    """
    # Connect to the SQLite database
    try:
        conn = sqlite3.connect(path_to_db)
    except sqlite3.Error as e:
        print(f"Error: Could not connect to database. {e}")
        return pd.DataFrame()
    
    # Load data from the specified table into a pandas DataFrame
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(query, conn)
    
    # Close the connection
    conn.close()
    
    return df

import sqlite3

def print_database_overview(db_path):
    """
    Prints an overview of the SQLite database, including tables and their columns, 
    with a single-line format for each table.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
    """
    # Connect to the SQLite database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"\033[1;31mError: Could not connect to database. {e}\033[0m")
        return

    # Define styles for the shell output using ANSI escape codes
    HEADER = "\033[1;34m"  # Bold Blue
    TABLE = "\033[1;33m"  # Bold Yellow
    COLUMN = "\033[0;36m"  # Cyan
    RESET = "\033[0m"  # Reset to default

    # Print header
    print(f"{HEADER}SQLite Database Overview{RESET}")

    # Get the list of available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        print(f"{TABLE}No tables found in the database.{RESET}")
    else:
        # Print tables and their columns in a single-line format
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]  # Extract column names
            column_list = ", ".join(column_names)
            print(f"{TABLE}Table: {table_name}{RESET} {COLUMN}- {column_list}{RESET}")

    # Close the connection
    conn.close()

import numpy as np 
def get_statistic(
    db_path: str, 
    column_names: list[str], 
    statistic_funcs: list[callable], 
    table_name: str = "processed_data", 
    timestamp_range: tuple = None,
    add_condition: str = None
):
    """
    Retrieve statistics for specific columns from a database table.

    Args:
        db_path (str): Path to the SQLite database.
        column_names (List[str]): List of column names to analyze.
        statistic_funcs (List[Callable]): List of functions to apply for statistics (e.g., np.min, np.mean).
        table_name (str): Table name to query data from. Defaults to "processed_data".
        timestamp_range (tuple): Optional tuple specifying (start_time, end_time) for filtering timestamps.

    Returns:
        dict: Dictionary with statistics for each column.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"\033[1;31mError: Could not connect to database. {e}\033[0m")
        return

    # Build SQL query
    columns_str = ", ".join(column_names)
    query = f"SELECT {columns_str} FROM {table_name}"

    # Add timestamp filtering if provided
    if timestamp_range:
        query += f" WHERE timestamp > '{timestamp_range[0]}' AND timestamp < '{timestamp_range[1]}'"
    if add_condition:
        query += f" AND {add_condition}"
    try:
        # Execute query and load data
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"\033[1;31mError: Could not execute query. {e}\033[0m")
        return
    finally:
        conn.close()

    # Process and compute statistics
    results = {}
    for col in column_names:
        if col in df.columns:
            # Handle potential array-like data
            if 'Welch' in col or 'RollingAverage' in col:
                df[col] = df[col].apply(lambda x: np.frombuffer(x, dtype=np.float32))

            # Flatten data if arrays are present
            flattened = np.concatenate(df[col].dropna().values) if isinstance(df[col].iloc[0], np.ndarray) else df[col].dropna()
            
            # Compute requested statistics
            stats = {}
            for func in statistic_funcs:
                try:
                    stats[func.__name__] = func(flattened)
                except Exception as e:
                    stats[func.__name__] = f"Error: {e}"
            
            results[col] = stats
        else:
            print(f"\033[1;33mWarning: Column '{col}' not found in the table.\033[0m")

    return results