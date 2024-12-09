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
