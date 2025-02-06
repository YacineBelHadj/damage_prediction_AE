CREATE_PROCCESSED_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS processed_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME,
    turbine_name TEXT,
    Welch_X BLOB,
    Welch_Y BLOB,
    Welch_Z BLOB,
    RMS_X DOUBLE,
    RMS_Y DOUBLE,
    RMS_Z DOUBLE,
    RollingAverage_X BLOB,
    RollingAverage_Y BLOB,
    RollingAverage_Z BLOB,
    Range_X DOUBLE,
    Range_Y DOUBLE,
    Range_Z DOUBLE,
    Mean_X DOUBLE,
    Mean_Y DOUBLE,
    Mean_Z DOUBLE
)
"""

DELETE_PROCESSED_DATA = """
DELETE FROM processed_data
"""
INSERT_PROCESSED_DATA = """
INSERT INTO processed_data
(timestamp, turbine_name, Welch_X, Welch_Y, 
 Welch_Z, RMS_X, RMS_Y, RMS_Z, RollingAverage_X, 
 RollingAverage_Y, RollingAverage_Z, Range_X, Range_Y,
 Range_Z, Mean_X, Mean_Y, Mean_Z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
ORDERED_COLUMNS_PROCESSED = ['timestamp', 'turbine_name', 'Welch_X', 'Welch_Y', 'Welch_Z', 'RMS_X', 'RMS_Y', 'RMS_Z', 'RollingAverage_X', 'RollingAverage_Y', 'RollingAverage_Z', 'Range_X', 'Range_Y', 'Range_Z', 'Mean_X', 'Mean_Y', 'Mean_Z']

CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS metadata (
    frequency_axis BLOB,
    sample_rate REAL,
    window_size INTEGER,
    processing_method TEXT
)
"""

DELETE_METADATA = """
DELETE FROM metadata
"""
INSERT_METADATA = """
INSERT INTO metadata
(frequency_axis, sample_rate, window_size, processing_method) VALUES (?, ?, ?, ?)
"""
ORDERED_COLUMNS_METADATA = ['frequency_axis', 'sample_rate', 'window_size', 'processing_method']