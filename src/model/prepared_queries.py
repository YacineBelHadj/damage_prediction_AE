create_melt_welch_only = """
DROP TABLE IF EXISTS melted;

CREATE TABLE melted AS  
    SELECT 
        id,
        "timestamp",
        turbine_name,
        'X' AS axis,
        Welch_X AS Welch
    FROM processed_data
    UNION ALL
    SELECT 
        id,
        "timestamp",
        turbine_name,
        'Y' AS axis,
        Welch_Y AS Welch
    FROM processed_data
    UNION ALL
    SELECT 
        id,
        "timestamp",
        turbine_name,
        'Z' AS axis,
        Welch_Z AS Welch
    FROM processed_data;
SELECT 
    ROW_NUMBER() OVER (ORDER BY id, axis) AS unique_id,
    id,
    "timestamp",
    turbine_name,
    axis,
    Welch
FROM melted;
""" 
simple_view = """
CREATE VIEW IF NOT EXISTS simple_view AS 
SELECT 
    id,
    "timestamp",
    turbine_name,
    Welch_X,
    Welch_Y,
    Welch_Z,
    RollingAverage_X,
    RollingAverage_Y,
    RollingAverage_Z
FROM processed_data
"""
    
welch_all_scada_all= """
DROP TABLE IF EXISTS dem;

CREATE TABLE dem AS
SELECT 
    pd.id,
    pd.timestamp,
    pd.turbine_name,
    pd.Welch_X,
    pd.Welch_Y,
    pd.Welch_Z,
    pd.RollingAverage_X,
    pd.RollingAverage_Y,
    pd.RollingAverage_Z,
    sc.DEM5_TP_SG_LAT014_Mtn AS DEM,
    sc.mean_windspeed,
    sc.mean_power,
    sc.mean_pitch,
    sc.std_pitch,
    sc.mean_rpm,
    sc.std_rpm,
    sc.caseID
FROM 
    processed_data pd
LEFT JOIN 
    scada sc
ON 
    pd.timestamp = sc.timestamp
    AND pd.turbine_name = sc.turbine_name
WHERE 
    pd.Welch_X IS NOT NULL 
    AND pd.Welch_Y IS NOT NULL 
    AND pd.Welch_Z IS NOT NULL 
    AND sc.DEM5_TP_SG_LAT014_Mtn IS NOT NULL;
"""
def welch1_scada1(not_null: list[str]= ['welch.Welch', 'scada.DEM']):
    # Base query
    res = """
    DROP TABLE IF EXISTS dem;
    CREATE TABLE dem AS
    WITH welch AS (
        SELECT 
            id,
            timestamp,
            turbine_name,
            Welch_X AS Welch
        FROM processed_data
    )
    SELECT 
        welch.id,
        welch.timestamp,
        welch.turbine_name,
        welch.Welch,
        scada.DEM 
    FROM welch
    LEFT JOIN (
        SELECT 
            timestamp,
            turbine_name,
            DEM5_TP_SG_LAT014_Mtn AS DEM
        FROM scada
    ) AS scada
    ON welch.timestamp = scada.timestamp 
    AND welch.turbine_name = scada.turbine_name
    """

    # Append additional conditions based on `not_null`
    for column in not_null:
        res += f" AND {column} IS NOT NULL"

    return res

add_key_to_scada="""
ALTER TABLE scada ADD COLUMN id INTEGER;

WITH ranked_rows AS (
    SELECT 
        rowid AS row_id, 
        ROW_NUMBER() OVER () AS id_value 
    FROM scada
)
UPDATE scada
SET id = (SELECT id_value FROM ranked_rows WHERE rowid = ranked_rows.row_id);

"""

do_nothing="""
SELECT 1 AS placeholder;
"""
