view_melt_welch_only = """
CREATE VIEW IF NOT EXISTS melted_view AS
WITH melted AS (
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
    FROM processed_data
)
SELECT 
    ROW_NUMBER() OVER (ORDER BY id, axis) AS unique_id,
    id,
    "timestamp",
    turbine_name,
    axis,
    Welch
FROM melted;
""" 
create_merge_scada= """ DROP TABLE IF EXISTS merged;
CREATE TABLE merged AS
WITH melted AS (
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
    FROM processed_data
)
SELECT 
    melted.id,
    melted.timestamp,
    melted.turbine_name,
    melted.Welch,
    melted.DEM,
    scada.PredDEM
FROM welch
LEFT JOIN (
    SELECT 
        timestamp,
        turbine_name,
        DEM5_TP_SG_LAT014_Mtn AS DEM,
        pred_Mtn_dnn AS PredDEM
        
    FROM scada
) AS scada
ON welch.timestamp = scada.timestamp 
   AND welch.turbine_name = scada.turbine_name
WHERE welch.Welch IS NOT NULL AND scada.DEM IS NOT NULL;
"""
