# TrawlingDetection_Toolkit

The Trawling Toolbox takes processed and unprocessed MBES point cloud data in the format, xyz, ping, beam and calculates a zero mean surface, which is used to detect all structures below and above a certain threshold level. The detected structures are statistically analysed for each Gird cell within the EU standard Gird. The results of the statistical analysis are exported as geopackage and getiff in EPSG: 3035. In addition, the toolbox provides some optional filter functions. 

For users who want to work in Spyder, there is a virtual environment provided in the "resource" folder that can be imported and activated via the Ananconda navigator. 

For all users who want to work outside spider, make sure the following packages are available: ‘pyproj’, ‘yaml’, ‘datetime’, ‘pandas’, ‘geopandas’, ‘numpy’, ‘scipy’, “shapely”, “rasterio”, “itertools”, ’fastparquet’


# How to use the demo version 

1) For demonstration purposes, there are some datasets in the resource folder. Each Dataset must first be unpacked and stay inside in the resource folder. 

2) The structure of the demo version should look like this after downloading the repository. 

    TrawlingDetection_Toolkit:
        
        |-Resources (folder)/
            |
            |- emb238_xyz_demo (folder)/ (containing 2 *txt-files)
            |- emb267_xyz_demo (folder)/ (containing 2 *txt-files)
            |- emb288_xyz_demo (folder)/ (containing 2 *txt-files)
            |- emb320_xyz_demo (folder)/ (containing 2 *txt-files)
            |- emb435_xyz_demo (folder)/ (containing 2 *txt-files)
        |
        |- Main_TM-Processing.py (only execute this function to run everything)
        |- package_checker (if not using spyder the lines 14/15 in "Main_Processing.py" can be commented out)
        |- actionlog.py
        |- config_handler.py
        |- data_handler.py
        |- processing_handler.py
        |- output_handler.py
        |- execute_task_fnc.py
        |- subroutines.py
        |
