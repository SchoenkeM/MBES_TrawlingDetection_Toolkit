# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:41:27 2024
version 0.1.1
@author: mischa reineccius-schonke
"""

# Import package checker
import sys
from execute_task_fct import execute_task
from package_checker import check_and_install_packages

# Run package check
if not check_and_install_packages():
    sys.exit()
                                    
# Enter a single file to be processed (only one file will be processed):
Config_List = 'EMB267_config_demo.yaml'

# Example of combine multiple files into a list to be processed:
#Config_List = ['EMB238_config_demo.yaml', 'EMB267_config_demo.yaml', 'EMB288_config_demo.yaml', 'EMB320_config_demo.yaml', 'EMB345_config_demo.yaml']  


'''    
Optional "execute_task" function input arguments:

,update_import= True       # will only import *txt files that have not yet been converted to *.df-files.
                           
,update_processing= True   # wills only process *.df-files that have not yet been porcessed yet
             
,make_tiles= False         # If tiles have already been created, the data can be re-gridded without need to created the tiles again
                           
,update_tif= True          # If processed data exists, only runs exports geotiffs from processed data. Ignore file import, processing and gridding
                           
,update_gpkg= True         # If processed data exists, only exports geopackage from processed data. Ignore file import, processing and gridding
          

# Examples
# If new txt files have been added or only the grid option has been changed, use the following line:
# execute_task(Config_List, update_import = True, update_processing = True)

# If a new threshold is defined, use the following line to export the geopackage with the new threshold:
# execute_task(Config_List, update_gpkg= True)

'''

# starts processing
execute_task(Config_List, update_import = True)



