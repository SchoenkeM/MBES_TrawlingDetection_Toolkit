# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:41:27 2024
version 1.0.4
@author: mischa reineccius-schonke
"""

# Import package checker
import sys
from execute_task_fct import execute_task
from package_checker import check_packages_and_package_version

# Run package check
if not check_packages_and_package_version():
    sys.exit()
                                    
# Enter a single file to be processed (only one file will be processed):
Config_List = 'EMB267_config_demo.yaml'

# Example of combine multiple files into a list to be processed:
#Config_List = ['EMB238_config_demo.yaml', 'EMB267_config_demo.yaml', 'EMB288_config_demo.yaml', 'EMB320_config_demo.yaml', 'EMB345_config_demo.yaml']  


Opperation_Procedure = {
    'import_data': 'auto',           # use 'clean' to re-import all files in the folder/ #auto' only import *txt files that have not yet been converted 
    'process_data': 'auto',          # use 'clean' to re-import all files in the folder/ #auto' only import *txt files that have not yet been processed
    'merge_dataset': 'enabled',     # use 'enable'/disable to toggle on/Off
    'grid_dataset': 'enabled',      # use 'enable'/disable to toggle on/Off
    'export_tif': 'enabled',        # use 'enable'/disable to toggle on/Off
    'export_gpkg': 'enabled'        # use 'enable'/disable to toggle on/Off
    }


# starts processing
execute_task(Config_List, Opperation_Procedure)



