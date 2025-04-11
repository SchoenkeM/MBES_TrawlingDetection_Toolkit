# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:22:37 2025

@author: misch
"""

import importlib
import sys
import subprocess
import os


def check_and_install_packages():
    

    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    env_name = "Trawling_Detection_env"
    
    if current_env != env_name:
    
        print("_" *75)
        print(">> Compatibility check")
        print(f">> Check for vitual Environment: '{env_name}'")
        print("\n" + "-" * 75 )
        all_env = subprocess.check_output("conda info --envs", shell=True).decode()
        print(all_env)
        print("-" * 75 )
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        
        if env_name in result.stdout:
            print(">> Environment exists but is not active!") 
            print(">> Active Environment:", current_env)
            print(">>")
            print('>> Recommended procedure:')
            print(">>")
            print('>> 1) Close Spyder') 
            print('>> 2) Open Anaconda Navigator')
            print('>> 3) In Anaconda Navigator >> go to "Environments" (left menu bar)') 
            print(f'>> 4) In the middel menu bar locate "{env_name}" and hit the green play bottom next to it')
            print('>> 5) In Anaconda Navigator >> go to "Home"  (left menu bar)"')
            print('>> 6) In "Home" >> Launch "Spyder" (as it is a new environment, spyder may need to be installed first)')   

        else:
            print(">>")
            print(">> Environment does NOT exist!")
            print(">>")
            print(">> Recommended procedure:")
            print(">>")
            print('>> 1) Close Spyder') 
            print('>> 2) Open Anaconda Navigator')
            print('>> 3) In Anaconda Navigator >> go to "Environments" (left menu bar)')
            print('>> 4) In Environments >> select "Import" (the menu bar at the bottom)')
            print('>> 5) Use the "local drive" option >> navigate to the "Resource" Folder of the "TM_Processing_Toolbox" package')
            print('>> 6) Select the virtual Environment file: "{env_name}.yml" and import the file')      
            print(f'>> 7) In the middel menu bar locate the "{env_name}" and hit the green play bottom next to it')
            print('>> 8) Go to "Home" (left menu bar) >> Launch "Spyder" (as it is a new environment, spyder may need to be installed first)')  
        

    print("-" * 75 )  
    print(">> Checking the compatibility of the current environment")
    
    REQUIRED_PACKAGES = [
        "os", "pyproj", "yaml", "datetime", "pandas", "geopandas", "numpy", 
        "scipy", "shapely", "rasterio", "itertools", "fastparquet"
    ]   
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if not missing_packages:
        print(">> ✅ All required packages are installed.\n\n")
        return True

    print(">> ⚠️ The following packages are missing:", ", ".join(missing_packages))
    if "conda" in sys.version or "Continuum" in sys.version:
        print(">> Detected Conda environment. \n>> Please install missing packages using:")
        print(f">> conda install {' '.join(missing_packages)}")
        print(">>or:")
        print('>> In Anaconda Navigator >> go to "Environments" >> install missing packages')
        print(">>or:")
        print('>> Follow the instructions above to import the supplied Anaconda environment')
    else:
        print(">> Attempting to install missing packages via pip...")
    
    print("\nScript cannot be executed without these packages\n")
    
    print("-" * 75 + "\n" )
     
    return False