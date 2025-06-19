# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:22:37 2025
Version 1.1.0
@author: misch
"""


from packaging.version import parse
import importlib

"""
Vesion 1.1.0
    - rebuild function to work outside anaconda environemnt
    - remove query of current environment
    - add query to give feedback on installed package version number 
"""

def check_packages_and_package_version():  
    # Map of required packages and their minimum versions
    REQUIRED_PACKAGES = {
        "pyproj": "3.6.1",
        "yaml": "6.0.2",          # pyyaml is the real package name
        "pandas": "2.2.3",
        "geopandas": "1.0.1",
        "numpy": "1.26.4",
        "scipy": "1.15.1",
        "shapely": "2.0.6",
        "rasterio": "1.4.3",
        "fastparquet": "2024.2.0"
    }

    print("\n\n\n" + "-" * 75 )
    print(">> Checking the compatibility of the current environment")
    
    missing_or_outdated = []
    
    for package_name, required_version in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(package_name)
            installed_version = getattr(module, '__version__', None)
            
            if installed_version is None:
                print(f"\t-⚠️ {package_name} -> does not have a __version__ attribute.")
                continue
            
            if parse(installed_version) < parse(required_version):
                print(f"\t-❌ {package_name} version {installed_version} -> is less than required {required_version}")
                missing_or_outdated.append(f"{package_name} >= {required_version}")
            else:
                print(f"\t-✅ {package_name} {installed_version} -> is OK")

        except ImportError:
            print(f"\t-❌ {package_name} -> is not installed")
            missing_or_outdated.append(f"{package_name} >= {required_version}")

    # Summary
    if missing_or_outdated:
        print("\nYou need to install or upgrade the following packages:")
        for pkg in missing_or_outdated:
            print(f"  - {pkg}")
        
        print("\n>> In Spyder >> install missing packages using: conda install 'package name'")    
        print(">>or")
        print('>> In Anaconda Navigator >> go to "Environments" >> install missing packages')
        print(">>or")   
        print(">> If not unsing Anaconda >> Attempting to install missing packages via pip...") 
        print("\nScript cannot be executed without these packages\n")
        return False   
        
    else:
        print("\n🎉 All required packages are installed and meet version requirements.")
    
    print("-" * 75)
     
    return True

