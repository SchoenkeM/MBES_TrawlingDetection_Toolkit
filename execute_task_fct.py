# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:48:18 2025

@author: misch
"""
from config_handler import ConfigHandler
from actionlog import Actionlog      
from data_handler import DataHandler
from processing_handler import ProcessingHandler
from output_handler import OutputHandler

def execute_task(config_files, update_import = False, update_processing = False, update_tif= False, update_gpkg= False, make_tiles = True):

    # in case export update is defined, skipp processing stepts and only 
    # update export
    skip_processing =  update_tif or update_gpkg
    
    # Do not edit here
    if isinstance(config_files, str):
        
        config_files = [config_files]   

    for config_file in config_files:
        
        print(f"Processing with config: {config_file}")  # Log progress
        #______________________________________________________________________
        # Initialize Classes
       
        log = Actionlog()  
        #log.set_autosave_off()     
        
        # Initialize configuration and handlers
        config = ConfigHandler(config_file)  
        
        # if no export update is set, init processing classes
        if skip_processing:
            Output_handler = OutputHandler(log, config)
        else: 
            MBES_data = DataHandler(log, config)
            Processing_handler = ProcessingHandler(log, config)
            Output_handler = OutputHandler(log, config)
        
        #______________________________________________________________________
        # Runs Processing Steps
        
        if skip_processing:
            # If update_tif is true
            if update_tif:
                Output_handler.export_geotiff()  # Export to GeoTIFF
            
            # If update_gpkg is true
            if update_gpkg:
                Output_handler.export_statisics()  # Export statistics
                
        else:       
            # converts txt2df
            MBES_data.import_data(update_import)  # if update import is true skipp already converted txt files
            # applies filter
            Processing_handler.process_data(update_processing)  # Process dataset
            
            # Output Handler
            if make_tiles:
                Output_handler.compile_tiles()  # Compile tiles
            Output_handler.grid_tiles()  # Grid tiles
            Output_handler.export_geotiff()  # Export to GeoTIFF
            Output_handler.export_statisics()  # Export statistics

        #Save configuration and log
        config.save_config()
        log.write_to_csv(path=config._data_dir + '/output')
    
        print(f"Completed processing for: {config_file}\n\n" + "-" * 50)  
        
    