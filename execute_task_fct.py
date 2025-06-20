# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:48:18 2025
version 1.0.3
@author: misch
"""
from config_handler import ConfigHandler
from actionlog import Actionlog      
from data_handler import DataHandler
from processing_handler import ProcessingHandler
from output_handler import OutputHandler

#def execute_task(config_files, clean_import = False, clean_processing = False, export_tif= False, export_gpkg= False, compile_dataset = False, create_grid = False):
def execute_task(config_files, opperation_procedure):
    # in case export update is defined, skipp processing stepts and only 
    # update export
    
    def validate_pperation_procedure(opperation_procedure):
        
        # Define allowed values per key group
        allowed_values = {
            "filter": {"enabled", "disabled"},
            "mode": {"auto", "clean"}
        }
        print(">> Check Opperation Procedure Input Arguments...", end='')  # Log progress
        input_para = {
            "mode_1": opperation_procedure.get('import_data'),
            "mode_2": opperation_procedure.get('process_data'),       
            "filter_1": opperation_procedure.get('merge_dataset'),   
            "filter_2": opperation_procedure.get('grid_dataset'),   
            "filter_3":  opperation_procedure.get('export_tif'),       
            "filter_4": opperation_procedure.get('export_gpkg')     
            }

        # Validation function
        all_valid = True
        for key, value in input_para.items():
            matched = False
            for prefix, valid_set in allowed_values.items():
                if key.startswith(prefix):
                    if value not in valid_set:
                        index = list(input_para.keys()).index(key)
                        option = list(opperation_procedure.keys())[index]
                        print(f"\n\t- FAILED: Invalid Input Argument '{value}' for '{option}' [allowed: {valid_set}]")
                        
                        all_valid = False
                    matched = True
                    break
            if not matched:
                print(f"\nUnexpected config key: '{key}'")
                all_valid = False
        
        if all_valid == True:
            print(" [success]")
        else:
            print("\n>> One or more input options do not match the permitted specifications. Please check the opperation settings and restart the processing. ")
            
        return all_valid

    # Do not edit here
    if isinstance(config_files, str):       
        config_files = [config_files]   


    is_valid_opperations = validate_pperation_procedure(opperation_procedure)
    if not is_valid_opperations:
        return
    

    for config_file in config_files:
        
        print('\n#---')
        print(f">> Start processing using config: {config_file}")  # Log progress
        print('#---')
        #______________________________________________________________________
        # Initialize Classes
       
        log = Actionlog()  
        #log.set_autosave_off()     
        
        # Initialize configuration and handlers
        config = ConfigHandler(config_file)  
        
        Processing_handler = ProcessingHandler(log, config)
        MBES_data = DataHandler(log, config)
        Output_handler = OutputHandler(log, config)
        
        # if no export update is set, init processing classes
        """
        if [opperation_procedure.get('merge_dataset') == 'enabled' or
            opperation_procedure.get('grid_dataset') == 'enabled' or
            opperation_procedure.get('export_tif') == 'enabled' or 
            opperation_procedure.get('export_gpkg') == 'enabled']== True:
        
            Output_handler = OutputHandler(log, config)
        """
        #______________________________________________________________________
        # Runs Processing Steps
        
        if opperation_procedure.get('import_data') == 'auto':
            MBES_data.import_data(True)  # if update import is true skipp already converted txt files
        else:
            MBES_data.import_data(False)  # if update import is true skipp already converted txt files

        if opperation_procedure.get('process_data') == 'auto':
            Processing_handler.process_data(True)  # Process dataset
        else:
            Processing_handler.process_data(False)  # Process dataset
            
        if opperation_procedure.get('merge_dataset') == 'enabled':
            Output_handler.compile_tiles()  # Compile tiles
            
        if opperation_procedure.get('grid_dataset') == 'enabled':   
            Output_handler.grid_tiles()  # Grid tiles
        
        if opperation_procedure.get('export_tif') == 'enabled':
            Output_handler.export_geotiff()  # Export to GeoTIFF
            
        if  opperation_procedure.get('export_gpkg') == 'enabled':
            Output_handler.export_statisics()


        #Save configuration and log
        print('\n#---')
        config.save_config()
        log.write_to_csv(path=config._data_dir + '/output')
        print('#---')
        print(f"Completed processing for: {config_file}\n\n" + "-" * 75)  
        
    