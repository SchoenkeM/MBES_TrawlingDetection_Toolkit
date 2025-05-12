# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:56:20 2024
v.0.1.1
@author: misch
"""

import os

import pandas as pd
import geopandas as gpd
from datetime import datetime as dtime
import subroutines as subr
import numpy as np


class DataHandler:
    """
    Class to handle mbes footprint data

    Method option:
    .import_data(update = True) Imports only files which are not imported yet
    .import_data(update = False) Imports all *txt files from curretn directory
    """
    
    def __init__(self,log,config):
        """
        Initializes the MBESDataHandler class.
        """
        version = 'v.1.0.1'
        
        self._log = log
        self.vararg2 = 'MBES Data Handler'
        self._log.add(arg1='[init]', arg2=self.vararg2, arg3=[f'Initialize... {version}'])
                      
        # Set default
        self._EU_Grid_crs     = 3035                                            # in ETRS89-LAEA coordinates: etrs89_laea = CRS("EPSG:3035") independent of in and output
        
        # Get input from Config file
        self._fdir            = config._data_dir
        self._input_CRS       = config._data_crs
        self._data_columns    = config._data_columns
        self._max_beamNr      = config._Sonar_Nr_of_Beams
        
        self._tile_size       = config.Tile_Size
        self._config_filename = config._config_filename
        
        
        self._log.add(arg3='--> success')
        
    # >>>>>>>>>>>>>>>> Opperator Functions <<<<<<<<<<<<<<<<<<<
    #---
    def import_data(self, update):
        """
        Function: 001
        Manage the data Import:
            
        """  
        self.arg1_ID='[f001]'
         


        def verbose_init(filename, run, N):
            fullfile = os.path.join(self._fdir, filename)
            file_size = os.path.getsize(fullfile) / (1024 * 1024)
            self._log.add(arg1=self.arg1_ID, arg2=self.vararg2,
                          arg3=[f'\t- Conversion of File ({run}/{N}) in process:'])

            self._log.add(arg1=self.arg1_ID, arg2=self.vararg2,
                          arg3=[f'\t\t- Current File: {filename} ({file_size:.2f} MB)'])

        def verbose_metadata(data):
            
            metadata = data.attrs
            
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
                          arg3=[f"\t\t- MBES Stats: Beam Range {metadata['Beam_range']}",
                                f"Ping Range {metadata['Ping_range']}",
                                f"Total Soundings [{metadata['Nr_Soundings']}]"])
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
                          arg3=[f"\t\t- Area Stats: Lat Extend {metadata['Lat_extend']}",
                                f"Lon Extend {metadata['Lon_extend']}",
                                f"Depth Range {metadata['Depth_range']}"])
            
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
                          arg3=[f"\t\t- Tile Stats: Nr of Tiles {metadata['Nr_of_Tiles']}",
                                f"Tile Buffer id Range {metadata['Buffer_id']}"])

        def verbose_completed(import_runtime):
            elapseTime = subr.toc(import_runtime)
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
                          arg3=f'\t\t- Raw data of current file successfully imported. Elapse time: {elapseTime}')

        def function_handler():
        # Function hanndler:
            
            print('\n#---',  end='', flush=True)
            
            self.arg1_ID='[f100]'
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=['Start converting *.txt data into dataframe...']) 
            
            if not self._check_inputDir():
                return
            
            filelist = self._get_filelist(update)
            
            if len(filelist)==0:
                self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
                arg3='\t\t- Empty Filelist. Import Process aborted')
                return
        
            # Loop over filelist to import text data
            # there could be two functions one to solely import text files
            # and one to reporcess
            run = 1
            N= len(filelist)
            for filename in filelist:
                
                import_runtime = subr.tic()
                
                # create fullfile name and output data name
                txt_fullfile = os.path.join(self._fdir, filename)
                pkl_fullfile = f'{txt_fullfile[:-4]}.pkl'

                verbose_init(filename, run, N)

                data = self._txt2pd_Dataframe(txt_fullfile)

                data = self._convert_to_gdf(data)
                
                data = self._add_tile_id(data)

                data = self._add_buffer_id(data)  
                
                data = self._create_metadata(data, filename)
                
                verbose_metadata(data)   
                
                self._clear_existing_df(pkl_fullfile)

                self._save_current_df(data, pkl_fullfile)

                verbose_completed(import_runtime)

                run += 1 # runcounter

            self._log.add(arg1=self.arg1_ID, arg2=self.vararg2,
            arg3='Import Process of Data Completed...')

        #--- Function Call
        function_handler()
            
        self.arg1_ID=[]

    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    def _check_inputDir(self):
        """
        Checks if input dir is valid
        """
        lokal_ID= '[f101]'
        # Input Vars:
        input_dir = self._fdir # From Config
        
        #__________________________________________________
        if not os.path.isdir(input_dir):
        # Check input directory
            self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                          arg3=['[FAILED] to open Input'
                                f' Dir from config.py: {input_dir}'])
            return False
        else:
            self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                    arg3=[f'\t- Input Dir from config.py: {input_dir}'])
        return True
    
    def _get_filelist(self,update):
        
        lokal_ID= '[f102]'
        
        # Input Vars:
        fdir = self._fdir # From Config
        
        #__________________________________________________

        self._log.add(arg1= lokal_ID, arg2=self.vararg2,
                      arg3="\t- Create  *.txt- Filelist for txt2pkl conversion")
        
        all_files = set(os.listdir(fdir))  # Load all filenames into a set for fast lookup
        txt_filelist = [f for f in all_files if f.endswith('.txt')]  # Find all .txt files
        
        self._log.add(arg3= f"--> ({len(txt_filelist)}) *txt-file(s) located" ) 
        
        if update==True:
            self._log.add(arg1= lokal_ID, arg2=self.vararg2,
                         arg3= ["\t- Option 'update' selected -> irgnoring already imported '*.txt' files"])
            
            self._log.add(arg1= lokal_ID, arg2=self.vararg2,
                          arg3="\t- Searching for unimported *txt-File(s) in process...")
            
            updated_fileslist = []
            for txt_file in txt_filelist:
                base_name = txt_file.rsplit('.', 1)[0]  # Remove .txt extension
            
                # Construct expected .pkl and _proc.pkl filenames
                pkl_file = f"{base_name}.pkl"
                proc_pkl_file = f"{base_name}_proc.pkl"
            
                # If neither exists, add to result
                if pkl_file not in all_files and proc_pkl_file not in all_files:
                    updated_fileslist.append(txt_file)
            
            txt_filelist = updated_fileslist     
            self._log.add(arg3= f"-->  ({len(updated_fileslist)}) unimported *.txt-files located" )    

        self._log.add(arg1= lokal_ID, arg2=self.vararg2,
                      arg3=f"\t- ({len(txt_filelist)}) file(s) selected for import")  
       
        return txt_filelist
    
    def _txt2pd_Dataframe(self, fullfile):
        # Predefine return output
        
        lokal_ID= '[f103]'
        
        # Input from Config
        data = None
        data_columns = self._data_columns
        
        #______________________________________________________________________
        self._log.add(arg1=lokal_ID, arg2=self.vararg2, arg3=['\t\t- Import *.txt-file in process...'])
 
        # --- Header detection
        with open(fullfile, 'r') as file:
            first_line = file.readline().strip()
    
        # Split the first line by commas (assuming comma-separated)
        first_row = first_line.split(',')
    
        # Check if all elements in the first row are numeric
        is_numeric = all(subr.is_float(element) for element in first_row)
        stime = subr.tic()
    
        try:
            # Read the file using the correct column order
            if is_numeric:
                data = pd.read_csv(fullfile, usecols=[0, 1, 2, 3, 4], names= data_columns, header=None)
                
            else:
                
                data = pd.read_csv(fullfile, usecols=[0, 1, 2, 3, 4], names= data_columns, header=0)
    
            elapseTime = subr.toc(stime)
            self._log.add(arg3=f'--> success [Elapse Time: {elapseTime}]') 

        except Exception as e:
            # Handle failure cases
            self._log.add(arg3=f'[FAILED]: Unable to open file - {e}')

        if isinstance(data, dict):
            data = pd.DataFrame(data)
    
        return data

    def _convert_to_gdf(self, data):
        """
        convert data in a gdf for faster visualisation
        """
        lokal_ID= '[f104]'
        
        # Input Vars from Config
        input_CRS = self._input_CRS
        EU_crs = self._EU_Grid_crs
        
        #______________________________________________________________________
        
        self._log.add(arg1=lokal_ID,arg2=self.vararg2,
                      arg3=[f'\t\t- Conversion of *txt-data (EPSG:{input_CRS}) '
                            f'into gdf (EPSG:{EU_crs}) in process...'])

        multipoint_series = gpd.GeoSeries.from_xy(data['lon'], data['lat'], crs = input_CRS)
        data['coords']= multipoint_series
        
        data['ping'] = data['ping'].astype('category')
        data['beam'] = data['beam'].astype('category')
        
        columns2keep=['coords','ping','beam','depth']
        data = data[columns2keep]

        data = gpd.GeoDataFrame(
            data, 
            geometry='coords',
            crs=input_CRS
        )
        
        data = data.to_crs(EU_crs)
        data= data[data.geometry.type == 'Point']
        self._log.add(arg3=' --> success')
        return data
    
    def _add_tile_id(self, data):
        
        
        lokal_ID= '[f105]'
        
        # Input Vars from Config
        tile_size = self._tile_size
        
        #______________________________________________________________________
        
        
        self._log.add(arg1=lokal_ID,arg2=self.vararg2,
                      arg3=['\t\t- Add EU standard Grid "tile_id" to dataframe in process...'])

        lon = data.geometry.x
        lat = data.geometry.y
           
        x_coords = np.floor(lon / tile_size).astype(int)
        y_coords = np.floor(lat / tile_size).astype(int)

        if tile_size<1000: 
            tileSize = str(tile_size) + 'm'
        else:
            tileSize = str(tile_size/1000) + 'km'
            
        # Combine into a single string Series
        tile_ID = pd.Categorical(tileSize + "E" + x_coords.astype(str) + "N" + y_coords.astype(str))
      
        data['tile_id'] = tile_ID
        
        self._log.add(arg3=' --> success.')
        return data
    
    def _add_buffer_id(self, data):
        """
        Assigns boundary layer IDs to the dataset:
        - Identifies outermost points per tile.
        - Assigns -1/1 to the outermost tiles form center of Profile,
          the largest values to the center part.
        - Iteratively moves inward until all points are labeled.

        Parameters:
        -----------
        data : pd.DataFrame
            Data containing 'ping', 'beam', 'tile_id', and 'coords' (shapely Point geometry).
        
        Returns:
        --------
        boundary_id : pd.Series
            A Pandas Series containing boundary layer IDs for each point.
        """
        
        #______________________________________________________________________
        lokal_ID = '[f106]'
        
        self._log.add(arg1= lokal_ID, arg2=self.vararg2, arg3='\t\t- Compute Tile Buffer ID... ')

        # Step 1: Extract coordinates and compute tile IDs
        data_temp = data[['ping', 'beam', 'tile_id']].copy()

        # Step 2: Initialize boundary_id with NaN (unassigned)
        boundary_id = pd.Series(np.nan, index=data_temp.index)

        data_temp['ping'] = pd.to_numeric(data_temp['ping'], errors='coerce')  
        data_temp['beam'] = pd.to_numeric(data_temp['beam'], errors='coerce')  
        
        id_group = 1  # Boundary ID counter
        counter = 0 
        N=len(data)+1
        while boundary_id.isna().any() and counter < N:
            # Step 3: Identify outermost points per ping
            first_points = data_temp.groupby('ping', observed=True)['beam'].idxmin()
            last_points = data_temp.groupby('ping', observed=True)['beam'].idxmax()

            # Get corresponding lat/lon for these points
            smallest_tiles = data_temp.loc[first_points, 'tile_id'].unique()
            largest_tiles = data_temp.loc[last_points, 'tile_id'].unique()

            # Step 4: Assign boundary IDs
            boundary_id.loc[data_temp[data_temp['tile_id'].isin(smallest_tiles)].index] = -id_group
            boundary_id.loc[data_temp[data_temp['tile_id'].isin(largest_tiles)].index] = id_group

            # Step 5: Remove assigned points from further iterations
            mask_unassigned = boundary_id.isna()
            data_temp = data_temp.loc[mask_unassigned]
            
            id_group += 1  # Increment boundary ID
            counter += 1
            
        data['buffer_id'] = boundary_id.astype('category')
        
        # Log success message with time taken
        self._log.add(arg3='--> success') 

        return data

    def _create_metadata(self, data, filename):
    
        lokal_ID = '[f107]'
        
        # Input from config:
        f_dir = self._fdir
        log_name = self._log.get_filename()
        config_fname = self._config_filename 
        input_crs = self._input_CRS
        max_BeamNr = self._max_beamNr
        
        #______________________________________________________________________
    
        self._log.add(arg1=lokal_ID,arg2=self.vararg2,
                      arg3=['\t\t- Add metadata to Dataframe in process...'])
        
        beam= np.unique(data['beam'].astype(int))
        ping= np.unique(data['ping'].astype(int))
        depth = data['depth'].astype(float)
        buffer_id = abs(np.unique(data['buffer_id'])).astype(int)
        
        metadata = {
            'file_name':  filename,
            'file_directory': f_dir,
            'creation_date': dtime.now().strftime('%Y-%m-%d'),
            'creation_time': dtime.now().strftime('%H:%M:%S'),
    
            'logfile_name': log_name,
            'config_name':  config_fname,
    
            'input_CRS_EPSG': input_crs,
            'Nr_Soundings': data.shape[0],
            'Nr_of_Beams':  max_BeamNr,
    
            'Beam_range': [beam.min(), beam.max()],
            'Ping_range': [ping.min(), ping.max()],
            'Depth_range': [depth.min(), depth.max()],
    
            'Lat_extend': [int(data['coords'].y.min()),
                           int(data['coords'].y.max())],
            'Lon_extend': [int(data['coords'].x.min()),
                           int(data['coords'].x.max())],
    
            'Nr_of_Tiles': len(data['tile_id'].unique()),
            'Buffer_id': [buffer_id.min(), buffer_id.max()],
            
            'Applied_Filter': None,
            'Nr_of_Outliers': None
            }
        
        data.attrs = metadata # add metadata to dataframe
        
        self._log.add(arg3=' --> success')
        return data

    def _clear_existing_df(self, pkl_fullfile):
        """
        Delet Recovery File, if exist
        """
        lokal_ID = '[f108]'
        msg= '\t\t- Delete existing "*.pkl"-file version to avoid overwrite conflicts...'
        
        # checks if pkl already exist
        if os.path.isfile(pkl_fullfile):
            self._log.add(arg1=lokal_ID, arg2=self.vararg2,arg3=msg)     
            os.remove(pkl_fullfile)
            self._log.add(arg3='--> Done')
        
        # checks if processed pkl already exist
        pkl_proc_fullfile = pkl_fullfile.replace(".pkl", "_proc.pkl")    
        if os.path.isfile(pkl_proc_fullfile):
            self._log.add(arg1=lokal_ID, arg2=self.vararg2,arg3=msg)  
            os.remove(pkl_proc_fullfile)
            self._log.add(arg3='--> Done')
            
    def _save_current_df(self, df, pkl_fullfile):
        
        lokal_ID = '[f109]'
        
        self._log.add(arg1=lokal_ID,arg2=self.vararg2,
                      arg3='\t\t- Save current file as "*.pkl" under import dir...')
        # store geodataframe in input direcoty as save file
        df.to_pickle(pkl_fullfile)
        file_size = os.path.getsize(pkl_fullfile) / (1024 * 1024)
        self._log.add(arg3=f'--> success ({file_size:.2f} MB)')