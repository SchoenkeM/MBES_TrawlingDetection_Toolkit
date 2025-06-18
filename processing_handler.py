# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:12:34 2024

@author: misch
"""

import os
import pandas as pd
import numpy as np
import subroutines as subr
from scipy.signal import savgol_filter


class ProcessingHandler:
    def __init__(self, log, config):
        """
        Initialize Processing Handler 
        Version 1.0.0
        """ 
        
        version = 'v.1.0.0'

        self._log = log
        
        # Feedback
        self.vararg2 = 'Processing Handler'
        self._log.add(arg1='[init]',arg2=self.vararg2,arg3=[f'Initialize... {version}']) 
        
        # get variables from dir
        self._fdir = config._data_dir
        self._max_BeamNr = config._Sonar_Nr_of_Beams
        
        # Tile Options
        self._tile_size = config.Tile_Size
        
        # Filter Settings
        self._beam_threshold =  config._reference_surface['min_required_Beams'] 
        self._filter_len = config._reference_surface['filter_window_in_perc']
        
        self._filter_beam_nr = config._filter_options['exclude_Beam_Numbers']
        #self._filter_buffer_id = config._filter_options['exclude_Buffer_ID']
        self._vertical_detection_range = config._filter_options['detection_Window_Range']
        self._vertical_detection_intervall = config._filter_options['max_allowed_vertical_distribution']
        

        self._log.add(arg3='--> success')

    def process_data(self, update):

        # predefine parameter status    
        total_runtime = subr.tic()
        
        print('\n#---',  end='', flush=True)
        
        self.arg1_ID='[f200]'
        self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=['Start process File(s) from Dataset...'])  
        
        #---
        # Check input directory
        if not self._check_dir():
            return 
                            
        #---
        # create filelist, ensures that only existing df are in list which
        # area contained in the dataset
        filelist = self._get_filelist(update)
        if len(filelist) ==0:
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
            arg3='\t\t- Filelist is empty with no files to process. Process aborted')
            return
        
        run = 1
        N = len(filelist)
        for fname in filelist:
   
            #--- Import and read dataframe        
            # get data from dataframe. "data" is a pointer to the dataframe
            # to the same cash. Changes to data, will automaticcaly effect
            # file_df. Store the row length of data for validation to check if 
            # output flags match length of input data
            import_runtime = subr.tic()
            
            if fname[-3:] != 'pkl':
                fname = fname + '.pkl'
                     
            fullfile_pkl = os.path.join(self._fdir, fname)
                
            self._log.add(arg1=self.arg1_ID, arg2=self.vararg2,
                       arg3=[f'\t- Process File({run}/{N}): {fname[:-4]}'])
            
            data = pd.read_pickle(fullfile_pkl)
            
            #---
            # Reset all Filters and detected Outliers
            self.applied_Filter =  []
            
            if 'outliers' in data:
                data.drop(columns='outliers', inplace=True)
            
            #--- Init Outlier df
            self.outlier_df = pd.DataFrame({'total_Outliers':
                               pd.Series([False] * len(data), dtype=bool)}) 
            
            #--- Compute reference surface 
            data['ref_depth'] = self._compute_reference_depth(data)
     
            #--- Call various Filter Functions
            self._exclude_beam_nr(data)
 
            #self._exclude_buffer_ids(data)
            
            self._apply_detection_window(data)

            self._compute_vertical_distribution(data)

            # Compute total outliers
            number_of_outliers = sum(self.outlier_df['total_Outliers'])
            Outlier_Per = number_of_outliers/len(data)*100
            
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
            arg3=f'\t\t- Total Number of Outliers detected: {number_of_outliers} [{Outlier_Per:.2f}%] ')
            
            #-- save Outlier values to file_df 
            data.attrs['Applied_Filter'] = self.applied_Filter
            data.attrs['Nr_of_Outliers'] = number_of_outliers 
            
            data['outliers'] = self.outlier_df['total_Outliers']
            
            #--- save dataframe to disk
            self._save_current_df(data, fullfile_pkl)
            
            #--- Feedback
            elapseTime = subr.toc(import_runtime)
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
                          arg3=f'\t\t- Processing of current File completed [Elapse time: {elapseTime}]')
        
            run += 1
    
        elapseTime = subr.toc(total_runtime)
        self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,arg3=[f'Processing completed. [Elapse time: {elapseTime}]']) 
  
        
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    def _check_dir(self):
        """
        Checks if input dir is valid
        """
        lokal_ID= '[f201]'
        input_dir = self._fdir
        
        #______________________________________________________________________
        
        if os.path.isdir(input_dir):
        # Check input directory
            self._log.add(arg1=lokal_ID, arg2=self.vararg2,
              arg3=[f'\t- Input Dir from config.py: {input_dir}']) 
        else:
            self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                          arg3=['[FAILED] to open Input'
                                f' Dir from config.py: {input_dir}'])
            return False
        
        return True

    def _get_filelist(self,update):
        """
        Checks if input 'pkl' files exist and creates filelist of valid df.
        Also renames selected '_proc.pkl' to 'pkl' before processing. 
        """
        
        lokal_ID= '[f202]'
        fdir = self._fdir
        
        
        #---
        self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                          arg3="\t- Create filelist of *.pkl-files from input directory:")

        # check for input option update
        all_files = set(os.listdir(fdir))
        
        # Just feedback section 
        # get file names from file directory ending with ".pkl"
        pkl_files = [f for f in all_files if f.endswith(".pkl") and not f.endswith("_proc.pkl")]
        self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                      arg3= [f'\t\t- ({len(pkl_files)}) unprocessed "*pkl-files" located']) 
        
        # get file names from file directory ending with "_proc.pkl"
        pkl_proc_files =  [f for f in all_files if f.endswith("_proc.pkl")]     
        self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                      arg3= [f'\t\t- ({len(pkl_proc_files)}) processed "*_proc.pkl" files located'])     
        
        #----
        # Section to compute updated filelist ignoring already processed pkl files
        filelist = []     
        if update:
            # only considers  uprocessed *.pkl files
            self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                         arg3= ['\t\t- Input option "update" selected. Ignoring aleady processed "*.pkl"-files'])
            
            pkl_files = [f for f in all_files if f.endswith('.pkl')]  # Find all .txt files
            filelist = []
            for pkl_file in pkl_files:
                
                # Construct expected  _proc.pkl filenames
                base_name = pkl_file.rsplit('.', 1)[0]  # Remove .pkl extension
                if "_proc" not in base_name:
                    proc_pkl_file = f"{base_name}_proc.pkl"
                else:
                    proc_pkl_file = f"{base_name}.pkl"
                
            
                # If neither exists, add to filelist
                if  proc_pkl_file not in all_files:
                    filelist.append(pkl_file)
            
        else:
            # rename all '_proc' files to mark them as unprocessed
            for file in all_files:
                if file.endswith("_proc.pkl"):
                    os.rename(os.path.join(fdir, file), os.path.join(fdir, file.replace("_proc.pkl", ".pkl")))
            
            # after renaming, get *pkl files list including all ".pkl" files  
            all_files = set(os.listdir(fdir))  
            filelist = [f for f in all_files if f.endswith(".pkl")] 
                               
        # remove dublicates from list    
        filelist = list(set(filelist)) 
           
        self._log.add(arg1=lokal_ID, arg2=self.vararg2,
                      arg3= f"\t\t- ({len(filelist)}) File(s) selected for processing" ) 
        
        return filelist

    def _compute_reference_depth(self, data):
        
        
        local_ID= '[f203]'
        
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        stime= subr.tic()
        self._log.add(arg1=local_ID,arg2=self.vararg2,
                      arg3='\t\t- Compute reference Surface... ')
        
        #--- Input Arguments from Config
        max_BeamNr = self._max_BeamNr
        lower_threshold =  self._beam_threshold
        filter_len = self._filter_len/100
        
        
        win_len= round(max_BeamNr*filter_len) 
        lat = data['coords'].geometry.y
        lon = data['coords'].geometry.x
        
        #--- opperation
        # Create DataFrame and group by 'ping'
        df = pd.DataFrame({'x': lon, 'y': lat, 'z': data['depth'], 'group': data['ping']})
        grouped = df.groupby('group', observed=True)
        
        # Initialize arrays
        z_ref = np.full_like(data['depth'], np.nan)
        bool_vec = np.full(len(data['depth']), False)
        
        # Count variables
        counter_valid_pings = 0
        counter_excluded_pings = 0
        

        # Process each group in a loop
        for group_id, group_data in grouped:
            idx = group_data.index
            x = group_data['x'].values - group_data['x'].min()
            y = group_data['y'].values - group_data['y'].min()
            z = group_data['z'].values
            
            # Convert to polar
            theta, rho, z_polar = subr.cart2pol(x, y, z)
            
            # Check if group has enough points
            if len(z_polar) >= win_len and len(z_polar) >= lower_threshold:
                z_smooth = savgol_filter(z_polar, win_len, polyorder=2)
                _, _, z_smooth = subr.pol2cart(theta, rho, z_smooth)
                z_ref[idx] = z_smooth
                counter_valid_pings += 1
            else:
                counter_excluded_pings += 1
            
        bool_vec = np.isnan(z_ref)
        Outlier_Per = sum(bool_vec)/len(bool_vec)*100

        #--- Add flag to Outlier_df
        N = len(data['depth']) #controll value
        flag = pd.Series(bool_vec, index=data.index)
        self._add_Outlier('mean_depth', flag, N) 
        
        #--- final feedback
        self._log.add(arg3=f'--> success [Elapse Time: {subr.toc(stime)}]') 
    
        self._log.add(arg1=local_ID,arg2=self.vararg2,
              arg3=f'\t\t  Number of Pings outside threshold --> less than {lower_threshold} beams per ping: ({counter_excluded_pings})')        
        self._log.add(arg1=local_ID,arg2=self.vararg2,
              arg3=f'\t\t  Total number of Footprints outside threshold: {sum(bool_vec)} [{Outlier_Per:.2f}%]')

        return z_ref        

    def _exclude_beam_nr(self, data):
        '''
        In Progress
        Input Parameters: 
        - data['beam']
        '''
        local_ID = '[f204]'
       
        self._log.add(arg1=local_ID,arg2=self.vararg2,
                      arg3='\t\t- Flag selected beam numbers to be excluded...')
        
        #--- Input Arguments from Config
        beam_array= self._filter_beam_nr
        
        if beam_array is None:
            self._log.add(arg3='--> skipped')
            return
        
        stime= subr.tic()
        #--- Opperations
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        beam = data['beam']
        if not isinstance(beam, pd.Series):
            raise ValueError("Input must be a Series")
        
        # Get the unique integers and their counts
        # Create a boolean mask where True represents matching values
        bool_vec = beam.isin(beam_array)
        Outlier_Per = sum(bool_vec)/len(bool_vec)*100
        
        #--- Add flag to Outlier_df
        N = len(data['depth']) 
        flag= pd.Series(bool_vec, index=data.index)
        self._add_Outlier('excluded_Beams', flag, N)

        #--- final feedback
        self.applied_Filter.append('exclude_beams')
        self._log.add(arg3=f'--> success [Elapse Time: {subr.toc(stime)}]')
        self._log.add(arg1=local_ID,arg2=self.vararg2,
        arg3=f'\t\t  Number of excluded Footprints by Beam number: {sum(bool_vec)} [{Outlier_Per:.2f}%] ')
    
    """
    def _exclude_buffer_ids(self, data):
        '''
        In Progress
        Input Parameters: 
        - data['beam']
        '''
 
        local_ID = '[f205]'
        self._log.add(arg1=local_ID,arg2=self.vararg2,
                      arg3='\t\t- Flag selected Tile Buffer Id to be excluded...')
        
        #--- Input Arguments from Config
        buffer_array = self._filter_buffer_id
        
        if buffer_array is None:
            self._log.add(arg3='--> skipped')
            return
        
        stime= subr.tic()
        #--- Opperations
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        buffer_id = data['buffer_id']
        if not isinstance(buffer_id, pd.Series):
            raise ValueError("Input must be a Series")
        
        # Get the unique integers and their counts
        # Create a boolean mask where True represents matching values
        bool_vec = buffer_id.isin(buffer_array)
        Outlier_Per = sum(bool_vec)/len(bool_vec)*100
        
        #--- Add flag to Outlier_df
        N = len(data['depth']) 
        flag= pd.Series(bool_vec, index=data.index)
        self._add_Outlier('excluded_Buffer_id', flag, N)

        #--- final feedback
        self.applied_Filter.append('Buffer_id')
        self._log.add(arg3=f'--> success [Elapse Time: {subr.toc(stime)}]')
        self._log.add(arg1=local_ID,arg2=self.vararg2,
        arg3=f'\t\t  Number of excluded Footprints by Tile Buffer id: {sum(bool_vec)} [{Outlier_Per:.2f}%] ')    
    """
    
    def _apply_detection_window(self, data):
        """
        Flags values in data['depth'] where the difference between consecutive values
        exceeds the detection_win threshold within each group in data['ping'].
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the 'depth' and 'ping' columns.
        detection_win (float): min/max Threshold for detecting large differences.
        
        Returns:
        pd.Series: A Boolean Series of the same length as data['depth'],
                   where True indicates a flagged outlier.
                   
        Important requires residual bathymertry comptued in advance 

        Input Parameters: 
        - data['depth']       
        - data['ref_depth']
        """
        
        local_ID = '[f206]'
        
        #--- Input Arg from Config
        detection_win = self._vertical_detection_range # vertical detection window in meter
        
        if detection_win is None:
            self._log.add(arg3='--> skipped')
            return
        
        #--- opperation
        stime= subr.tic()
        self._log.add(arg1=local_ID,arg2=self.vararg2,
                      arg3='\t\t- Flag values outside detection window...')
        
        # Convert data to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame(data)
    
        z = data['depth'].values
        z_ref = data['ref_depth'].values
        
        spacings = z-z_ref
        Outlier_flags = (spacings < min(detection_win)) | (spacings > max(detection_win))
        
        bool_vec= pd.Series(Outlier_flags, index=data.index)
        Outlier_Perc= sum(Outlier_flags)/len(Outlier_flags)*100
        
        #--- Add flag to Outlier_df
        N = len(data['depth']) 
        flag= pd.Series(bool_vec, index=data.index)
        self._add_Outlier('detection_Window',flag ,N)
        
        
        #--- final feedback
        self.applied_Filter.append('detection_win')
        self._log.add(arg3=f'--> success [Elapse Time: {subr.toc(stime)}]') 
        self._log.add(arg1=local_ID,arg2=self.vararg2,
        arg3=f'\t\t  Number of excluded Footprints: {sum(bool_vec)} [{Outlier_Perc:.2f}%] ')

    def _compute_vertical_distribution(self, data):
        """
        Computs the vertical spacing between zero mean surface and the Foorpint.
        The zero-mean-surface is assumed to be equal the reference surface.
        
        Important: Function requires computing of the reference surface in advance
        
        Input Parameters:
            
            - data['depth']
            - data['ref_depth']
            - data['ping']
        """

        local_ID = '[f207]'
        
        self._log.add(arg1=local_ID,arg2=self.vararg2,
                      arg3='\t\t- Flag values outside vertical distribution intervall...')
        
        #--- Input Arg from Config
        sigma_cap = self._vertical_detection_intervall
        
        
        if sigma_cap is None:
            self._log.add(arg3='--> skipped')
            return
        
        #--- Opperations
        stime= subr.tic()
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        df = pd.DataFrame({'z': data['depth'], 'z_ref':data['ref_depth'], 'group': data['ping']})
        grouped = df.groupby('group', observed=True)
          
        n = len(data['depth'])
        sigma_area_vec = np.full(n, np.nan)  # to store sigma levels for each point
        bool_vec = np.full(n, False)         # to flag outliers
    
        all_sigma_spacings = {i: [] for i in range(1, 7)}  # Store spacings for sigma groups 1-6
   
        # Process each group
        for group_id, group_data in grouped:
            indices = group_data.index
            
            z = group_data['z'].values
            z_ref = group_data['z_ref'].values
            
            # Calculate spacings and statistics
            spacings = z-z_ref
            mean_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)
            
            # Compute thresholds for each sigma level
            sigma_thresholds = mean_spacing + np.arange(1, 7) * std_spacing
            
            # Determine sigma level for each spacing
            abs_diff = np.abs(spacings - mean_spacing).reshape(-1, 1)
            sigma_labels = (abs_diff <= sigma_thresholds).argmax(axis=1) + 1
            sigma_labels[sigma_labels > sigma_cap] = -1  # Cap sigma for outliers
            
            # Assign sigma labels and outlier flags to the main vectors
            sigma_area_vec[indices] = sigma_labels  # Offset for spacings length
            bool_vec[indices] = sigma_labels == -1
    
            # Store spacings by sigma level for mean calculation
            for level in range(1, 7):
                all_sigma_spacings[level].extend(spacings[sigma_labels == level])
     
        
        # Calculate mean distribution for each sigma level
        mean_sigma = [np.mean(all_sigma_spacings[i]) if all_sigma_spacings[i] else np.nan for i in range(1, 7)]
        
        # Calculate outlier stats
        Outlier_Per = np.sum(bool_vec) / n * 100
        Outlier_Nr = np.sum(bool_vec)
        
        
        #--- Add flag to Outlier_df
        N = len(data['depth']) 
        flag= pd.Series(bool_vec, index=data.index) 
        self._add_Outlier('vertical_Distribution', flag, N)
    
        #--- final feedback
        self.applied_Filter.append('vertical_distribution')
        self._log.add(arg3=f'--> success [Elapse Time: {subr.toc(stime)}]')
        self._log.add(arg1=local_ID,arg2=self.vararg2,
        arg3=f'\t\t  Number of excluded Values: {Outlier_Nr} [{Outlier_Per:.2f}%]')
        return mean_sigma
      
    def _add_Outlier(self, var_name , flag, N):
        
        local_ID = '[f208]'
        
        # Validate flag length and type
        # N = length data x,y,z value
        if len(flag) != N:
            raise ValueError(f"- ErrorFlag {local_ID} in add_Outlier fct. The length of Input Data to flag does not match the specified length N.")
        if not pd.api.types.is_bool_dtype(flag):
            raise TypeError(f"- ErrorFlag {local_ID} in add_Outlier fct. Data Input to Outlier Dataframe must be a boolean Series.")

        # Add the flag as a new column
        self.outlier_df[var_name] = flag

        # Update 'total_Outliers' based on any True values in other columns
        self.outlier_df['total_Outliers'] = self.outlier_df.drop(columns=['total_Outliers']).any(axis=1)

    def _save_current_df(self,df,output_fullfile):
        
        local_ID = '[f209]'
        
        self._log.add(arg1= local_ID ,arg2=self.vararg2,
                      arg3='\t\t- Saving processing results to dataframe...')
        
        # store geodataframe in input direcoty as save file
        df.attrs['is_Processed'] = 'True'
        df.to_pickle(output_fullfile)
        
        # do not delete existing datafame. 
        if output_fullfile.endswith(".pkl") and not output_fullfile.endswith("_proc.pkl"):
            updated_fullfile = output_fullfile.replace(".pkl", "_proc.pkl")
            os.rename(output_fullfile, updated_fullfile)
        
        self._log.add(arg3='--> success')

    # <<<<<<<<<<<<<<<<<<<<<< Feedback Functions >>>>>>>>>>>>>>>>>>>>>>>>>>
    def print_Outlier(self):
        
        # Print the sum of True values for each column
        print('Display Number of Outlier per Filter method applied:')
        print(f"\t- Number of Datapoints: {len(self.outlier_df['total_Outliers'])}")
        for column in self.outlier_df.columns:
            print(f"\t- {column}: {self.outlier_df[column].sum()}")

    def get_Outliers(self):
        # Return the total_Outliers column
        return self.outlier_df['total_Outliers']

