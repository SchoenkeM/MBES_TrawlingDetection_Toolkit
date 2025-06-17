
import os
import pyproj
import yaml
from datetime import datetime as dtime

class ConfigHandler:
    def __init__(self, config_file):
        """
        Initialize the configuration from a YAML file.
        Version 1.0.2 

        Version 1.0.3
            - added gridding method as input argument
            
        Version 1.0.4
            - fix spelling error
            - add  Feature Segmentation Threshold to config file
            - remove Buffer ID from config file    
            
        """
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Import Settings
        self._data_tag = config['Import Settings']['Cruise Tag']
        self._data_dir = config['Import Settings']['Current Dir']
        self._data_columns = config['Import Settings']['Column Order']
        self._data_crs = config['Import Settings']['Input crs']
        self._Sonar_Nr_of_Beams = config['Import Settings']['Number of Sonar Beam']
        self.Tile_Size = config['Import Settings']['Tile Size in m']
        
        # Reference Surface
        self._reference_surface = {
            'filter_window_in_perc': config['Refernce Surface']['Filter Window Size in percentage'],
            'min_required_Beams': config['Refernce Surface']['Min Number of Beams required']
        }

        #----
        # make sure input options are valid 
        
        # Filter Settings
        buffer_input = config['Filter Settings']['Exclude Buffer ID']
        if  buffer_input is None: 
            buffer_input = None
        elif len(buffer_input)==0:
            buffer_input = None
        else:
            buffer_input = self.parse_int_list(config['Filter Settings']['Exclude Buffer ID'])
                    
        excl_beam_input = config['Filter Settings']['Exclude Beam Numbers']
        if excl_beam_input is None:
            excl_beam_input = None
        elif len( excl_beam_input)==0:
            excl_beam_input = None
        else:
            excl_beam_input = self.parse_range_string(config['Filter Settings']['Exclude Beam Numbers'])
            
        detec_win_input = config['Filter Settings']['Vertical Detection Window in m']
        if detec_win_input is None:
            detec_win_input = None
        elif len(detec_win_input)==0:
            detec_win_input = None
        else:
            detec_win_input = config['Filter Settings']['Vertical Detection Window in m']
                
        dev_win_input = config['Filter Settings']['Vertical Statistical Deviation Window']
        if dev_win_input is None:
            dev_win_input = None
        elif len([dev_win_input])==0:
            dev_win_input = None
        else:
            dev_win_input = config['Filter Settings']['Vertical Statistical Deviation Window']
            
            
        self._filter_options = {
            'exclude_Beam_Numbers': excl_beam_input,
            'exclude_Buffer_ID': buffer_input,
            'detection_Window_Range': detec_win_input,
            'max_allowed_vertical_distribution':dev_win_input
        }

        # Gridding Settings
        self._gridding_options = {
            'exclude_Outlier': config['Grid Settings']['Exclude detected Outliers'],
            'grid_Resolution_m': config['Grid Settings']['Grid Resolution in m'],
            'gridding_method': config['Grid Settings']['Gridding method'],
            'min_required_Points': config['Grid Settings']['Min number of Points per Tile required'],
            'treat_Tile_Overlaps': config['Grid Settings']['Treat Tile Overlaps']
        }

        feature_thres = config['Export Settings']['Feature Segmentation Threshold in m']
        if feature_thres is None or feature_thres == '' or feature_thres == ' ' or feature_thres == [] or feature_thres == 0:
            feature_thres = None
            
        self._export_options = {
            'feature_threshold_m': feature_thres  
        }
        
        # Generate a configuration filename with timestamp
        if 'Config' in config_file[:-5] or 'config' in config_file[:-5]: 
            self._config_filename = f"{dtime.now().strftime('%Y%m%d-%H%M%S')}_{config_file[:-5]}.txt"
        else:
            self._config_filename = f"{dtime.now().strftime('%Y%m%d-%H%M%S')}_ConfigFile_{config_file[:-5]}.txt"
        self.check_epsg(self._data_crs)

    @staticmethod
    def parse_range_string(range_str):
        """
        Parse a range string into a list of individual integers.
        Example: '0:5, 250:512, 1000' → [0, 1, 2, 3, 4, 5, 250, ..., 512, 1000]
        """
        result = []
        for part in range_str.split(','):
            part = part.strip()
            if ':' in part:
                start, end = map(int, part.split(':'))
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
        return result

    @staticmethod
    def parse_int_list(int_list_str):
        """
        Convert a comma-separated string of integers into a list.
        Example: '-1,-2,2,1' → [-1, -2, 2, 1]
        """
        return [int(x.strip()) for x in int_list_str.split(',')]

    def save_config(self):
        """
        Stores the config file in the working directory.
        """
        path = os.path.join(self._data_dir, 'output')
        os.makedirs(path, exist_ok=True)

        full_path = os.path.join(path, self._config_filename)

        with open(full_path, 'w') as file:
            file.write(f"[_data_dir]: {self._data_dir}\n\n")
            for attr_name, attr_value in self.__dict__.items():
                if isinstance(attr_value, dict):
                    for key, value in attr_value.items():
                        file.write(f"[{attr_name}]: {'.' * (30 - len(attr_name))} '{key}': {value}\n")
                    file.write("\n")  

        print(f"\nConfiguration saved to: {full_path}")

    def print_config(self):
        """
        Prints the content of the config to the command line.
        """
        print('\n' + '_' * 60)
        print('Print Config content...\n')

        print(f"[_data_tag]: {'.' * (30 - len('_data_tag'))} {self._data_tag}")
        print(f"[_data_dir]: {'.' * (30 - len('_data_dir'))} {self._data_dir}")
        print(f"[_data_columns]: {'.' * (30 - len('_data_columns'))} {self._data_columns}")
        print(f"[_data_crs]: {'.' * (30 - len('_data_crs'))} {self._data_crs}")
        print(f"[_Sonar_Nr_of_Beams]: {'.' * (30 - len('_Sonar_Nr_of_Beams'))} {self._Sonar_Nr_of_Beams}")
        print(f"[Tile_Size]: {'.' * (30 - len('Tile_Size'))} {self.Tile_Size}\n")
        print(f"[config filename]: {'.' * (30 - len('config filename'))} {self._config_filename}\n")
        
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    print(f"[{attr_name}]: {'.' * (30 - len(attr_name))} '{key}': {value}")
                print()  

        print('\nEnd of Config')
        print('_' * 60)

    def check_epsg(self, epsg_code):
        """
        Validate the EPSG coordinate reference system.
        """
        if isinstance(epsg_code, int):
            try:
                pyproj.CRS.from_epsg(epsg_code)
                return True  
            except pyproj.exceptions.CRSError:
                print(f'WARNING: Invalid EPSG Code: {epsg_code}')
                print('WARNING: Cannot proceed with invalid georeference system --> Process aborted')
                return False  
        else:
            print(f'WARNING: Invalid EPSG Code: {epsg_code}')
            print('WARNING: Cannot proceed with invalid georeference system --> Process aborted')
            return False  
