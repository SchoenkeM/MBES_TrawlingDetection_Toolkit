# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:19:36 2025

@author: misch
"""

import os
import re
import glob

import numpy as np
import subroutines as subr

import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon

from scipy.interpolate import griddata
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.ndimage import generic_filter

import rasterio
from rasterio.transform import from_origin

from itertools import chain
import fastparquet  # is not direcly used but required for writing *.parquet files

class OutputHandler:
    def __init__(self, log, config):
        """
        Initialize Processing Handler:
        
        Version 1.0.0
            init version

        Version 1.0.1
            - fixed typos  
        Version 1.0.2
            - Implemented that the area coverage of mounds and furrows is tracked
              accodingly
        Version 1.0.3
            - fixed mixed between manusctipt version from local and release version
            - fixed version control number 
            - added area computation to statistic values
            - added point cloud basic statistic to geopackage   
            - remove redundant gridding function "grid_current_group"
            - rename griddig function "_get_single_grid" to" to  "_get_tile_grid" 
        Version 1.0.4
            - added "gridding method" as config input paramter 
            - added check to make sure valid gridding method is selected as input option
        Version 1.0.5
            - Refactored OutputHandler to import threshold from config instead of passing it as an input argument.
            - removed feature option form "export_geotiff" and adapted promp msg accordingly
            - Refactored export_statisics according to the changes concenring import threshold from config     
        Version 1.0.5.1
            - Bug fixed due to Typos
            - Removed Buffer ID Filter, by commenting relevant lines. It is computed, but not used in filter opperation
            - Adding "No Data Values" to Config
        Version 1.0.6 
            - add filter function to interpolate outliers by nearest neighbor median that occur after the gridden 
            - add a threshold value, that only objects consisting of at least 2 pixels are taken into account
        Version 1.0.6.1
            - fix bug in median_filter function. Causing crash if size detection window is > 2     
            
        """
        
        version = 'v.1.0.6'

        self._log = log
        self.vararg2 = 'Output Handler'
        self._log.add(arg1='[init]',arg2=self.vararg2,arg3=[f'Initialize... {version}'])

        # Get parameter from config
        self._suffix = config._data_tag
        self._fdir = config._data_dir

        self._grid_space = config._gridding_options['grid_Resolution_m']
        self._grid_method= config._gridding_options['gridding_method']
        self._min_points_4_gridding = config._gridding_options['min_required_Points']
        self._tile_size = config.Tile_Size
        self._stack_method = config._gridding_options['treat_Tile_Overlaps']
        self._vertical_detection_range = config._filter_options['detection_Window_Range']
        
        self._depth_threshold = config._export_options['feature_threshold_m']
        self._no_data_value = config._export_options['no_data_value']
        
        # init parameter
        self._EU_Grid_crs = 3035
        self.tile_gdf = []

        # set defaults path
        self._grid_dir = self._fdir  + '/tiles'
        self._output_dir = self._fdir  + '/output'
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._grid_dir, exist_ok=True)

        self._log.add(arg3='--> success')


    def compile_tiles(self):

        self.arg1_ID  = '[f300]'

        def get_proc_pkl_filelist(fdir):

            local_arg1_ID = '[f301]'
            self._log.add(arg1=local_arg1_ID , arg2=self.vararg2,
                          arg3="\t- Get Filelist of all processed *_proc.pkl files")

            pkl_filelist=[]
            pkl_filelist = {f for f in list(os.listdir(fdir)) if f.endswith("_proc.pkl")}

            # take care of possible dublicats
            pkl_filelist = list(pkl_filelist)

            self._log.add(arg1=local_arg1_ID , arg2=self.vararg2,
                          arg3= f"\t- A total of ({len(pkl_filelist)}) *.pkl-file(s) located" )
            return pkl_filelist

        def delete_parquet_files(folder_path):
            # List all files in the folder

            local_arg1_ID = '[f302]'
            self._log.add(arg1= local_arg1_ID, arg2=self.vararg2,
                          arg3= f'\t- Deleted ".parquet" files in dir: {folder_path}')

            files = glob.glob(os.path.join(folder_path, "*.parquet"))

            # Loop through each file and remove it
            for file in files:
                if not file.endswith("gridded.parquet"):
                    try:
                        os.remove(file)
                    except Exception as e:
                            self._log.add(arg1= local_arg1_ID, arg2=self.vararg2,
                                          arg3=f'\t\t-> Failed to delete {file}: {e}')

            self._log.add(arg3= "--> [completed]" )

        def gather_tile_data(grid_dir, pkl_filelist):
            """
            Process a list of GeoDataFrames: group by 'tile_id', filter, and write to files.
            """
            local_arg1_ID = '[f303]'

            self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                          arg3= '\t- Gathering of Tile data across datasets in process...' )

            len_filelist = len(pkl_filelist)
            run = 1
            columns2keep=["x","y","depth","ref_depth","buffer_id"] # for .parquet data
            for file in pkl_filelist:
                # Ensure it's a GeoDataFrame

                # import data
                self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                           arg3=[f'\t\t- Import File({run}/{len_filelist}): {file}...'])

                input_file_path= os.path.join(self._fdir, file)
                df = pd.read_pickle(input_file_path)
                self._log.add(arg3='--> [success]')

                # filter data
                data_to_group = self._get_filtered_df(df)

                # init for loop for all Tiles in a file
                len_tilelist= len(set(data_to_group['tile_id']))
                counter= 0
                self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                           arg3=['\t\t\t- Group and write Tile data to temp. file(s)... '])

                # Group by 'tile_id' and process each group
                for tile_id, group in data_to_group.groupby("tile_id", observed=True):
                    output_file_path = os.path.join(grid_dir, f"{tile_id}.parquet")

                    data_to_save = group[columns2keep].copy()
                    data_to_save["fid"] = 1

                    #---
                    # Append or create a new Parquet file
                    if os.path.exists(output_file_path):
                        try:
                            existing_data = pd.read_parquet(output_file_path)

                            # Get the current max 'fid' or use 0 if not present
                            max_fid = existing_data["fid"].max() if "fid" in existing_data.columns else 0

                            # Increment 'fid' for the new data
                            data_to_save["fid"] = max_fid + 1

                            combined_data = pd.concat([existing_data, data_to_save], ignore_index=True)
                            combined_data.to_parquet(output_file_path, index=False)
                        except:
                            print("\nERROR: Invalid dataframe format. Can not add df data to existing parquet file")

                    else:
                        try:
                            #data_to_save = data_to_save.map(lambda x: int(x) if isinstance(x, np.int64) else x)
                            data_to_save.to_parquet(output_file_path, index=False)
                        except:
                            print("\nERROR: Invalid dataframe format. Can not convert df to empty parquet file")

                    if counter % max(1, len_tilelist // 15) == 0:
                        print('#', end='', flush=True)
                    counter +=1

                self._log.add(arg3=' [done]')
                run += 1

            self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                          arg3= '\t- Gathering of Tile data completed' )

        def verbose_completed(import_runtime):
            elapseTime = subr.toc(import_runtime)
            self._log.add(arg1=self.arg1_ID,arg2=self.vararg2,
                          arg3=f'\t- Compilation of tile data across datasets completed. Elapse time: {elapseTime}')

        def function_handler():
            """
            Input Arguements:
                self._grid_dir: defines the direcory where the processed
                                *_proc.pkl is stored
            -------
            Creates multiple *.parquet files in path "_grid_dir"
            """
            print('\n#---',  end='', flush=True)
            import_runtime = subr.tic()

            rdir      = self._fdir #reading directory
            wdir      = self._grid_dir #writing directory

            #print('\n#---',  end='', flush=True)
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=['Compilation of Tile Data across all Profile in process...'])

            # get file list of all files in dataframe
            pkl_filelist = get_proc_pkl_filelist(rdir)

            # delete existing data
            delete_parquet_files(wdir)

            # loop over all files to match footpint to tiles
            gather_tile_data(wdir, pkl_filelist)

            verbose_completed(import_runtime)
        #--- function calls
        function_handler()

    def grid_tiles(self):

        self.arg1_ID = '[f400]'

        
        def median_filter(grid_values):
            # Compute median of neighbors for each cell
            if self._vertical_detection_range is None: 
                return grid_values
            
            mask = (grid_values > max(self._vertical_detection_range)) | (grid_values< min(self._vertical_detection_range))           
            if not np.any(mask):
                return grid_values
            
            median_grid = generic_filter(grid_values, self._median_surrounding, size=3, mode='nearest')  
            filtered_grid = np.where(mask, median_grid, grid_values)
            return np.median(filtered_grid)

        def verbose_parameters(threshold):
            # Feedback for input option here
            if threshold < 100:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                      arg3=[f'\t- WARNING: current Threshold of {threshold} set very low.'])

            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                        arg3=[f'\t- Tiles with number of data points < {threshold}, will be ignored. Threshold defined in config.'])

            # print grid option feedback
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                        arg3=[f"\t- Gridding method selected: '{self._grid_method}'. Default is 'linear'."])

            if self._grid_method == 'cubic':
                txt= "\t- Using 'cubic' interpolation for smooth surface generation. Requires sufficient data points and may produce NaNs at boundaries."
            elif self._grid_method == 'linear':
                txt= "\t- Using 'linear' interpolation for moderate accuracy with better boundary handling."
            elif self._grid_method == 'nearest':
                txt= "\t- Using 'nearest' interpolation, which assigns values from the closest data point. Fastest but less smooth."
            else:
                txt= f"\t- Warning: '{self._grid_method}' is not a standard method. Ensure it is a valid scipy.griddata method."
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=[txt])

        def get_parquet_filelist(fdir):
            local_arg1_ID = '[f401]'
            self._log.add(arg1= local_arg1_ID ,arg2=self.vararg2,arg3=['\t- Create filelist of *.parquet-files from input directory...'])

            all_files = set(os.listdir(fdir))  # Load all filenames into a set for fast lookup
            parquet_filelist = [f.replace(".parquet","") for f in all_files if f.endswith('.parquet')]  # Find all .txt files

            self._log.add(arg3='--> [success]')

            self._log.add(arg1= local_arg1_ID ,arg2=self.vararg2,arg3=[f'\t- ({len(parquet_filelist)}) parquet-files found'])

            return parquet_filelist

        def check_filelist(filelist):
            local_arg1_ID = '[f402]'
            if len(filelist)>0:
                return True
            else:
                self._log.add(arg3='--> [failed]')

                self._log.add(arg3='gridding process aborted')
                self._log.add(arg1=local_arg1_ID ,arg2=self.vararg2,
                              arg3=['\t- No *.parquet- files located'])
                self._log.add(arg1=local_arg1_ID ,arg2=self.vararg2,
                              arg3=['\t- If not done jet, use the ".compile_tiles" method first'])
                return False

        def parse_local_coordinates(expression, dx):

            if '.parquet' in expression:
                expression.replace(".parquet","")

            # If input is a single string, wrap it in a list
            if isinstance(expression, str):
                expression = [expression]

            # Ensure input is a numpy array
            expressions = np.array(expression, dtype=str)

            # Initialize lists to store results
            tile_sizes = []
            xcoords = []
            ycoords = []

            for expr in expressions:
                match = re.match(r'(\d+km|\d+m)E(\d+)N(\d+)', expr)
                #match = re.match(r'^(\d+km|\d+m)E-?\d+N-?\d+$', expr)
                if not match:
                    raise ValueError(f"Invalid format: {expr}")

                tile_str, x_str, y_str = match.groups()

                # Determine tile size
                TileSize = 10000 if 'km' in tile_str else 10

                # Convert coordinates
                xcoord = int(x_str) * TileSize
                ycoord = int(y_str) * TileSize

                tile_sizes.append(TileSize)
                xcoords.append(xcoord)
                ycoords.append(ycoord)

            # Return scalars for a single input, arrays for multiple
            if len(tile_sizes) == 1:
                N = int(np.array(tile_sizes)[0] / dx)

                xx = np.linspace(np.array(xcoords), np.array(xcoords) + np.array(tile_sizes)-dx, N)
                yy = np.linspace(np.array(ycoords), np.array(ycoords) + np.array(tile_sizes)-dx, N)

                xx_grid, yy_grid = np.meshgrid(xx, yy)

                return xx_grid, yy_grid, tile_sizes[0]
            else:
                print('Enexprected Error by parsing parquet file name to coordinates')

        def get_buffer_id(buffer_series):
            """
            Finds the smallest positive buffer index and the largest negative buffer index.

            Parameters:
            buffer_series : pd.Series
                A Pandas Series containing buffer ID values (e.g., from -5 to 5).

            Returns:
            tuple(int, int)
                Smallest positive buffer ID and largest negative buffer ID.
                Returns (None, None) if no valid values exist.
            """
            # Filter for positive and negative values separately
            positive_values = buffer_series[buffer_series > 0]
            negative_values = buffer_series[buffer_series < 0]

            # Get the smallest positive value and the largest negative value
            smallest_positive = positive_values.min() if not positive_values.empty else None
            largest_negative = negative_values.max() if not negative_values.empty else None

            output = np.array([val for val in [largest_negative, smallest_positive] if val is not None])
            return output

        def loop_over_parquets(wdir, parquet_filelist, dx, threshold):
            """
            Loop over all parquet files, grid the data, and construct a DataFrame
            with all results instead of saving individual parquet files.

            dx: Input gridding space
            threshold: Min number of points required for gridding
            """
            local_arg1_ID = '[f403]'

            total = 15  # Number of progress updates
            self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                          arg3=['\t- Start loop over *.parquet-files for gridding: '])

            # Prepare storage for all results
            all_results = []

            N = len(parquet_filelist)
            counter = 0
            is_valid = 0
            is_ignored = 0
            tile_std = []

            for file_name in parquet_filelist:
                # Get grid coordinates from file name
                xx_grid, yy_grid, tile_size = parse_local_coordinates(file_name, dx)

                # Load data
                fullfile = os.path.join(wdir, f"{file_name}.parquet")

                try:
                    data = pd.read_parquet(fullfile)
                except Exception as e:
                    self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                                  arg3=f'\t\t-> Failed to read {file_name}: {e}')
                    continue

                # Check if tile has enough points
                if len(data) > threshold:
                    is_valid += 1
                else:
                    is_ignored += 1
                    if counter % max(1, N // total) == 0:
                        print('#', end='', flush=True)
                    counter += 1
                    continue

                coverage = self._get_pt_coverage_perc(data, dx, tile_size)
                nni_out = self._get_nearest_neighbor_index(data, tile_size)
                z_pt_min, z_pt_max, z_pt_std = self._get_basic_pt_stats(data)

                # Compute tile grid
                if max(data["fid"]) == 1:
                    z_grid = self._get_tile_grid(data, xx_grid, yy_grid)
                else:
                    z_grid = self._get_average_grids(data, xx_grid, yy_grid)

                # ensure no data outside detection window is passed to grid
                z_grid = median_filter(z_grid)
                
                # Compute metadata
                rq, ra = self._get_surface_roughness(z_grid)
                valid_values = len(z_grid[~np.isnan(z_grid)])
                z_grid_min, z_grid_max, z_grid_std = self._get_basic_grid_stats(z_grid)
                tile_std.append(z_grid_std)
                buffer_id = get_buffer_id(data['buffer_id'])[0]

                if max(data["fid"]) == 1:
                    is_overlap = False
                else:
                    is_overlap = True

                # Store results in a dictionary
                tile_result = {
                    "tile_id": file_name,
                    "xx": [xx_grid[0,:].flatten().tolist()][0],  # Store as list
                    "yy": [yy_grid[:,0].flatten().tolist()][0],  # Store as list
                    "zz": [z_grid.flatten().tolist()],   # Store as list
                    "grid_size": [xx_grid.shape[0], xx_grid.shape[1]],
                    "is_overlap": is_overlap,
                    "buffer_id": buffer_id,
                    "pt_per_tile": len(data),
                    "nni": nni_out,
                    "pt_area_coverage_perc": coverage,
                    "z_pt_min": z_pt_min,
                    "z_pt_max": z_pt_max,
                    "z_pt_std": z_pt_std,
                    "z_grid_nr_not_nan": valid_values,
                    "z_grid_min": z_grid_min,
                    "z_grid_max": z_grid_max,
                    "z_grid_std": z_grid_std,
                    "z_grid_rq": rq,
                    "z_grid_ra": ra
                }

                # Append to list
                all_results.append(tile_result)

                if counter % max(1, N // total) == 0:
                    print('#', end='', flush=True)
                counter += 1

            # Convert list of dictionaries to a DataFrame
            tile_df = pd.DataFrame(all_results)
            #self.tile_gdf .info(memory_usage="deep")
            global_z_std = np.nanmean(tile_std)

            self._log.add(arg3=' [done]')
            self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                          arg3=[f'\t- {is_valid}/{N} tiles were gridded. {is_ignored} were ignored'])

            self._log.add(arg1=local_arg1_ID ,arg2=self.vararg2,
                          arg3=[f'\t- Mean standard variation over all tile: {np.round(global_z_std,3)} m'])

            return tile_df

        def function_handler():

            print('\n#---',  end='', flush=True)
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=['Gridding of Tile data in process...'])


            # parameter form config
            threshold = self._min_points_4_gridding
            dx        = self._grid_space
            wdir      = self._grid_dir
            Data_Tag  = self._suffix

            output_fullfile_pkl = os.path.join(wdir, f"{Data_Tag}_griddata.pkl")

            # check if stacking is valid
            options = ["first", "last", "average"]
            if not self._stack_method in options:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                              arg3=[f'\t- {self._stack_method} is not a valid gridding stacking option. Process aborted'])
                return

            options = ["cubic","linear","nearest"]
            if not  self._grid_method in options:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                              arg3=[f'\t- {self._grid_method} is not a valid gridding method option. Process aborted'])
                return
            
            stime = subr.tic()
            #---
            verbose_parameters(threshold)

            # ---
            # returns Seires with parqwut file names to grid, which have not
            # been gridded yet
            parquet_filelist = get_parquet_filelist(wdir)

            #---
            # make sure file_list is not empty
            if not check_filelist(parquet_filelist):
                if os.path.isfile(output_fullfile_pkl):
                    os.remove(output_fullfile_pkl)
                return

            #---
            # get global grid bounds
            # parquet_filelist contsint ending with .parquet and
            tile_gdf = loop_over_parquets(wdir, parquet_filelist, dx, threshold)

            tile_gdf = self._get_tile_gdf(tile_gdf)

            self._save_current_df(tile_gdf,output_fullfile_pkl)

            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                          arg3=[f'\t> Gridding of tile data completed [Elapse Time: {subr.toc(stime)}]'])

        function_handler()

    def export_geotiff(self, crs = 3035):
        """
        Export Options
        ----------
        nodata value to manual define alpha channel  
        

        Returns
        -------
        Export grid to /temp folder

        """
        
        self.arg1_ID = '[f500]'

        def get_global_bounds(tile_gdf, dx):

            local_arg1_ID = '[f501]'

            self._log.add(arg1=local_arg1_ID ,arg2=self.vararg2,arg3=['\t- Create global Grid to insert gridded tile data...'])

            data =  tile_gdf.bounds
            x_min= min(data['minx'])
            x_max= max(data['maxx'])

            y_min= min(data['miny'])
            y_max= max(data['maxy'])

            xx_bounds= [x_min, x_max]
            yy_bounds= [y_min, y_max]

            self._log.add(arg3='--> [success]')
            size_of_x_vec = len(np.arange(x_min, x_max, dx))
            self._log.add(arg3=f'Nr. of x-values: {size_of_x_vec}')

            size_of_y_vec = len(np.arange(y_min, y_max, dx))
            self._log.add(arg3=f'Nr. of y-values: {size_of_y_vec}')
            return xx_bounds, yy_bounds

        def local2globe(xx_global,yy_global,xx_grid,yy_grid, z_grid):

            # Merge gridded area into the global sparse matrix
            x_start_idx = np.searchsorted(xx_global, xx_grid[0][0])
            y_start_idx = np.searchsorted(yy_global, yy_grid[0][0])

            # Prepare indices and values for the sparse update
            local_rows, local_cols = np.indices(z_grid.shape)

            global_rows = y_start_idx + local_rows.flatten()
            global_cols = x_start_idx + local_cols.flatten()
            valid_mask = ~np.isnan(z_grid.flatten())  # Ignore NaN values
            global_rows = global_rows[valid_mask]
            global_cols = global_cols[valid_mask]
            values = z_grid.flatten()[valid_mask]

            return global_rows,global_cols,values

        def loop_over_df(tile_gdf, dx):

            local_arg1_ID = '[f502]'
            [xx_bounds, yy_bounds] = get_global_bounds(tile_gdf, dx)

            #---
            # Prepare global CRS sparse matrix
            xx_global = np.arange(xx_bounds[0], xx_bounds[1] + dx, dx)
            yy_global = np.arange(yy_bounds[0], yy_bounds[1] + dx, dx)
            nrows, ncols = len(yy_global), len(xx_global)

            # COO format for incremental updates. Coo_matrix is a sparse
            # matrix with coordinates for faster handling
            global_matrix_sparse = coo_matrix((nrows, ncols))

            #---
            self._log.add(arg1= local_arg1_ID ,arg2=self.vararg2,arg3=['\t- Embedding local tile surface data in a global grid:'])

            N = len(tile_gdf)
            counter = 0

            for index, row in tile_gdf.iterrows():

                #convert vector to grid
                xx_grid, yy_grid = np.meshgrid(row["xx"], row["yy"])
                z_grid = np.array(row["zz"][0]).reshape(row["grid_size"][0], row["grid_size"][1])

                #---
                # match local grid values with global sparse matrix
                [global_rows,global_cols,values] = local2globe(xx_global,
                                                               yy_global,
                                                               xx_grid,
                                                               yy_grid,
                                                               z_grid)
                # Update sparse matrix
                global_matrix_sparse += coo_matrix(
                    (values, (global_rows, global_cols)), shape=(nrows, ncols)
                )

                if counter % max(0, N // 15) == 0:
                    print('#', end='', flush=True)
                counter +=1

            # Convert to CSR for efficient storage and computation
            global_matrix_csr = global_matrix_sparse.tocsr()

            self._log.add(arg1= local_arg1_ID ,arg2=self.vararg2,
                          arg3=['\t- Merging of gridded tile data completed '])

            return global_matrix_csr, xx_global, yy_global

        def matrix_to_geotiff(fdir, xx_global, yy_global, global_matrix_csr, crs):
            """
            Exports a sparse CSR matrix as a GeoTIFF file with a given CRS.
            Parameters:
                global_matrix_csr: scipy.sparse.csr_matrix
                    The sparse matrix to be exported.
                xx_global: ndarray
                    The 1D array of x-coordinates (global grid).
                yy_global: ndarray
                    The 1D array of y-coordinates (global grid).
                output_path: str
                    Path to save the GeoTIFF file.
                crs: int
                    EPSG code for the coordinate reference system (default: 3035).
            """

            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                          arg3=['\t- Building of Geotif in process...'])


            # Convert the CSR matrix to a dense array
            global_matrix_dense = global_matrix_csr.toarray()

            # **Identify unset values in the CSR matrix**
            mask_unset = global_matrix_csr.tocoo().toarray() == 0  # True for missing values

            # Apply NoData value only to originally unset elements
            global_matrix_dense[mask_unset] = self._no_data_value

            # the coordinate system anchor point is defined bot-left, but
            # the geofiff functio expects top-left, so the matrix needs to be
            # flipped
            global_matrix_dense = np.flipud(global_matrix_dense)

            # Calculate the resolution of the grid
            x_res = (xx_global.max() - xx_global.min()) / (len(xx_global) - 1)
            y_res = (yy_global.max() - yy_global.min()) / (len(yy_global) - 1)

            # Create a rasterio transform (origin is the top-left corner)
            transform = from_origin(xx_global.min(), yy_global.max(), x_res, y_res)

            # Define metadata for the GeoTIFF
            raster_meta = {
                "driver": "GTiff",
                "height": global_matrix_dense.shape[0],
                "width": global_matrix_dense.shape[1],
                "count": 1,  # Single band
                "dtype": global_matrix_dense.dtype,
                "crs": f"EPSG:{crs}",  # Coordinate Reference System
                "transform": transform,
                "nodata":  self._no_data_value  # Set NoData value for transparency
            }

            time_str = subr.time_stamp()
            output_path = fdir  + f"/{self._suffix}_residual_bathymetry_{time_str}.tif"

            # Write the matrix to a GeoTIFF file
            with rasterio.open(output_path, "w", **raster_meta) as dst:
                dst.write(global_matrix_dense, 1)  # Write the dense matrix to band 1

            self._log.add(arg3='--> [completed]')

            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                          arg3=[f'\t- GeoTIFF saved at {output_path}'])

        def function_handler():

            # global variables defined in input with default values:
            # crs

            print('\n#---',  end='', flush=True)
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=['Export Tile data to Geotiff in process...'])

            # parameter form config

            output_dir = self._output_dir
            dx         = self._grid_space
            wdir       = self._grid_dir
            Data_Tag   = self._suffix
            
            input_fullfile_pkl = os.path.join(wdir, f"{Data_Tag}_griddata.pkl")

            if not os.path.isfile( input_fullfile_pkl):
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- WARNING: No gridded Tile Dataset. use ".grid_tiles" method first.')
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- process aborded')
                return

            tile_gdf = pd.read_pickle(input_fullfile_pkl)

            #---
            if len(tile_gdf)==0:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- WARNING: Dataset is empty. No "gridded tile" data found.')
                return

            #---
            # loop over tile grid to create global spare matrix
            stime = subr.tic()
            global_matrix_csr, xx_global, yy_global = loop_over_df(tile_gdf, dx)

            # convert global spare matrix to grid
            matrix_to_geotiff(output_dir, xx_global, yy_global, global_matrix_csr, crs)

            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=f'\t> Export GeoTIF completed [Elapse Time: {subr.toc(stime)}]')

        function_handler()

    def export_statisics(self):

        self.arg1_ID = '[f600]'

        def sum_below_threshold(z_grid, dx, threshold ):
            flat_values = list(chain.from_iterable(z_grid)) # Flatten nested list
            below_threshold = [x for x in flat_values if x < -threshold]  # Filter values below threshold
            if len(below_threshold)<=2:
                volume = 0
            else:  
                volume = sum(below_threshold) * dx *dx
            return volume  # Sum them

        def sum_above_threshold(z_grid, dx, threshold):
            flat_values = list(chain.from_iterable(z_grid))  # Flatten nested list
            above_threshold = [x for x in flat_values if x > threshold]  # Filter values below threshold
            if len(above_threshold)<=2:
                volume = 0
            else:          
                volume = sum(above_threshold) * dx *dx
            return volume  # Sum them

        def area_below_threshold(z_grid, dx, threshold ):
            flat_values = list(chain.from_iterable(z_grid)) # Flatten nested list
            below_threshold = [x for x in flat_values if x < -threshold]  # Filter values below threshold
            if len(below_threshold)<=2:
                area = 0
            else:          
                area = len(below_threshold) * dx *dx
            return area  # Sum them

        def area_above_threshold(z_grid, dx, threshold ):
            flat_values = list(chain.from_iterable(z_grid)) # Flatten nested list
            above_threshold = [x for x in flat_values if x > threshold]  # Filter values below threshold
            area = len(above_threshold) * dx *dx
            if len(above_threshold)<=2:
                area = 0
            else:          
                area = len(above_threshold) * dx *dx
            return area  # Sum them

        def function_handler():

            print('\n#---',  end='', flush=True)
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=['Export Statistic as Geopackage in process...'])

            # parameter form config
            dx         = self._grid_space
            wdir       = self._grid_dir
            Data_Tag   = self._suffix
            threshold  = self._depth_threshold
            
            input_fullfile_pkl = os.path.join(wdir, f"{Data_Tag}_griddata.pkl")

            if not os.path.isfile( input_fullfile_pkl):
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- WARNING: No gridded Tile Dataset. use ".grid_tiles" method first.')
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- process aborded')
                return
            else:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- Import of gridded Tile Dataset in process.')
                tile_gdf = pd.read_pickle(input_fullfile_pkl)
                self._log.add(arg3='--> [succsess]')
            #---
            if len(tile_gdf)==0:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- WARNING: Dataset is empty. No "gridded tile" data found.')
                return
            else:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=f'\t- ({len(tile_gdf)}) Tiles detected in Dataset.')


            stime = subr.tic()
            if threshold is None:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- WARNING: No threshold selected. No Trawling Index will be computed .')
            else:
                self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3=f'\t- Threshold +/- {threshold}m selected. Compute trawling intensity in process...')
                threshold = abs(threshold)
                tile_gdf["Furrow_Volume"] = round(tile_gdf["zz"].apply(lambda x: sum_below_threshold(x,dx,threshold)),3)
                tile_gdf["Mound_Volume"]  = round(tile_gdf["zz"].apply(lambda x: sum_above_threshold(x,dx,threshold)),3)
                tile_gdf["Furrow_area"]   = round(tile_gdf["zz"].apply(lambda x: area_below_threshold(x,dx,threshold)),3)
                tile_gdf["Mound_area"]    = round(tile_gdf["zz"].apply(lambda x: area_above_threshold(x,dx,threshold)),3)

                tile_gdf["Furrow_Mound_diff"] = round((tile_gdf["Mound_Volume"] + tile_gdf["Furrow_Volume"]),3)
                self._log.add(arg3='--> [success]')

            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,arg3='\t- Preparing the dataframe for conversion to geopackage...')
            current_columns = list(tile_gdf.keys())

            # Define columns to remove
            columns_to_remove = [ 'xx', 'yy', 'zz','grid_size']

            # Remove unwanted columns
            columns_to_keep = [col for col in current_columns if col not in columns_to_remove]

            tile_gdf=tile_gdf[columns_to_keep]
            self._log.add(arg3='--> [done]')
            #---
            # Save as geopackage
            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                          arg3=['\t- Create Geopackage in process...'])

            time_str = subr.time_stamp()
            output_path_gpkg = self._fdir  + f"/output/{self._suffix}_statistics_{time_str}.gpkg"
            output_path_pkl = self._fdir  + f"/output/{self._suffix}_statistics_{time_str}.pkl"
            tile_gdf.to_file(output_path_gpkg,  driver="GPKG")
            tile_gdf.to_pickle(output_path_pkl)

            self._log.add(arg3='--> [succsess]')

            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                          arg3=[f'\t- Geopackge saved at: {output_path_gpkg}'])

            #---
            # loop over tile grid to create global spare matrix
            self.tile_gdf = []
            self.tile_gdf = tile_gdf


            self._log.add(arg1=self.arg1_ID ,arg2=self.vararg2,
                          arg3=f'\t> Export Statistic as Geopackage completed [Elapse Time: {subr.toc(stime)}]')

        function_handler()

    #--------------------------------------------------------------------------
    # Statistic functions

    def _median_surrounding(self, z_grid):
        """
        Remove the center value and compute median of the 8 neighbors
        """
        neighbors = np.delete(z_grid, len(z_grid) // 2)
        return np.median(neighbors)

    def _get_filtered_df(self, df):
        """
        Removes outliers and prepares a DataFrame for Parquet
        storage efficiently.
        """

        local_arg1_ID = '[f700]'

        self._log.add(arg1=local_arg1_ID, arg2=self.vararg2,
                      arg3=['\t\t\t- Remove outliers and extract relevant columns with np.int64 conversion... '])

        # Step 1: Remove outliers efficiently
        df = df.loc[~df["outliers"]].reset_index(drop=True)

        # Step 2: Create a filtered DataFrame using NumPy arrays for performance
        filtered_df = pd.DataFrame({
            'tile_id': df['tile_id'].to_numpy(),
            'y': df['coords'].y.to_numpy().astype(np.float64),
            'x': df['coords'].x.to_numpy().astype(np.float64),
            'depth': df['depth'].to_numpy().astype(np.float64),
            'ref_depth': df['ref_depth'].to_numpy().astype(np.float64),
            'buffer_id': df['buffer_id'].to_numpy().astype(int)
        })

        self._log.add(arg3=f" [{len(set(filtered_df['tile_id']))} Tiles detected]")

        return filtered_df

    def _get_nearest_neighbor_index(self, data, tile_length):
        """
        Computes the Nearest Neighbor Index (NNI) for a given set of points i
        n a 10x10m tile.

        Parameters:
        data[['x', 'y']]: Nx2 array of (x, y) coordinates.
        tile_length (float): Side length of the square area in meters
        (default: 10m x 10m).

        Returns:
        nni (float): Nearest Neighbor Index.

        NNI < 1 → Points are clustered
        NNI ≈ 1 → Points are randomly distributed
        NNI > 1 → Points are evenly spaced (regular)
        """

        points = data[['x', 'y']].to_numpy()
        N = len(points)

        if N < 2:
            return np.nan  # Not enough points for calculation

        # Compute observed mean nearest neighbor distance
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)  # k=2 because k=1 is the point itself
        d_o = np.mean(distances[:, 1])  # Exclude self-distance

        # Compute expected mean distance for a random distribution
        A = tile_length ** 2  # 10m x 10m
        d_e = 0.5 / np.sqrt(N / A)

        # Compute NNI
        nni = d_o / d_e

        # round
        nni = round(nni,2)

        return nni

    def _get_pt_coverage_perc(self, data, dx, tile_size):

        # can be computed in the Fly based on the coords
        """
        Computes the percentage of the tile covered by the point cloud in a
        e.g. 10x10m area with 0.25m grid.

        Parameters:
        points (ndarray): Nx2 array of (x, y) point coordinates.
        dx (float): Resolution of the grid in meters (default: 0.25m).
        tile_size (float): Side length of the square area in meters (default: 10m).

        Returns:
        coverage_percent (float): Percentage of area covered by point cloud.
        grid_coverage (2D array): Binary matrix representing occupied grid cells.
        """

        points = data[['x', 'y']].to_numpy()
        N = len(points)

        if N < 2:
            return np.nan  # Not enough points for calculation


        points[:, 0] = points[:, 0]-min(points[:, 0])
        points[:, 1] = points[:, 1]-min(points[:, 1])

        # Create a 40x40 grid (10m / 0.25m)
        num_cells = int(tile_size / dx)
        grid = np.zeros((num_cells, num_cells), dtype=int)  # 0: empty, 1: occupied

        # Normalize point coordinates to grid indices
        x_indices = ((points[:, 0]) / dx).astype(int)
        y_indices = ((points[:, 1]) / dx).astype(int)

        # Ensure indices are within bounds
        valid_mask = (x_indices >= 0) & (x_indices < num_cells) & (y_indices >= 0) & (y_indices < num_cells)
        x_indices, y_indices = x_indices[valid_mask], y_indices[valid_mask]

        # Mark occupied grid cells
        grid[x_indices, y_indices] = 1

        # Compute coverage percentage
        total_cells = num_cells * num_cells
        occupied_cells = np.count_nonzero(grid)
        coverage_percent = (occupied_cells / total_cells) * 100

        #self._log.add(arg3=f"Point Cloud Coverage: {coverage_percent:.2f}%")

        coverage_percent= round(coverage_percent,2)

        return coverage_percent #, grid

    def _get_basic_grid_stats(self, z_grid):

        # Remove NaN values
        valid_values = z_grid[~np.isnan(z_grid)]

        if len(valid_values) == 0:
            return np.nan, np.nan, np.nan

        # values rounded to mm
        z_grid_min = round(min(valid_values),4)
        z_grid_max = round(max(valid_values),4)
        z_grid_std = round(np.std(valid_values),4)

        return  z_grid_min,  z_grid_max,  z_grid_std

    def _get_basic_pt_stats(self, data):
        z = (data["depth"] - data["ref_depth"]).values
        z = z - np.nanmean(z)

        # Remove NaN values
        valid_values = z[~np.isnan(z)]

        if len(valid_values) == 0:
            return np.nan, np.nan, np.nan

        # values rounded to mm
        z_min = round(min(valid_values),4)
        z_max = round(max(valid_values),4)
        z_std = round(np.std(valid_values),4)

        return  z_min,  z_max,  z_std

    def _get_surface_roughness(self, z_grid):
        """
        Computes the surface roughness of a zero-mean gridded surface.

        Parameters:
            grid (np.ndarray): 2D NumPy array representing the gridded surface (can contain NaNs).

        Returns:
            roughness_metrics (dict): Dictionary containing Rq, Ra, and variance.
        """

        # Remove NaN values
        valid_values = z_grid[~np.isnan(z_grid)]

        if len(valid_values) == 0:
            return np.nan, np.nan

        # Compute roughness metrics
        Rq = np.sqrt(np.mean(valid_values**2))  # Root Mean Square Height
        Ra = np.mean(np.abs(valid_values))      # Mean Absolute Deviation
        #variance = np.var(valid_values)         # Variance-based roughness

        # values rounded to mm
        if (Rq is not np.nan) and np.isfinite(Rq):
            Rq = round(Rq,4)
        else:
            Rq =  np.nan # error code

        if Ra is not np.nan and np.isfinite(Rq):
            Ra = round(Ra,4)
        else:
            Ra =  np.nan # error code
        return Rq, Ra

    def _get_average_grids(self, data, xx_grid, yy_grid, min_points=100):
        """
        Parameters
        ----------
        data : DataFrame
            Contains x, y, z point cloud data from MBES footprint.
        xx_grid : ndarray
            X coordinates of the current tile.
        yy_grid : ndarray
            Y coordinates of the current tile.

        min_points : int, Local threshold not set in config.
            Minimum number of points per 'fid' to be considered. Default is 10.

        Returns
        -------
        grid : ndarray
            Processed gridded surface based on the selected gridding method.
        """

        def rm_goups_with_low_coverage(data,  min_points):
            # Remove groups with fewer than min_points
            fid_counts = data['fid'].value_counts()
            valid_fids = fid_counts[fid_counts >= min_points].index
            filtered_data = data[data['fid'].isin(valid_fids)].copy()
            return filtered_data

        # Get the stacking method ('average', 'first', or 'last')
        stacking_method =  self._stack_method

        # Remove groups with fewer than min_points
        filtered_data = rm_goups_with_low_coverage(data,  min_points)

        # Initialize accumulation grids
        stack_mask = np.full_like(xx_grid, 0, dtype=float) # mask to stack datasets values in Tile
        nan_mask   = np.full_like(xx_grid, True, dtype=bool) # mask for nan values in Tile
        count_grid = np.zeros_like(xx_grid, dtype=int) # counter for datasets in Tile

        # --- Iterate through groups ---
        for fid, group in filtered_data.groupby('fid'):

            z_grid = self._get_tile_grid(group, xx_grid, yy_grid )


            # checks which vales are none
            is_nan = np.isnan(z_grid)

            # Tell the nan mask where values are not nan
            nan_mask[~is_nan] = False

            # --- Handling Different Stacking Methods ---
            if stacking_method == 'average':

                # Add none nan values to count_grid mask
                count_grid[~np.isnan(z_grid)] +=1

                # replaced nan by zero to avoid error
                z_grid[is_nan] = 0

                # stack surface
                stack_mask = stack_mask + z_grid

            elif stacking_method == 'first':
                # Only set values where NaN is present in the mask
                mask = ~is_nan & nan_mask

                stack_mask[mask] = z_grid[mask]

            elif stacking_method == 'last':
                # Always overwrite with the latest values
                stack_mask[~is_nan] = z_grid[~is_nan]

        # --- Finalizing the Output Grid ---
        if stacking_method == 'average':

            # Compute average. Set all 0 values to 1 to avoiding division
            # by zero error
            count_grid[count_grid==0] = 1

            # Devide stack grid by number of datasets per grid cell
            avg_grid = stack_mask / count_grid

            # Uses the nan mask to set values to nan for apha channel
            avg_grid[nan_mask]=np.nan

            return avg_grid
        else:
            return np.full_like(xx_grid, np.nan, dtype=float)

    def _get_tile_grid(self, data, xx_grid, yy_grid):
        """
        Parameters MISSING DESCRIPTION
        ----------
        data : containting x,y,z point cloud data, from MBES footprint.
        xx_grid : x coordinats of the current tile
        yy_grid : y coordinats of the current tile

        -------
        z_grid : computes for a gridded surface and substract the mean
        """
        # get config settings
        coords = np.column_stack((data["x"], data["y"]))

        # Substract reference surface
        z_residual = (data["depth"] - data["ref_depth"]).values

        # Substract mean from point cloud z_residual
        #z_residual = z_residual  - np.nanmean(z_residual)                                                 # Debug add remove mean

        z_grid = griddata(coords, z_residual, (xx_grid, yy_grid),
                          method= self._grid_method, fill_value=np.nan)

        # Substract mean from surface
        if ~np.isnan(z_grid).all():
            z_grid = z_grid - np.nanmean(z_grid)

            # round to 5th comma digit
            z_grid = np.round(z_grid,5)                                           # Debug comment rounding
            z_grid[z_grid==0] = 0.0001  # Ersetze alle 0-Werte
        else:
            z_grid = np.full_like(xx_grid, np.nan, dtype=float)
        
        return z_grid

    def _get_tile_gdf(self,data):
        """Generates a GeoDataFrame of tile polygons from a
        directory of parquet files.
        """

        crs=self._EU_Grid_crs
        local_arg1_ID = '[f701]'

        self._log.add(arg1=local_arg1_ID ,arg2=self.vararg2,
                      arg3=['\t- Add Polygon geometry to dataframe...'])

        def parse_coordinates(expressions):
            """Parses coordinate strings and extracts x, y, and tile size."""
            expressions = np.array(expressions, dtype=str)
            tile_sizes, xcoords, ycoords = [], [], []

            for expression in expressions:
                match = re.match(r'(\d+km|\d+m)E(\d+)N(\d+)', expression)
                #match = re.match(r'^(\d+km|\d+m)E-?\d+N-?\d+$', expression)
                if not match:
                    raise ValueError(f"Invalid format: {expression}")

                tile_str, x_str, y_str = match.groups()
                TileSize = 10000 if 'km' in tile_str else 10  # Determine tile size
                xcoord, ycoord = int(x_str) * TileSize, int(y_str) * TileSize

                tile_sizes.append(TileSize)
                xcoords.append(xcoord)
                ycoords.append(ycoord)

            return np.array(xcoords), np.array(ycoords), np.array(tile_sizes)

        def construct_tile_polygons(tile_ids):
            """Constructs square polygons from tile coordinate expressions."""

            #tile_ids = np.unique(tile_ids).astype(str)
            xcoords, ycoords, tile_sizes = parse_coordinates(tile_ids)

            tiles = [Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])
                     for x, y, s in zip(xcoords, ycoords, tile_sizes)]

            return gpd.GeoSeries(tiles)

        # Step 1: Construct polygons
        tile_polygons = construct_tile_polygons(data['tile_id'])

        #data["tile_polygons"]= tile_polygons

        # Step 2: Create GeoDataFrame
        data = gpd.GeoDataFrame(data, geometry=tile_polygons, crs=crs)
        #data.info(memory_usage="deep")

        return data

    def _save_current_df(self,df,output_fullfile):

        local_ID = '[f702]'

        self._log.add(arg1= local_ID ,arg2=self.vararg2,
                      arg3='\t- Saving results to pkl dataframe...')

        # store geodataframe in input direcoty as save file
        if os.path.isfile(output_fullfile):
            os.remove(output_fullfile)

        df.to_pickle(output_fullfile)

        self._log.add(arg3='--> success')