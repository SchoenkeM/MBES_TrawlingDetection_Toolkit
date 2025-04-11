# subroutines.py

import os
from sys import getsizeof
import numpy as np

from datetime import datetime
from time import time, sleep



#---
# Simplified input option for beam values
class BeamNr:
    """Generate a list of beam numbers given a range as a slice object."""
    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else 0
            step = key.step if key.step is not None else 1
            return list(range(start, stop + 1, step))
        else:
            raise TypeError(">> [In Subroutines]: Invalid operation")


#---
def get_varSize(obj):
    """
    Print the size of an object in memory in a readable format (bytes, KB, MB, GB).
    
    Parameters:
        obj: Any Python object.
    """
    size_in_bytes = getsizeof(obj)
    
    def convert_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    print(f"\n\t\tOccupied working memory: {convert_size(size_in_bytes)}")


#---
def tic():
    """
    Record the current time for timing operations.
    
    Returns:
        float: The start time in seconds since the epoch.
    """
    return time()


#---
def toc(stime, bool_print=None):
    """
    Calculate and optionally print the elapsed time since `stime`.
    
    Parameters:
        stime (float): The start time.
        bool_print (bool, optional): If True, print the elapsed time.
    
    Returns:
        str: Formatted elapsed time as 'HH:MM:SS.sss'.
    """
    ctime = time()
    runtime = ctime - stime
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = runtime % 60
    timestr = "{:02d}:{:02d}:{:.3f}".format(hours, minutes, seconds)
    
    if bool_print:
        print("\n\t\tElapsed time is:", timestr)
    
    return timestr

#---
def pause(duration):
    """
    Pause execution for a specified number of seconds.
    
    Parameters:
        duration (float): Time in seconds to pause execution.
    """
    sleep(duration)

#---
def time_stamp():
    """
    Get a formatted timestamp.
    
    Returns:
        str: Current date and time as 'yyyymmdd-HHMMSS'.
    """
    current_datetime = datetime.now()
    return current_datetime.strftime('%Y%m%d-%H%M%S')


#---
def fullfile(fpath, fname):
    """
    Join a file path and file name, and check if the file exists.
    
    Parameters:
        fpath (str): Directory path.
        fname (str): File name.
    
    Returns:
        str: Full file path if valid, otherwise prints an error message.
    """
    fullpath = os.path.join(fpath, fname)
    
    if not os.path.isfile(fullpath):
        print(f"Invalid file directory: {fullpath}")
    else:
        return fullpath


#---
def cart2pol(x, y, z):
    """
    Convert Cartesian coordinates to polar coordinates (theta, rho, z).
    
    Parameters:
        x (float or np.ndarray): X coordinates.
        y (float or np.ndarray): Y coordinates.
        z (float or np.ndarray): Z coordinates.
    
    Returns:
        tuple: (theta, rho, z) in polar coordinates.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return theta, rho, z


#---
def pol2cart(theta, rho, z):
    """
    Convert polar coordinates to Cartesian coordinates (x, y, z).
    
    Parameters:
        theta (float or np.ndarray): Angle in radians.
        rho (float or np.ndarray): Radius.
        z (float or np.ndarray): Z coordinates.
    
    Returns:
        tuple: (x, y, z) in Cartesian coordinates.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y, z

#---

def fieldnames(df):
    """
    Get key names of a dataframe.
    """
    print(list(df.keys()))

#--- 
def is_float(element):
   """ 
   Check if all elements in the first row can be converted to float
   """
   try:
       float(element)
       return True
   except ValueError:
       return False    

#---
# ment for debugging
def plot_PointCloud(data, set_elev= 30, set_azim=15):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    x= data["x"]
    y= data["y"]
    z= data["depth"]
    z_ref = data ["ref_depth"]
    
    z_res = [z - z_ref]
    
    z_res = z_res - np.mean(z_res)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(x, y, z_res, c=z_res, cmap='viridis', s=30)
    plt.colorbar(sc, ax=ax, label='Z value')

    # Set x and y axis limits
    x_min = min(np.floor(x/10)*10)
    y_min = min(np.floor(y/10)*10)
    
    plt.xlim(x_min, x_min +10)  # X-axis range from 0 to 10
    plt.ylim(y_min, y_min +10)  # Y-axis range from 0 to 10

    # Adjust the view
    ax.view_init(elev=set_elev, azim=set_azim)  # Change elevation and azimuth angles
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title('3D Point Cloud')
    plt.show()

    
    
def plot_gridded_tile(xx_grid, yy_grid, z_grid, file_name='active'):
    import matplotlib.pyplot as plt
   
    plt.figure(figsize=(8, 6))
    plt.imshow(z_grid, extent=(xx_grid.min(), xx_grid.max(), yy_grid.min(), yy_grid.max()), 
               origin='lower',
               cmap='viridis',
               aspect='auto',
               )
    plt.colorbar(label='z values (depth - ref_depth)')
    plt.xlabel('X (grid)')
    plt.ylabel('Y (grid)')

    plt.title(f'Interpolated Grid (z_grid): {file_name}')
    plt.show()