# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:19:21 2024
v 0.9.3, Global evaluation 10
@author: misch
"""
from datetime import datetime as dtime
import os
import pandas as pd

class Actionlog:
    """
    This class represents an outomatic legacy logging system.
    to provide a processing protocol

    It provides functionality to log actions, objects, and feedback.

    Input Arruments:
    .add(arg1= "user txt") to pass current Action
    .add(arg2= "user txt") to pass current Opperation
    .add(arg3= "user txt") to pass current Feedback

    .waitbar()= to display the progress within a waitbar process as '#'
    .get_filename(): Disp the Filename of the current logfile
    .import_saved_log('import_file') Import log of aprevious session 
    .write_to_csv() writes the logfile pandas frame to csv in current location 
    .save_log() Saves the current logfile pandas frame in current location

    .set_flash_log_on() displays output in command window
    .set_flash_log_off() mutes output in command window
    .set_autosave_on() auto saves the pandas frame as soon as a new line starts 
    .set_autosave_off() disable auto save

    """

    def __init__(self):
        self._logfile_name = f"Logfile_{dtime.now().strftime('%Y%m%d-%H%M%S')}"
        self._autosave_log = True
        self._flash_log = True  # toggle
        self._hold = False  # toggle
        self._nr_argin = 3
        self._acitve_Waitbar = False
        self._data = pd.DataFrame({
            'Date': [dtime.now().strftime('%Y%m%d')],
            'Time': [dtime.now().strftime('%H%M%S')],
            'Action': ['[init]'],
            'Opperation': ['Actionlog Function'],
            'Feedback': (', '.join(['Success',
                                    f'active Logfile: {self._logfile_name}']))
        })
        if self._flash_log:
            print('\n')
            print('#---')
            print('Call Class: "actionlog"')
            self.out()
            
        os.makedirs('./temp', exist_ok=True) 
        
    def add(self, arg1=None, arg2=None, arg3=None):
        """
        Function to pass arguments to the handler
        """
        self._nr_argin = sum(1 for arg in [arg1, arg2, arg3] if arg is not None)
        if arg1 is not None and arg2 is not None:
            # For each new input line, values needs to be reset

            if self._hold:
                print('')
                self._hold = False

            # always save last complete line before starting the next request
            # this is to ensure no time delay during waitbar opperations
            if self._autosave_log:
                self.save_log()

            # create a new input row
            new_row = {
                'Date': dtime.now().strftime('%Y%m%d'),
                'Time': dtime.now().strftime('%H%M%S'),
                'Action': str(arg1),
                'Opperation': str(arg2),
                'Feedback': ''
            }

            # Add the next row to the existing data frame
            self._data = pd.concat([self._data,
                              pd.DataFrame([new_row])], ignore_index=True)

            # set toggle to true
            self._hold = True

        # Check if feedback is given or if still pending
        if arg3 is not None:
            last_index = self._data.index[-1]
            existing_feedback = self._data.at[last_index, 'Feedback']
            if existing_feedback:
                # Wenn bereits Feedback vorhanden ist, fügen Sie arg3 mit einem Komma hinzu
                #new_feedback = ', '.join([f"{existing_feedback}", arg3])
                if isinstance(arg3, list):
                    new_feedback_as_str = (', '.join(arg3) if isinstance(arg3, list)
                                                else arg3 if arg3 else '')
                    new_feedback = ', '.join([existing_feedback,new_feedback_as_str])
                else:
                    new_feedback = ', '.join([f"{existing_feedback}", arg3])
            else:
                # Wenn kein Feedback vorhanden ist
                new_feedback = (', '.join(arg3) if isinstance(arg3, list)
                                                else arg3 if arg3 else '')

            # Aktualisieren Sie das Feedback-Feld im DataFrame
            self._data.at[last_index, 'Feedback'] = new_feedback

        # gives feedback
        if self._flash_log:
            self.out()

    def out(self):
        """
        Prints the formatted last row to the command window.
        """
        formatted_row = self.format_lastrow()

        if self._hold:
            print(formatted_row, end='')
        else:
            print(formatted_row)

    def format_lastrow(self):
        """
        function to format the output accordingly
        """
        if self._data.empty:
            return "No data available."

        last_index = self._data.iloc[-1]

        part1 = (f"{last_index['Date']} {last_index['Time']} :"
                 f" {last_index['Action'].strip('')}"
                 f"{'.' * (25 - len(last_index['Opperation']))}"
                 f" {last_index['Opperation']} | ")

        part2 = f"{last_index['Feedback']}"

        if  self._nr_argin==1 and len(part2.split(",")) > 1:
            part2_out = part2.split(",")[-1]
            part2_out = f",{part2_out}"
        else:
            part2_out =part2

        formatted_row = {
            '1': part2_out,
            '2': part1,
            '3': part1 + part2
        }.get(str(self._nr_argin), '')

        return formatted_row

    def import_saved_log(self, import_file):
        """
        Imports a previously saved log file into the legacy system.

        Parameters:
            import_file(str): The file path to the saved log file.

        Returns:
            bool: True if the import was successful, False otherwise.
        """

        if os.path.isfile(import_file):
            # Load the saved log file into a DataFrame
            imported_data = pd.read_pickle(import_file)

            # Check if the imported DataFrame has the correct structure
            expected_columns = ['Date', 'Time', 'Action', 'Opperation', 'Feedback']
            if imported_data.columns.tolist() != expected_columns:
                print("\nError: The structure of the imported log file is not compatible.")

            # Update the current log DataFrame with the imported data
            self._data = imported_data
            print("\nLog file imported successfully.")

        else:
            print(f"\nError: Failed to import log file: {import_file}")

    def write_to_csv(self, fname = [], path = []):
        """
        Writes the log data to a CSV file with specified formatting.

        Returns:
            bool: True if the writing was successful, False otherwise.
        """
           
        if len(fname) == 0:
            file_name = self._logfile_name + '.txt'
        else:
            file_name = fname
            if not file_name.endswith(".txt"):
                file_name= os.path.splitext(file_name)[0] + '.txt'
                
        if len(path) == 0:
            fullfile = file_name
        else:
            fullfile = os.path.join(path, file_name)
            
            
        if len(self._data)>0:
            # Open the CSV file for writing
            with open(fullfile, 'w', encoding='ascii') as file:

                # Write each row of data to the CSV file
                for _, row in self._data.iterrows():
                    # Format the row according to the specified format
                    formatted_row  = (f"{row['Date']} {row['Time']} : "
                             f"{row['Action'].strip()}"
                             f"{'.' * (25 - len(row['Opperation']))} "
                             f"{row['Opperation']} | {row['Feedback']}")

                    # Write the formatted row to the CSV file
                    file.write(formatted_row + "\n")

            print(f"\n|--> Log File has been written to '{fullfile}' "
                  f"successfully.")
            self._hold = False
        else:
            print(f"\nError: Failed to write log data"
                  f" to CSV file: {fullfile}. Log data is empty")

    def pkl2txt(self, import_file, path = []):
        self.import_saved_log(import_file)
        path_out = path
        self.write_to_csv(fname = import_file, path = path_out)
        
    def waitbar(self,run,N):
        """
        Function to print a waitbar in command line
        
        """
        self._nr_argin = 0
        last_index = self._data.index[-1]
        existing_feedback = self._data.at[last_index, 'Feedback']

        if existing_feedback is not None:
            new_feedback = f"{existing_feedback}#"
        else:
            new_feedback = "#"

        self._data.at[last_index, 'Feedback'] = new_feedback

		  # gives feedback
        if self._flash_log:
            print('#', end='')

    def set_autosave_on(self):
        """
        enable/disable logfile autosave.
        """
        self._autosave_log = True
        return "\t- Enable autosave"

    def set_autosave_off(self):
        """
        enable/disable logfile autosave.
        """
        self._autosave_log = False
        return ('\t- Disable autosave, dataframe will '
                    'not be saved as an output file')

    def set_flash_log_on(self):
        """
        enable print of logfile in command line
        """
        self._flash_log = True
        return "\t- Flash logfile in command window enabled"

    def set_flash_log_off(self):
        """
        disable print of logfile in command line
        """
        self._flash_log = False
        return "\t- Flash logfile in command window disabled"

    def get_filename(self):
        """
        Returns the file name.
        """
        return self._logfile_name

    def get_log(self):
        """
        return the formated data stored in the active pandas frame.
        """
        print(f"\n\nReplay: {self.get_filename()}")

        if len(self._data)>0:
            for _, row in self._data.iterrows():
                # Format the row according to the specified format
                formatted_row  = (f"{row['Date']} {row['Time']}: "
                         f"{row['Action'].strip()}"
                         f"{'.' * (25 - len(row['Opperation']))} "
                         f"{row['Opperation']} | {row['Feedback']}")
                print(formatted_row)
        else:
            print("\nError: Failed to read dataframe. Make sure dataframe "
                  "exist and is not empty")

    def get_last(self):
        
        if len(self._data) > 0:
            row = self._data.iloc[-1]  # Get the last row directly
        
            formatted_row = (f"{row['Date']} {row['Time']}: "
                             f"{row['Action'].strip()}"
                             f"{'.' * (25 - len(row['Opperation']))} "
                             f"{row['Opperation']} | {row['Feedback']}")
        else:
            formatted_row = ("No Data")
            
        return formatted_row

    def save_log(self):
        """
        stores dataframe containing processing log to pandas frame format
        """
        
        log_out= f"./temp/{self._logfile_name}.pkl"

        
        self._data.to_pickle(log_out)
