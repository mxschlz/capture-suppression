import os
import shutil
from SPACEPRIME import get_data_path

def move_files(source_dir, destination_dir, file_types):
    """
    Moves files of specified types from a source directory to a destination directory.

    Args:
        source_dir: The path to the source directory.
        destination_dir: The path to the destination directory.
        file_types: A list of file extensions (e.g., ['.txt', '.pdf'])
                    or a single file extension string to move.  Case-insensitive.
    """

    try:
        # Ensure destination directory exists
        os.makedirs(destination_dir, exist_ok=True)  # Create if doesn't exist

        if isinstance(file_types, str): # If a single string is passed
            file_types = [file_types] # make it a list

        for filename in os.listdir(source_dir):
            source_path = os.path.join(source_dir, filename)

            if os.path.isfile(source_path):  # Only process files
                for file_type in file_types:
                  if filename.lower().endswith(file_type.lower()): # Case-insensitive check
                      destination_path = os.path.join(destination_dir, filename)
                      shutil.move(source_path, destination_path)  # Use move for efficiency
                      print(f"Moved: {filename} to {destination_dir}")
                      break # Once file type is matched, no need to check other types for the same file
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


subject = input("Enter the subject name (without extension): ")
source_directory = f"{get_data_path()}sourcedata/raw/sub-{subject}/eeg/"  # Replace with your source directory
destination_directory = f"{get_data_path()}sourcedata/raw/sub-{subject}/headgaze" # Replace with your destination directory

# Moving .txt and .pdf files:
file_types_to_move = ['.VideoConfig', '.asf']
move_files(source_directory, destination_directory, file_types_to_move)

print("File moving process complete.")
