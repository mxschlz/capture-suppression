import socket
import os


# --- Path Utilities ---
def get_data_path():
    """
    Determines the online behavioral raw-data path based on the operating
    system and hostname (top-level folder with per-block csv files).

    NOTE (Nov 2026): DataPipe now saves the online data directly to
    'G:\\Meine Ablage\\DataPipe\\SpaceCue Online'. This folder contains all
    previous datasets (copy of the old 'PhD\\data\\SPACECUE_online\\beh' folder)
    plus all new incoming datasets (from subject 39 onwards).
    """
    if os.name == 'nt':
        data_path = 'G:\\Meine Ablage\\DataPipe\\SpaceCue Online\\'
        # old location (no longer updated): 'G:\\Meine Ablage\\PhD\\data\\SPACECUE_online\\beh\\'
    elif os.name == 'posix':
        hostname = socket.gethostname()
        if "rechenknecht" in hostname:
            # NOTE: not yet synced with the new DataPipe location (Nov 2026)
            data_path = '/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/SPACECUE_online/beh/'
        elif 'MaxPC' in hostname:
            # Insync mirror of the new DataPipe location on Google Drive
            data_path = '/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/DataPipe/SpaceCue Online/'
        else:
            raise OSError(f"Unknown Linux machine: {hostname}")
    else:
        raise OSError("Unsupported operating system.")
    return data_path


def get_concatenated_data_path():
    """
    Determines the path of the concatenated online data folder.
    """
    if os.name == 'nt':
        data_path = 'G:\\Meine Ablage\\PhD\\data\\SPACECUE_online\\concatenated\\'
    elif os.name == 'posix':
        hostname = socket.gethostname()
        if "rechenknecht" in hostname:
            data_path = '/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/SPACECUE_online/concatenated/'
        elif 'MaxPC' in hostname:
            data_path = '/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACECUE_online/concatenated/'
        else:
            raise OSError(f"Unknown Linux machine: {hostname}")
    else:
        raise OSError("Unsupported operating system.")
    return data_path

