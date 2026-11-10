import socket
import os

# --- Path Utilities ---
def get_data_path():
    """
    Determines the data path based on the operating system and hostname.
    (This function is from your existing code.)
    """
    if os.name == 'nt':
        data_path = 'G:\\Meine Ablage\\PhD\\data\\SPACECUE_explicit\\'
    elif os.name == 'posix':
        hostname = socket.gethostname()
        if "rechenknecht" in hostname:
            data_path = '/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/SPACECUE_explicit/'
        elif 'MaxPC' in hostname:
            data_path = '/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACECUE_explicit/'
        else:
            raise OSError(f"Unknown Linux machine: {hostname}")
    else:
        raise OSError("Unsupported operating system.")
    return data_path

def get_online_data_path():
    """
    Determines the online data path based on the operating system and hostname.
    """
    if os.name == 'nt':
        data_path = 'G:\\Meine Ablage\\PhD\\data\\SPACECUE_online\\beh\\'
    elif os.name == 'posix':
        hostname = socket.gethostname()
        if "rechenknecht" in hostname:
            data_path = '/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/SPACECUE_online/beh/'
        elif 'MaxPC' in hostname:
            data_path = '/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACECUE_online/beh/'
        else:
            raise OSError(f"Unknown Linux machine: {hostname}")
    else:
        raise OSError("Unsupported operating system.")
    return data_path

def get_online_concatenated_data_path():
    """
    Determines the online concatenated data path based on the operating system and hostname.
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
