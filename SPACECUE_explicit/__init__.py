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