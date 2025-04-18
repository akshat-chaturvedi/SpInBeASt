import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import os

def list_fits_files_hst(directory):
    """
    Recursively finds all files with a x1d.fits extension in a given directory and its subdirectories.

    Parameters:
        directory (str): The path to the directory to search for x1d.fits files.

    Returns:
        list: A list of file paths to all x1d.fits files found in the directory and its subdirectories.
     """
    fits_files = []

    # Loop through all subdirectories
    for subdir, _, files in os.walk(directory):
        # Loop through files in each subdirectory
        for file in files:
            # Check if the file has a .fits extension
            if file.endswith("x1d.fits"):
                # Append full file path to the list
                fits_files.append(os.path.join(subdir, file))

    return fits_files