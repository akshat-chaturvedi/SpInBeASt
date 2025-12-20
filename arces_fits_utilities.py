"""
Utilities for handling APO/ARCES FITS files.

Author: Akshat S. Chaturvedi
Created: 2025-12-17
"""

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()


RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'


def time_correcter(file_name):
    """
    Function to correct the header timestamp of APO/ARCES fits files generated between UT2024Aug26 and UT2025Sep17 where
    the echelle timestamps were set LATER than they actually were. Also adds a history card to the header to reflect
    this change, along with a 'TIMECORR' flag which is set to TRUE if the time has been updated.

    :param file_name : The file name

    :return: None
    """
    times_1 = ['2024-08-26T00:00:00', '2024-10-29T23:59:59']
    time_range_1 = Time(times_1, format='isot', scale='utc')

    times_2 = ['2024-10-30T00:00:00', '2025-08-14T23:59:59']
    time_range_2 = Time(times_2, format='isot', scale='utc')

    times_3 = ['2025-08-15T00:00:00', '2025-09-17T23:59:59']
    time_range_3 = Time(times_3, format='isot', scale='utc')

    with fits.open(file_name, mode='update') as hdul:
        hdr = hdul[0].header

        if hdr.get('TIMECORR', False):
            print("Header already corrected — skipping.")
            return

        header_timestamp = Time(hdr["DATE-OBS"], format='isot', scale='utc')
        print(f"{file_name} {BLUE}DATE-OBS = {hdr['DATE-OBS']}{RESET}")

        if (header_timestamp > time_range_1[0]) & (header_timestamp < time_range_1[1]):
            print(f"{YELLOW}DATE-OBS in time range 1!{RESET}")
            hdr['DATE-OBS'] = (header_timestamp - 42.5 * u.minute).isot
            hdr['TIMECORR'] = (True, 'DATE-OBS timestamp corrected')
            hdr.add_history('Timestamp reduced by 42m30s (APO/ARCES clock error)')
            print(f"{GREEN}{hdr['history']}{RESET}")

        elif (header_timestamp > time_range_2[0]) & (header_timestamp < time_range_2[1]):
            print(f"{YELLOW}DATE-OBS in time range 2!{RESET}")
            hdr['DATE-OBS'] = (header_timestamp - 2639 * u.second).isot
            hdr['TIMECORR'] = (True, 'DATE-OBS timestamp corrected')
            hdr.add_history('Timestamp reduced by 43m59s (APO/ARCES clock error)')
            print(f"{GREEN}{hdr['history']}{RESET}")

        elif (header_timestamp > time_range_3[0]) & (header_timestamp < time_range_3[1]):
            print(f"{YELLOW}DATE-OBS in time range 3!{RESET}")
            hdr['DATE-OBS'] = (header_timestamp - 2635 * u.second).isot
            hdr['TIMECORR'] = (True, 'DATE-OBS timestamp corrected')
            hdr.add_history('Timestamp reduced by 43m55s (APO/ARCES clock error)')
            print(f"{GREEN}{hdr['history']}{RESET}")
        else:
            print(f"{GREEN}DATE-OBS outside of erroneous time range, no change needed!{RESET}")

        hdul.flush()

    return


def file_renamer(file_name):
    """
    Function to update default reduction pipeline resultant file names (i.e. tell[star_name].[4-digit-number].ex.ec.fits)
    to tell[star_name].[obs-date].ex.ec.fits to avoid duplicate file names. Reads in file header to obtain obs-date, and
    creates a separate file with the new name.

    :param file_name: The file name
    :return: None
    """
    with fits.open(file_name) as hdul:
        hdr = hdul[0].header
        date_obs = hdr['DATE-OBS'].replace(':', '_').split('.')[0]
        updated_file_name = file_name.replace(file_name.split(".ex.ec.fits")[0].split(".")[1], date_obs)
        hdul.writeto(updated_file_name, overwrite=True)

    return


if __name__ == '__main__':
    time_correcter("Sandbox/tellTOI1287.0015.ex.ec.fits")
