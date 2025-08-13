import time
from astropy.visualization import quantity_support
quantity_support()
from specutils.manipulation import FluxConservingResampler
fluxcon = FluxConservingResampler()
import concurrent.futures
from chiron import *
from arces import *
from hst import *
from chara import *
from misc_analysis import *
import logging

# __version__ = '0.1 | 2025/07/21' # First version with number and new name :)
#__version__ = '1.0 | 2025/07/22' # Significant structure updates
# __version__ = '1.1 | 2025/07/23' # Added visual orbit analysis code
# __version__ = '1.2 | 2025/07/25' # Cleaning up file structure, visual changes to plots, added stellar props dictionary
__version__ = '1.3 | 2025/08/13' # Added scripts, TLUSTY model generator, and CCF analysis functionality for HST/STIS spectra

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

spin_beast = f"""
            ############################ This is {RED}Sp{RESET}{GREEN}In{RESET}{YELLOW}Be{RESET}{BLUE}A{RESET}{MAGENTA}St{RESET} #############################
               [{RED}Sp{RESET}]ectroscopic and [{GREEN}In{RESET}]terferometric [{YELLOW}Be{RESET}]-star [{BLUE}A{RESET}]nalysis [{MAGENTA}St{RESET}]ructure
                                     Version: {__version__}                           
                           https://github.com/akshat-chaturvedi/SpInBeASt                
            ############################################################################
"""

def apo_main(file_name):
    logging.basicConfig(
        filename='SpectrumAnalyzer.log',
        encoding='utf-8',
        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=logging.INFO,
        force=True  # IMPORTANT: Overwrites previous configs, needed in subprocesses
    )
    star = ARCESSpectrum(file_name)
    star.spec_plot()
    star.multi_epoch_spec()
    star.radial_velocity_bisector()
    logging.info(f'ARCES Spectrum Analyzed: {star.star_name}, Observation Date: {star.obs_date}')
    print("\a")

def hst_main(file_name):
    logging.basicConfig(
        filename='SpectrumAnalyzer.log',
        encoding='utf-8',
        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=logging.INFO,
        force=True  # IMPORTANT: Overwrites previous configs, needed in subprocesses
    )
    star = HSTSpectrum(file_name)
    star.spec_plot(full_spec=True)
    logging.info(f'HST Spectrum Analyzed: {star.star_name}, Observation Date: {star.obs_date}')
    print("\a")


def chiron_main(file_name):
    logging.basicConfig(
        filename='SpectrumAnalyzer.log',
        encoding='utf-8',
        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=logging.INFO,
        force=True  # IMPORTANT: Overwrites previous configs, needed in subprocesses
    )

    star = CHIRONSpectrum(file_name)
    star.blaze_corrected_plotter(full_spec=True)
    star.multi_epoch_spec()
    # star.radial_velocity()
    star.radial_velocity_bisector()
    logging.info(f'CHIRON Spectrum Analyzed: {star.star_name}, Observation Date: {star.obs_date}')

    # rv_files = glob.glob("CHIRON_Spectra/StarSpectra/RV_Measurements/*_RV.txt", recursive=True)
    # for file in rv_files:
    #     with open(file, "r") as f:
    #         jds = sorted(f.read().splitlines())
    #     with open(file, "w") as f:
    #         f.write("\n".join(jds))
    #     rv_plotter(file)

if __name__ == '__main__':
    # pass
    print(f"{spin_beast}")
    t1 = time.perf_counter()
    # chiron_fits_files = list_fits_files("CHIRON_Spectra/StarSpectra")
    # apo_main()
    # hst_main()
    # hst_fits_files = list_fits_files_hst("HST_Spectra")
    # chiron_main("CHIRON_Spectra/StarSpectra/HR2142_Second.fits")
    # with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "r") as f:
        #Read the names
        # star_names = sorted(f.read().splitlines())
    #
    # # Modify the names based on the condition
    # # for name in star_names:
    # #     # name = name.strip()  # Remove leading/trailing whitespaces
    # #     creation_date = get_file_creation_date(name)
    # #     if creation_date:
    # #         name += f"-->DONE-{creation_date}"
    # #     modified_names.append(name)
    #
    # Write back to the file
    # with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "w") as file:
    #     file.write("\n".join(star_names))

    sky_plot()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(chiron_main, chiron_fits_files)

    # files = glob.glob("CHARA/Prepped/HD191610/*28_Cyg*.oifits")
    # files.append('CHARA/Prepped/HD191610/2025Jul05_HD191610_NFiles01_H_test1split5m_prepped.oifits')
    # star_names = ['HD191610'] * len(files)
    # star_diams = [0.214] * len(files)
    #
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(binary_fit, files, star_names, star_diams)

    # json_files = glob.glob("CHARA/CompanionParams/*HD191610_H.json")
    # json_files.sort()
    # companion_data = []
    # for file in json_files:
    #     companion_data.append(companion_position(file))
    #
    # orbit_plotter(companion_data, "HD191610")


    t2 = time.perf_counter()
    print(f'Finished in \033[94m{round(t2 - t1, 2)}\033[0m second(s)')
    print("\a")

# Finished in 166.4 second(s) - w/o multiproc
# Finished in 42.56 second(s) - w/ multiproc