import glob
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
# __version__ = '1.3 | 2025/08/13' # Added scripts, TLUSTY model generator, and CCF analysis functionality for HST/STIS spectra
__version__ = '1.4 | 2025/12/10' # Updated bisector method and error calculation, added Na doublet RV functionality, added archival spectra analysis functionality

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


def apo_main(file_name: str):
    logging.basicConfig(
        filename='SpectrumAnalyzer.log',
        encoding='utf-8',
        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=logging.INFO,
        force=True  # IMPORTANT: Overwrites previous configs, needed in subprocesses
    )
    star = ARCESSpectrum(file_name)
    star.spec_plot(full_spec=True, na_1_doublet=True)
    # star.multi_epoch_spec()
    star.radial_velocity_bisector()
    star.radial_velocity_doublet()
    logging.info(f'ARCES Spectrum Analyzed: {star.star_name}, Observation Date: {star.obs_date}')
    # print("\a")

def hst_main(file_name: str):
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


def chiron_main(file_name: str):
    logging.basicConfig(
        filename='SpectrumAnalyzer.log',
        encoding='utf-8',
        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=logging.INFO,
        force=True  # IMPORTANT: Overwrites previous configs, needed in subprocesses
    )

    star = CHIRONSpectrum(file_name)
    star.blaze_corrected_plotter(he_1_6678=True)
    star.multi_epoch_spec(avg_h_alpha=True)
    # star.multi_epoch_spec(dynamic_h_alpha=True, p=236.50, t_0=2458672.10, na_1_doublet=False, avg_na_1_doublet=False,
    #                       dynamic_na_doublet=False)
    # star.radial_velocity()
    star.radial_velocity_bisector(print_crossings=False)
    star.radial_velocity_doublet()
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
    # chiron_fits_files = list_fits_files("CHIRON_Spectra/StarSpectra/")
    # chiron_fits_files += list_fits_files("CHIRON_Spectra/Archival/")
    # chiron_fits_files = glob.glob("CHIRON_Spectra/StarSpectra/alfAra*.fits")
    apo_fits_files = list_fits_files("APO_Spectra/FitsFiles/")
    # apo_fits_files = glob.glob("APO_Spectra/Spec_Reductions/Final/UT251208/tellHD041335.0056.ex.ec.fits")
    # chiron_main("CHIRON_Spectra/StarSpectra/alfAra_First.fits")
    # apo_main()
    # hst_main()
    # hst_fits_files = list_fits_files_hst("HST_Spectra")
    # with open("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt", "r") as f:
    #     # Read the names
    #     star_names = sorted(f.read().splitlines())
    #
    # Write back to the file
    # with open("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt", "w") as file:
    #     file.write("\n".join(star_names))

    # sky_plot()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(apo_main, f) for f in apo_fits_files]

        # iterate over futures as they complete
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), colour='#8e82fe', file=sys.stdout):
            pass  # each finished future advances the bar
        # executor.map(apo_main, apo_fits_files)
    # chiron_fits_files = glob.glob("CHIRON_Spectra/StarSpectra/ANCol*.fits")
    # for file in chiron_fits_files:
    #     print(file)
    #     chiron_main(file)
    #     except:
    #         pass
    # for file in apo_fits_files:
    #     print(file)
    #     apo_main(file)

    # files = glob.glob("CHARA/Prepped/HD183537/2025Oct31_HD183537_NFiles01_K_test1split5m_prepped.oifits")
    # # files += glob.glob("CHARA/Prepped/HD191610/*split5m_prepped.oifits")
    # # # files = glob.glob("CHARA/Prepped/HD214168/2025Aug23*_H_*test1split5m_prepped.oifits")
    # # # files.append("CHARA/Prepped/HD214168/MIRCX_L2.2021Sep23.8_Lac_A.MIRCX_IDL.1.SPLIT.oifits")
    # star_names = ['HD183537'] * len(files)
    # star_diams = [0.1288989] * len(files)
    # comp_fluxes = [0.0174] * len(files)
    # bands = ['K'] * len(files)
    # # # chara_file = 'CHARA/Prepped/HD041335/MIRCX_L2.2021Nov17.HR_2142.MIRCX_IDL.1.SPLIT.oifits'
    # # binary_fit(files[0], 'HD191610', 0.2139596, "K")
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(binary_fit, files, star_names, star_diams, bands, comp_fluxes)
    # # # for file in files:
    # # #     binary_fit(file, 'HD191610', 0.2139596, "H")
    # json_files = glob.glob("CHARA/CompanionParams/*HD183537*.json")
    # json_files.sort()
    # companion_data = []
    # for file in json_files:
    #     companion_data.append(companion_position(file))
    #
    # orbit_plotter(companion_data, "HD183537")


    t2 = time.perf_counter()
    print(f'Finished in \033[94m{round(t2 - t1, 2)}\033[0m second(s)')
    print("\a")

# Finished in 166.4 second(s) - w/o multiproc
# Finished in 42.56 second(s) - w/ multiproc