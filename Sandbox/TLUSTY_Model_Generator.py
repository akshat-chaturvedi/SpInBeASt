import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from astropy.table import Table
from astropy.io import ascii, fits
from astropy import units as u
from cmcrameri import cm
import time
import os

from astropy.visualization import quantity_support
quantity_support()
from specutils.manipulation import FluxConservingResampler
fluxcon = FluxConservingResampler()
from specutils import Spectrum1D, SpectralRegion

def sdo_model_generator(model_spec, model_cont):
    """
    Generates model spectra for sdOB stars from initial TLUSTY OSTAR2002 model grid. All initial TLUSTY spectra are
    picked to have varying temperatures with fixed log g value of 4.75 dex. Resultant model spectra are on a log
    wavelength grid corresponding to pixel shifts of ~7.5 km/s

    :param model_spec: Model spectrum from TLUSTY OSTAR2002 grid. In the format of an ascii file with the first column
    representing wavelength, and the second representing flux
    :param model_cont: Model continuum from TLUSTY OSTAR2002 grid corresponding to model spectrum. In the same format.

    :returns: The resampled wavelength grid, the resampled fluxes, and the model temperature
    """
    mod_spec = ascii.read(model_spec)
    mod_cont = ascii.read(model_cont)
    mod_temp = model_spec.split("/")[-1].split('g')[0].split('G')[-1]

    cs = CubicSpline(mod_cont['col1'], mod_cont['col2'])
    interp_cont = cs(mod_spec['col1'])
    norm_spec = mod_spec['col2']/interp_cont

    ind = np.where((mod_spec['col1'] > 1150) & (mod_spec['col1'] < 1708.9618))[0]

    log_step = np.log10(1 + 10**(-4.601))

    log_grid = np.arange(np.log10(1150), np.log10(1708.9618)+ log_step/2, log_step)
    # resampled_grid = 10**log_grid
    resampled_grid = pd.read_fwf("../HST_Spectra/HST_STIS_Spectra/wavelength_grid_hst.dat", header=None)[0]


    cs_resample = CubicSpline(mod_spec['col1'][ind], norm_spec[ind])
    resampled_spec = cs_resample(resampled_grid)

    resampled_spec[resampled_grid < 1239] = 1
    resampled_spec[(resampled_grid > 1289.9) & (resampled_grid < 1309.96)] = 1
    resampled_spec[(resampled_grid > 1386.9) & (resampled_grid < 1409.9129)] = 1
    resampled_spec[(resampled_grid > 1544.9108) & (resampled_grid < 1554.9147)] = 1
    resampled_spec[(resampled_grid > 1649.9363) & (resampled_grid < 1708.9618)] = 1

    # resampled_grid = pd.Series(resampled_grid)
    # resampled_spec = pd.Series(resampled_spec)
    # dat = pd.concat([resampled_grid, resampled_spec], axis='columns')
    hdu = fits.PrimaryHDU(data=resampled_spec)
    hdul = fits.HDUList([hdu])
    hdul.writeto(f"../HST_Spectra/Models/Generic_Models/TLUSTY{mod_temp}_rec.fits", overwrite=True)

    return resampled_grid, resampled_spec, mod_temp

def limb_dark_coefficient(log_g, temp):
    """
    Calculate an interpolated linear limb-darkening coefficient using data from Wade, R. A., & Rucinski, S. M. 1985,
    A&AS, 60, 471 for log g ∈ [3.5,4.5] and effective temperature ∈ [10_000,50_000]

    :param log_g: The desired log g value
    :param temp: The desired effective temperature value

    :returns: The interpolated linear limb-darkening coefficient
        """
    temp_grid = [[10_000, 11_000, 12_000, 13_000, 14_000, 15_000, 16_000, 17_000, 18_000, 20_000, 22_500, 25_000, 30_000,
                  35_000],
                 [10_000, 10_500, 11_000, 11_500, 12_000, 12_500, 13_000, 14_000, 15_000, 16_000, 17_000, 18_000, 20_000,
                  22_500, 25_000, 30_000, 35_000, 40_000],
                 [10_000, 11_000, 12_000, 13_000, 14_000, 15_000, 16_000, 17_000, 18_000, 20_000, 22_500, 25_000, 30_000,
                  35_000, 40_000, 45_000, 50_000]]

    log_g_grid = np.array([3.5, 4.0, 4.5])

    coefficient_grid = [[1313, 1180, 1103, 1046, 1000, 959, 923, 889, 857, 799, 732, 668, 545, 491],
                  [1344, 1254, 1193, 1148, 1112, 1081, 1055, 1009, 970, 935, 903, 873, 817, 752, 689, 558, 470, 443],
                  [1383, 1209, 1121, 1062, 1015, 977, 943, 913, 884, 830, 767, 705, 573, 460, 415, 396, 384]]

    common_temp_grid = [10_000, 10_500, 11_000, 11_500, 12_000, 12_500, 13_000, 14_000, 15_000, 16_000, 17_000, 18_000,
                        20_000, 22_500, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000]

    coeff_35_common = CubicSpline(temp_grid[0], coefficient_grid[0])(common_temp_grid)
    coeff_4_common = CubicSpline(temp_grid[1], coefficient_grid[1])(common_temp_grid)
    coeff_45_common = CubicSpline(temp_grid[2], coefficient_grid[2])(common_temp_grid)

    coefficient_grid_common = np.vstack([coeff_35_common, coeff_4_common, coeff_45_common])

    interp = RegularGridInterpolator((log_g_grid, common_temp_grid), coefficient_grid_common)

    return round(interp((log_g, temp))/1000, 4)

def be_model_generator(model_spec, model_cont, vsini, epsilon, star_name):
    """
    Generates model spectra for Be stars from initial TLUSTY BSTAR2006 model grid. All initial TLUSTY spectra are picked
    to have varying temperatures and log g values. Resultant model spectra are on a log wavelength grid corresponding to
    pixel shifts of ~7.5 km/s, and are broadened with a provided Vsini and linear limb-darkening coefficient.

    :param model_spec: Model spectrum from TLUSTY BSTAR2006 grid. In the format of an ascii file with the first column
    representing wavelength, and the second representing flux
    :param model_cont: Model continuum from TLUSTY BSTAR2006 grid corresponding to model spectrum. In the same format.
    :param vsini: Projected rotational velocity (from Simbad)
    :param epsilon: Linear limb-darkening coefficient from Wade, R. A., & Rucinski, S. M. 1985, A&AS, 60, 471
    :param star_name: Name of target star, used for saving model files in directory with star name.

    :returns: The resampled wavelength grid, the resampled fluxes, and the model temperature
    """

    mod_spec = ascii.read(model_spec)
    mod_cont = ascii.read(model_cont)
    mod_temp = model_spec.split("/")[-1].split('g')[0].split('BG')[-1]

    mod_wavs = mod_spec['col1']

    ind = np.where((mod_wavs > 1150) & (mod_wavs < 1708.9618))[0]
    equid_mod_wavs, equid_norm_spec = pyasl.equidistantInterpolation(mod_wavs[ind], mod_spec['col2'][ind], 'mean')
    broadened_flux1 = pyasl.fastRotBroad(equid_mod_wavs, equid_norm_spec, epsilon, vsini)

    cs = CubicSpline(mod_cont['col1'], mod_cont['col2'])
    interp_cont = cs(equid_mod_wavs)
    norm_spec = broadened_flux1 / interp_cont

    resampled_grid = pd.read_fwf("../HST_Spectra/HST_STIS_Spectra/wavelength_grid_hst.dat", header=None)[0]

    cs_resample = CubicSpline(equid_mod_wavs, norm_spec)
    resampled_spec = cs_resample(resampled_grid)

    resampled_spec[resampled_grid < 1239] = 1
    resampled_spec[(resampled_grid > 1289.9) & (resampled_grid < 1309.96)] = 1
    resampled_spec[(resampled_grid > 1386.9) & (resampled_grid < 1409.9129)] = 1
    resampled_spec[(resampled_grid > 1544.9108) & (resampled_grid < 1554.9147)] = 1
    resampled_spec[(resampled_grid > 1649.9363) & (resampled_grid < 1708.9618)] = 1

    hdu = fits.PrimaryHDU(data=resampled_spec)
    hdul = fits.HDUList([hdu])

    if os.path.exists(f"../HST_Spectra/Models/{star_name}_Models_Be"):
        pass
    else:
        os.mkdir(f"../HST_Spectra/Models/{star_name}_Models_Be")
        print(f"-->HST_Spectra/Models/{star_name}_Models_Be directory created, models will be saved here!")

    hdul.writeto(f"../HST_Spectra/Models/{star_name}_Models_Be/TLUSTY{mod_temp}_rec.fits", overwrite=True)

    return resampled_grid, resampled_spec, mod_temp

if __name__ == '__main__':
    # list_of_model_specs = sorted(glob.glob("Guvspec/*.uv.7"))
    # list_of_model_conts = sorted(glob.glob("Guvspec/*.uv.17"))

    # models = []
    # for i in range(len(list_of_model_specs)):
    #     models.append(sdo_model_generator(list_of_model_specs[i], list_of_model_conts[i]))

    # cmap = cm.managua  # or cm.roma, cm.lajolla, etc.
    # N = len(models)  # Number of colors (e.g., for 10 lines)
    # colors = [cmap(i / N) for i in range(N)]
    # fig, ax = plt.subplots(figsize=(20, 10))
    # offset = np.arange(N)
    # for i in range(len(models)):
    #     ax.plot(models[i][0], models[i][1]+offset[i], c=colors[i], label=models[i][2])
    # ax.tick_params(axis='x', labelsize=18)
    # ax.tick_params(axis='y', labelsize=18)
    # ax.set_xlabel(r"Wavelength", fontsize=20)
    # ax.set_ylabel("Relative Flux", fontsize=20)
    # # ax.set_title("Cross Correlation Function", fontsize=24)
    # ax.legend(title="Model Temperature", title_fontsize=16, ncols=3, fontsize=12)
    # fig.savefig(f"../HST_Spectra/Models/Generic_Models/Models.pdf", bbox_inches="tight", dpi=300)
    # plt.close()

    # model_spect = 'BGuvspec/BG22000g325v2.uv.7'
    # model_contt = 'BGuvspec/BG22000g325v2.uv.17'

    list_of_model_specs = sorted(glob.glob("BGuvspec/*g400*.uv.7"))
    list_of_model_conts = sorted(glob.glob("BGuvspec/*g400*.uv.17"))

    stars_name = 'HD214168'

    models = []
    for i in range(len(list_of_model_specs)):
        models.append(be_model_generator(list_of_model_specs[i], list_of_model_conts[i], 300, limb_dark_coefficient( 4.1,27380), stars_name))

    cmap = cm.managua  # or cm.roma, cm.lajolla, etc.
    N = len(models)  # Number of colors (e.g., for 10 lines)
    colors = [cmap(i / N) for i in range(N)]
    fig, ax = plt.subplots(figsize=(20, 10))
    offset = np.arange(N)
    for i in range(len(models)):
        ax.plot(models[i][0], models[i][1]+offset[i], c=colors[i], label=models[i][2])
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel(r"Wavelength", fontsize=20)
    ax.set_ylabel("Relative Flux", fontsize=20)
    # ax.set_title("Cross Correlation Function", fontsize=24)
    ax.legend(title="Model Temperature", title_fontsize=16, ncols=3, fontsize=12)
    fig.savefig(f"../HST_Spectra/Models/{stars_name}_Models_Be/Models.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    # be_model_generator(model_spect, model_contt, 315, 0.732)
