import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyAstronomy as pyasl
from scipy.interpolate import CubicSpline
from astropy.table import Table
from astropy.io import ascii, fits
from astropy import units as u
from cmcrameri import cm

from astropy.visualization import quantity_support
quantity_support()
from specutils.manipulation import FluxConservingResampler
fluxcon = FluxConservingResampler()
from specutils import Spectrum1D, SpectralRegion

def model_generator(model_spec, model_cont):
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
    resampled_grid1 = pd.read_fwf("../HST_Spectra/HST_STIS_Spectra/HD029441_spectrum_data.txt", header=None, skiprows=1)[0]


    cs_resample = CubicSpline(mod_spec['col1'][ind], norm_spec[ind])
    resampled_spec = cs_resample(resampled_grid1)

    resampled_spec[resampled_grid1 < 1239] = 1
    resampled_spec[(resampled_grid1 > 1289.9) & (resampled_grid1 < 1309.96)] = 1
    resampled_spec[(resampled_grid1 > 1386.9) & (resampled_grid1 < 1409.9129)] = 1
    resampled_spec[(resampled_grid1 > 1544.9108) & (resampled_grid1 < 1554.9147)] = 1
    resampled_spec[(resampled_grid1 > 1649.9363) & (resampled_grid1 < 1708.9618)] = 1

    # resampled_grid = pd.Series(resampled_grid)
    # resampled_spec = pd.Series(resampled_spec)
    # dat = pd.concat([resampled_grid, resampled_spec], axis='columns')
    hdu = fits.PrimaryHDU(data=resampled_spec)
    hdul = fits.HDUList([hdu])
    hdul.writeto(f"../HST_Spectra/Models/Generic_Models/TLUSTY{mod_temp}_rec.fits", overwrite=True)

    return resampled_grid1, resampled_spec, mod_temp

# model_spec = ascii.read("Guvspec/G45000g475v10.uv.7")
# model_cont = ascii.read("Guvspec/G45000g475v10.uv.17")

if __name__ == '__main__':
    list_of_model_specs = sorted(glob.glob("Guvspec/*.uv.7"))
    list_of_model_conts = sorted(glob.glob("Guvspec/*.uv.17"))

    models = []
    for i in range(len(list_of_model_specs)):
        models.append(model_generator(list_of_model_specs[i], list_of_model_conts[i]))

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
    fig.savefig(f"../HST_Spectra/Models/Generic_Models/Models.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    with fits.open("../HST_Spectra/Models/Generic_Models/TLUSTY45000_rec.fits") as hdul:
        dat1 = hdul[0].data

    with fits.open("../HST_Spectra/Models/HD029441_Models/TLUSTY45000_rec.fits") as hdul:
        dat2 = hdul[0].data

    omit_inds = [[0, 3012],
                 [4592, 5207],
                 [7490, 8147],
                 [11802, 12060],
                 [14431, 15834]]

    for ind1 in omit_inds:
        dat2[ind1[0]:ind1[1]] = 1

    x_arr1 = np.linspace(1500, 1708.9618, len(dat1))
    x_arr2 = np.linspace(1500, 1708.9618, len(dat2))

    cmap = cm.managua  # or cm.roma, cm.lajolla, etc.
    N = 2  # Number of colors (e.g., for 10 lines)
    colors = [cmap(i / N) for i in range(N)]
    fig, ax = plt.subplots(figsize=(20, 5))
    offset = np.arange(N)
    ax.plot(x_arr1, dat1, c='k', label="Generic Model")
    ax.plot(x_arr2, dat2, c='r', alpha=0.8, label="HD029441 Model")
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel(r"Wavelength", fontsize=20)
    ax.set_ylabel("Relative Flux", fontsize=20)
    # ax.set_title("Cross Correlation Function", fontsize=24)
    ax.legend(title="Model Temperature", title_fontsize=16, ncols=3, fontsize=12)
    # ax.set_xlim(1578, 1584)
    # ax.set_ylim(0.04, 0.13)
    # fig.savefig(f"../HST_Spectra/Models/Generic_Models/Models.pdf", bbox_inches="tight", dpi=300)
    plt.show()