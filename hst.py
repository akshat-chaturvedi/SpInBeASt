import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.time import Time
import sys
import json
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()
from specutils.manipulation import FluxConservingResampler
fluxcon = FluxConservingResampler()
from cmcrameri import cm
from hstHelperFunctions import *

class HSTSpectrum:
    def __init__(self, filename: str):
        """
        Initializes a new HSTSpectrum instance

        Parameters:
            filename (str): the filepath to the HST fits file, in the form "HST_Spectra/FitsFiles/[ROOTNAME]_x1d.fits"
        """
        self.filename = filename

        with fits.open(self.filename) as hdul:
            self.hdr = hdul[0].header
            self.dat = hdul[1].data
            self.star_name = self.hdr["TARGNAME"]
            self.obs_date = self.hdr["TDATEOBS"]
            self.wavs = self.dat["WAVELENGTH"]
            self.flux = self.dat["FLUX"]

        # Getting name of star from FITS file header
        self.star_name = self.hdr["TARGNAME"]

        # Obtaining UTC time of observation
        self.obs_date = self.hdr["TDATEOBS"].split("T")[0]
        self.obs_jd = Time(self.hdr["TDATEOBS"], format='isot', scale='utc').jd

        with open("HST_Spectra/HSTInventory.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(self.obs_jd) in line for line in jds):
            with open("HST_Spectra/HSTInventory.txt", "a") as f:
                f.write(f"{self.star_name},{self.obs_jd},{self.obs_date}\n")

        spec_dict = {}
        for i in tqdm(range(len(self.wavs)), colour='#8e82fe', file=sys.stdout):
            spec_dict[i + 1] = {"Wavelengths": list(np.array(self.wavs[i], dtype=float)),
                                "Fluxes": list(np.array(self.flux[i], dtype=float))}

            with open(f"HST_Spectra/SpectrumData/{self.star_name}_{self.obs_date}.json",
                      "w") as f:
                json.dump(spec_dict, f)

    def spec_plot(self, full_spec=False):
        """
        Plots the HST spectrum between 1710.89 Å and 1729.77 Å

        Parameters:
            full_spec (bool): Default=False, plots the whole spectrum
        Returns:
            None
        """
        if os.path.exists("HST_Spectra/Plots"):
            pass
        else:
            os.mkdir("HST_Spectra/Plots")
            print("-->Plots directory created, plots will be saved here!")

        
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.wavs[9], self.flux[9], color="k")
        ax.set_xlabel("Wavelength [Å]", fontsize=20)
        ax.set_ylabel("Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='major', length=10, width=1)
        ax.yaxis.get_offset_text().set_size(20)
        ax.set_title(f"{self.star_name} {self.obs_date}", fontsize=24)
        fig.savefig(f"HST_Spectra/Plots/{self.star_name}_{self.obs_date}.pdf", bbox_inches="tight", dpi=300)
        plt.close()

        if full_spec:
            if os.path.exists("HST_Spectra/Plots/FullSpec"):
                pass
            else:
                os.mkdir("HST_Spectra/Plots/FullSpec")
                print("-->FullSpec directory created, plots will be saved here!")

            
            fig, ax = plt.subplots(figsize=(20, 10))
            for i in range(len(self.wavs)):
                ax.plot(self.wavs[i], self.flux[i], color="k")
            ax.set_xlabel("Wavelength [Å]", fontsize=20)
            ax.set_ylabel("Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.tick_params(axis='both', which='minor', labelsize=18)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.yaxis.get_offset_text().set_size(20)
            ax.set_ylim(-0.2e-9, 1.5e-9)
            ax.set_title(f"{self.star_name} {self.obs_date}", fontsize=24)
            fig.savefig(f"HST_Spectra/Plots/FullSpec/{self.star_name}_{self.obs_date}.pdf", bbox_inches="tight",
                        dpi=300)
            plt.close()