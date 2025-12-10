import os
from shutil import which

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.time import Time
from matplotlib.colors import Normalize
from matplotlib import cm as mpcm
import sys
import glob
import json
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
fluxcon = FluxConservingResampler()
from barycorrpy import get_BC_vel, utc_tdb
from cmcrameri import cm
from chironHelperFunctions import *
import logging

class CHIRONSpectrum:
    """
    A class that represents a pipeline for analyzing CTIO/CHIRON slicer mode spectra to obtain radial velocities. It
    follows the steps laid out by §3.3 of Paredes, L. A., et al. 2021, AJ, 162, 176.

    Attributes:
        filename (str): the filepath to the CHIRON FITS file, in the form "CHIRON_Spectra/StarSpectra/[star name]_
        [observation number].fits"
        dat (numpy.ndarray): a NumPy array containing the spectroscopic data
        hdr (astropy.io.fits.header.Header): the FITS file header for the spectrum, containing important metadata
        star_name (str): the name of the star being observed
        obs_jd (float): the UTC Julian date of the observation
        bc_corr (float): the barycentric correction that needs to be applied to the radial velocities derived from this spectrum
        bc_corr_time (float): the barycentric JDUTC time stamp corrected to BJD_TDB

    Methods:
        blaze_corrected_plotter(h_alpha=True, h_beta=True, full_spec=False): Plots the 'blaze-corrected' H Alpha and
        H Beta orders of the spectrum by default. Also plots the full spectrum if full_spec=True
    """

    def __init__(self, filename: str):
        """
        Initializes a new CHIRONSpectrum instance

        Parameters:
            filename (str): the filepath to the CHIRON fits file, in the form "CHIRON_Spectra/StarSpectra/[star name]_[observation number].fits"
        """
        self.filename = filename
        # self.observation_number = self.filename.split("/")[2].split(".fits")[0].split("_")[1]

        with fits.open(self.filename) as hdul:
            self.dat = hdul[0].data
            self.hdr = hdul[0].header

        # Getting name of star from FITS file header
        self.star_name = clean_star_name(self.hdr["OBJECT"])
        # Obtaining JD UTC time of observation
        self.obs_date = self.hdr["EMMNWOB"].split("T")[0]
        self.obs_jd = Time(self.hdr["EMMNWOB"], format='isot', scale='utc').jd

        # Get BC correction using barycorrpy package in units of m/s
        try:
            self.bc_corr = get_BC_vel(JDUTC=self.obs_jd, starname=self.star_name, obsname="CTIO", ephemeris="de430")[0][0]
            self.bc_corr_time = utc_tdb.JDUTC_to_BJDTDB(JDUTC=self.obs_jd, starname=self.star_name, obsname="CTIO", ephemeris="de430")[0][0]
            # print(f"BC Correction: {self.bc_corr}, Raw Time: {self.obs_jd}, BC Correction Time: {self.bc_corr_time}")
        except:
            self.bc_corr = 0
            self.bc_corr_time = self.obs_jd
            print(f"\033[91mWARNING: BC Not Found for\033[0m \033[93m{self.star_name}!\033[0m")
        # Append to inventory file containing star name and observation JD time
        with open("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(self.bc_corr_time) in line for line in jds):
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt", "a") as f:
                f.write(f"{self.star_name},{self.bc_corr_time},{self.obs_date},{self.bc_corr/1000:.3f}\n")

    def blaze_corrected_plotter(self, h_alpha=True, h_beta=False, he_1_6678=False, na_1_doublet=False, full_spec=False):
        """
        Plots the full blaze-corrected CHIRON spectra as well as just the H Alpha and Beta orders (orders 37 and 7)

        Parameters:
            h_alpha (bool): Default=True, plots the order of the spectrum containing H Alpha 6563 Å (order 37)
            h_beta (bool): Default=False, plots the order of the spectrum containing H Beta 4862 Å (order 7)
            he_1_6678 (bool): Default=False, plots the order of the spectrum containing He I 6678 Å (order 38)
            na_1_doublet (bool): Default=False, plots the order of the spectrum containing the Na I D doublet ~5890 Å (order 26)
            full_spec (bool): Default=False, plots the full spectrum

        Returns:
            None
        """

        if h_alpha:
            if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/HAlpha"):
                pass
            else:
                os.mkdir("CHIRON_Spectra/StarSpectra/Plots/HAlpha")
                print("-->H Alpha directory created, plots will be saved here!")
            wavs = []
            fluxes = []
            for j in range(3200):
                wavs.append(self.dat[37][j][0])
                fluxes.append(self.dat[37][j][1])

            if self.star_name == 'HR 2142' and self.obs_date == '2024-12-13':
                continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=38,
                                                               degree=3, sigma_threshold=3, blaze_plots=True)
            else:
                continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=38,
                                                               degree=5, sigma_threshold=3, blaze_plots=True)
            wavs = np.array(wavs)
            fluxes = np.array(fluxes)

            # Plotting H Alpha order
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(wavs, fluxes / continuum_fit, c='k')
            # ax.spines['bottom'].set_color('white')
            # ax.spines['top'].set_color('white')
            # ax.spines['right'].set_color('white')
            # ax.spines['left'].set_color('white')
            # ax.tick_params(axis='x', colors='white', which='both')
            # ax.tick_params(axis='y', colors='white', which='both')
            # ax.yaxis.label.set_color('white')
            # ax.xaxis.label.set_color('white')
            # ax.title.set_color('white')
            ax.set_title(f'{self.star_name}' + fr' {self.obs_date} H$\alpha$', fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.set_xlim(6550, 6575)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/HAlpha/HAlpha_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            wavs = pd.Series(wavs)
            fluxes = pd.Series(fluxes / continuum_fit)
            df = pd.concat([wavs, fluxes], axis="columns")
            df.columns = ["Wavelength", "Flux"]
            df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/HAlpha/{self.star_name}_{self.obs_date}.csv",
                      index=False)

        if h_beta:
            if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/HBeta"):
                pass
            else:
                os.mkdir("CHIRON_Spectra/StarSpectra/Plots/HBeta")
                print("-->H Beta directory created, plots will be saved here!")
            wavs = []
            fluxes = []

            for j in range(3200):
                wavs.append(self.dat[7][j][0])
                fluxes.append(self.dat[7][j][1])

            continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=8,
                                                           degree=5, sigma_threshold=3, blaze_plots=True)
            wavs = np.array(wavs)
            fluxes = np.array(fluxes)

            # Plotting H Beta order
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(wavs, fluxes / continuum_fit, c='k')
            ax.set_title(f'{self.star_name}' + fr' {self.obs_date} H$\beta$', fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/HBeta/HBeta_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            wavs = pd.Series(wavs)
            fluxes = pd.Series(fluxes / continuum_fit)
            df = pd.concat([wavs, fluxes], axis="columns")
            df.columns = ["Wavelength", "Flux"]
            df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/HBeta/{self.star_name}_{self.obs_date}.csv",
                      index=False)

        if he_1_6678:
            if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/He_I_6678"):
                pass
            else:
                os.mkdir("CHIRON_Spectra/StarSpectra/Plots/He_I_6678")
                print("-->He_I_6678 directory created, plots will be saved here!")
            wavs = []
            fluxes = []
            for j in range(3200):
                wavs.append(self.dat[10][j][0])
                fluxes.append(self.dat[10][j][1])

            if self.star_name == 'HR 2142' and self.obs_date == '2024-12-13':
                continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=39,
                                                               degree=3, sigma_threshold=3, blaze_plots=True)
            else:
                continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=39,
                                                               degree=5, sigma_threshold=3, blaze_plots=True)
            wavs = np.array(wavs)
            fluxes = np.array(fluxes)

            # Plotting H Alpha order
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(wavs, fluxes / continuum_fit, c='k')
            ax.set_title(f'{clean_star_name3(self.star_name)}' + fr' {self.obs_date} He I $\lambda$5016', fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.set_xlim(5000, 5030)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/He_I_6678/He_I_6678_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            wavs = pd.Series(wavs)
            fluxes = pd.Series(fluxes / continuum_fit)
            df = pd.concat([wavs, fluxes], axis="columns")
            df.columns = ["Wavelength", "Flux"]
            df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/He_I_6678/{self.star_name}_{self.obs_date}.csv",
                      index=False)

        if na_1_doublet:
            if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/Na_I_Doublet"):
                pass
            else:
                os.mkdir("CHIRON_Spectra/StarSpectra/Plots/Na_I_Doublet")
                print("-->Na_I_Doublet directory created, plots will be saved here!")
            wavs = []
            fluxes = []
            for j in range(3200):
                wavs.append(self.dat[27][j][0])
                fluxes.append(self.dat[27][j][1])

            if self.star_name == 'HR 2142' and self.obs_date == '2024-12-13':
                continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=7,
                                                               degree=3, sigma_threshold=3, blaze_plots=True)
            else:
                continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=7,
                                                               degree=5, sigma_threshold=3, blaze_plots=True)
            wavs = np.array(wavs)
            fluxes = np.array(fluxes)

            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(wavs, fluxes / continuum_fit, c='k')
            ax.set_title(f'{self.star_name}' + fr' {self.obs_date} Na I Doublet', fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.set_xlim(5886, 5900)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/Na_I_Doublet/Na_I_Doublet_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            wavs = pd.Series(wavs)
            fluxes = pd.Series(fluxes / continuum_fit)
            df = pd.concat([wavs, fluxes], axis="columns")
            df.columns = ["Wavelength", "Flux"]
            df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/Na_I_Doublet/{self.star_name}_{self.obs_date}.csv",
                      index=False)

        if full_spec:
            if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/FullSpec"):
                pass
            else:
                os.mkdir("CHIRON_Spectra/StarSpectra/Plots/FullSpec")
                print("-->FullSpec directory created, plots will be saved here!")
            total_wavs = []
            blaze_fluxes = []
            for i in tqdm(range(59), colour='#8e82fe', file=sys.stdout, desc=f"Blaze Correction for {self.star_name}"):
                wavs = []
                fluxes = []
                for j in range(3200):
                    wavs.append(self.dat[i][j][0])
                    fluxes.append(self.dat[i][j][1])

                if self.star_name == "HR 2142" and self.obs_date == "2024-12-13":
                    continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=f"{i + 1}",
                                                                   degree=3, sigma_threshold=3)
                else:
                    continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, self.star_name, self.obs_date, order=f"{i + 1}",
                                                                   degree=5, sigma_threshold=3)
                total_wavs.append(wavs)
                fluxes = np.array(fluxes)

                blaze_fluxes.append(fluxes / continuum_fit)

            # Saves order-wise wavelengths and fluxes as a dictionary to a json file.
            spec_dict = {}
            for i in range(59):
                spec_dict[i + 1] = {"Wavelengths": list(np.array(total_wavs[i], dtype=float)),
                                    "Fluxes": list(np.array(blaze_fluxes[i], dtype=float))}

            with open(f"CHIRON_Spectra/StarSpectra/SpectraData/FullSpec/{self.star_name}_{self.obs_date}.json",
                      "w") as f:
                json.dump(spec_dict, f)

            total_wavelengths = np.array(total_wavs).flatten()
            blaze_flux = np.array(blaze_fluxes).flatten()

            # Plotting full flattened & normalized spectrum
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(total_wavelengths, blaze_flux, c='k')
            ax.set_title(f'{self.star_name}' + f' {self.obs_date}', fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/FullSpec/fullSpec_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            # wavs = pd.Series(total_wavelengths)
            # fluxes = pd.Series(blaze_flux)
            # df = pd.concat([wavs, fluxes], axis="columns")
            # df.columns = ["Wavelength", "Flux"]
            # df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/FullSpec/{self.star_name}_{self.obs_date}.csv",
            #           index=False)

    def multi_epoch_spec(self, h_alpha=True, h_beta=True, he_I_6678=True, avg_h_alpha=False, dynamic_h_alpha=False,
                         avg_h_beta=False, avg_he_I_6678=False, na_1_doublet=False, avg_na_1_doublet=False, dynamic_na_doublet=False, p=None,
                         t_0=None):
        """
        Plots the multi-epoch H Alpha and Beta orders (orders 37 and 7) for stars with multiple observations

        Parameters:
            h_alpha (bool): Default=True, plots the multi-epoch orders of the spectra containing H Alpha 6563 Å (order 37)
            h_beta (bool): Default=True, plots the multi-epoch orders of the spectra containing H Beta 4862 Å (order 7)
            he_I_6678 (bool): Default=True, plots the multi-epoch orders of the spectra containing He I 6678 Å (order 38)
            avg_h_alpha (bool): Default=False, plots an average multi-epoch spectrum for H Alpha 6563 Å (order 37)
            dynamic_h_alpha (bool): Default=False, plots a dynamic multi-epoch spectrum of H Alpha
            avg_h_beta (bool): Default=False, plots an average multi-epoch spectrum for H Beta 4862 Å (order 7)
            p (float): Default=None, the period for phase determination
            t_0 (float): Default=None, the reference time for phase determination
        Returns:
            None
        """
        if h_alpha:
            csv_files = glob.glob(f"CHIRON_Spectra/StarSpectra/SpectraData/HAlpha/{self.star_name}*BCCorrected.csv")
            if len(csv_files) > 1:
                chiron_inventory = pd.read_csv("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt",
                                              header=None)
                wavs = []
                fluxes = []
                jds = []
                for f in csv_files:
                    ind = np.where((chiron_inventory[2] == f.split("/")[4].split("_")[1].split(".")[0]) &
                                   (chiron_inventory[0] == f.split("/")[4].split("_")[0]))[0]
                    # breakpoint()
                    jds.append(np.array(chiron_inventory[1][ind])[0])
                    dat = pd.read_csv(f)
                    wavs.append(np.array(dat["Wavelength"]))
                    fluxes.append(np.array(dat["Flux"]))

                sorted_inds = np.argsort(jds)
                jds = np.array(jds)[sorted_inds]
                wavs = np.array(wavs)[sorted_inds]
                fluxes = np.array(fluxes)[sorted_inds]

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                # colors = ["k", "r"]
                # Colormap
                cmap = cm.berlin
                norm = Normalize(vmin=(jds-2400000).min(), vmax=(jds-2400000).max())
                sm = mpcm.ScalarMappable(norm=norm, cmap=cmap)

                # Plot stacked spectra
                offset_step = 0.1
                offset = 0

                for i in range(len(wavs)):
                    color = sm.to_rgba((jds-2400000)[i])
                    ax.plot(wavs[i], fluxes[i] + offset, c=color)
                    offset += offset_step

                # Labels / axis
                ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} H$\alpha$', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux + offset", fontsize=22)

                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)

                ax.set_xlim(6550, 6590)

                # Add colorbar
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label("HJD-2400000", fontsize=20)
                cbar.ax.tick_params(labelsize=18)

                # Save
                fig.savefig(
                    f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/HAlpha/ME_HAlpha_{self.star_name}.pdf",
                    bbox_inches="tight", dpi=300
                )
                plt.close()

                if avg_h_alpha:
                    plt.rcParams['font.family'] = 'Geneva'
                    fig, ax = plt.subplots(figsize=(20, 10))

                    ax.plot(wavs[0],np.mean(np.stack(fluxes), axis=0), c="k", linewidth=3, zorder=10, label="Average Spectrum")
                    for i in range(len(wavs)):
                        ax.plot(wavs[i], fluxes[i], color="lightgray", linewidth=1)

                    # Labels / axis
                    ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} H$\alpha$', fontsize=24)
                    ax.set_xlabel("Wavelength [Å]", fontsize=22)
                    ax.set_ylabel("Normalized Flux + offset", fontsize=22)

                    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                    ax.tick_params(axis='y', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.tick_params(axis='both', which='major', length=10, width=1)
                    ax.yaxis.get_offset_text().set_size(20)

                    ax.set_xlim(6550, 6590)

                    fig.savefig(
                        f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/HAlpha/ME_HAlpha_{self.star_name}_avg.pdf",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()


                if dynamic_h_alpha:
                    phases = ((jds - t_0) / p) % 1
                    sort_idx = np.argsort(phases)
                    sorted_fluxes = fluxes[sort_idx]
                    sorted_phases = phases[sort_idx]

                    plt.rcParams['font.family'] = 'Geneva'
                    fig, ax = plt.subplots(figsize=(20, 10))
                    img = ax.imshow(
                        sorted_fluxes,
                        aspect='auto',
                        extent=[wavs.min(), wavs.max(), sorted_phases.min(), sorted_phases.max()],  # flips time axis so earliest is at top
                        cmap=cm.berlin,
                        origin="lower"
                    )

                    # Attach colorbar to image & axis
                    cbar = fig.colorbar(img, ax=ax, pad=0.01)
                    cbar.set_label('Normalized flux', fontsize=18)
                    cbar.ax.tick_params(labelsize=18, length=8, width=1)

                    ax.set_xlabel('Wavelength (Å)', fontsize=22)
                    ax.set_ylabel('Orbital Phase', fontsize=22)
                    ax.set_title(f'Dynamic spectrum of Hα – {self.star_name}', fontsize=24)
                    ax.set_xlim(6550, 6590)
                    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                    ax.tick_params(axis='y', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.tick_params(axis='both', which='major', length=10, width=1)
                    ax.yaxis.get_offset_text().set_size(20)

                    fig.savefig(
                        f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/HAlpha/ME_HAlpha_{self.star_name}_dynamic.pdf",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

        if h_beta:
            csv_files = glob.glob(f"CHIRON_Spectra/StarSpectra/SpectraData/HBeta/{self.star_name}*.csv")
            if len(csv_files) > 1:
                chiron_inventory = pd.read_csv("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt",
                                               header=None)
                wavs = []
                fluxes = []
                jds = []
                for f in csv_files:
                    ind = np.where((chiron_inventory[2] == f.split("/")[4].split("_")[1].split(".")[0]) &
                                   (chiron_inventory[0] == f.split("/")[4].split("_")[0]))[0]
                    # breakpoint()
                    jds.append(np.array(chiron_inventory[1][ind])[0])
                    dat = pd.read_csv(f)
                    wavs.append(np.array(dat["Wavelength"]))
                    fluxes.append(np.array(dat["Flux"]))

                sorted_inds = np.argsort(jds)
                jds = np.array(jds)[sorted_inds]
                wavs = np.array(wavs)[sorted_inds]
                fluxes = np.array(fluxes)[sorted_inds]

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                # colors = ["k", "r"]
                # Colormap
                cmap = cm.berlin
                norm = Normalize(vmin=(jds - 2400000).min(), vmax=(jds - 2400000).max())
                sm = mpcm.ScalarMappable(norm=norm, cmap=cmap)

                # Plot stacked spectra
                offset_step = 0.1
                offset = 0

                for i in range(len(wavs)):
                    color = sm.to_rgba((jds - 2400000)[i])
                    ax.plot(wavs[i], fluxes[i] + offset, c=color)
                    offset += offset_step

                # Labels / axis
                ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} H$\beta$', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux + offset", fontsize=22)

                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)

                # ax.set_xlim(6550, 6590)

                # Add colorbar
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label("HJD-2400000", fontsize=20)
                cbar.ax.tick_params(labelsize=18)

                # Save
                fig.savefig(
                    f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/HBeta/ME_HBeta_{self.star_name}.pdf",
                    bbox_inches="tight", dpi=300
                )
                plt.close()

                if avg_h_beta:
                    plt.rcParams['font.family'] = 'Geneva'
                    fig, ax = plt.subplots(figsize=(20, 10))

                    ax.plot(wavs[0], np.mean(np.stack(fluxes), axis=0), c="k", linewidth=3, zorder=10,
                            label="Average Spectrum")
                    for i in range(len(wavs)):
                        ax.plot(wavs[i], fluxes[i], color="lightgray", linewidth=1)

                    # Labels / axis
                    ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} H$\beta$', fontsize=24)
                    ax.set_xlabel("Wavelength [Å]", fontsize=22)
                    ax.set_ylabel("Normalized Flux + offset", fontsize=22)

                    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                    ax.tick_params(axis='y', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.tick_params(axis='both', which='major', length=10, width=1)
                    ax.yaxis.get_offset_text().set_size(20)

                    # ax.set_xlim(6550, 6590)

                    fig.savefig(
                        f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/HBeta/ME_HBeta_{self.star_name}_avg.pdf",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

        if he_I_6678:
            csv_files = glob.glob(f"CHIRON_Spectra/StarSpectra/SpectraData/he_I_6678/{self.star_name}*.csv")
            if len(csv_files) > 1:
                chiron_inventory = pd.read_csv("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt",
                                              header=None)
                wavs = []
                fluxes = []
                jds = []
                for f in csv_files:
                    ind = np.where((chiron_inventory[2] == f.split("/")[4].split("_")[1].split(".")[0]) &
                                   (chiron_inventory[0] == f.split("/")[4].split("_")[0]))[0]
                    jds.append(np.array(chiron_inventory[1][ind])[0])
                    dat = pd.read_csv(f)
                    wavs.append(np.array(dat["Wavelength"]))
                    fluxes.append(np.array(dat["Flux"]))

                sorted_inds = np.argsort(jds)
                jds = np.array(jds)[sorted_inds]
                wavs = np.array(wavs)[sorted_inds]
                fluxes = np.array(fluxes)[sorted_inds]

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                # colors = ["k", "r"]
                cmap = cm.roma  # or cm.roma, cm.lajolla, etc.
                N = len(wavs)  # Number of colors (e.g., for 10 lines)
                colors = [cmap(i / N) for i in range(N)]
                for i in range(len(wavs)):
                    ax.plot(wavs[i], fluxes[i], c=colors[i], label=f"HJD={jds[i]:.3f}")
                ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} He I $\lambda$6678', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux", fontsize=22)
                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)
                ax.legend(loc="upper right", fontsize=18)
                fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/he_I_6678/ME_he_I_6678_{self.star_name}.pdf",
                            bbox_inches="tight", dpi=300)
                plt.close()

                if avg_he_I_6678:
                    plt.rcParams['font.family'] = 'Geneva'
                    fig, ax = plt.subplots(figsize=(20, 10))

                    ax.plot(wavs[0],np.mean(np.stack(fluxes), axis=0), c="k", linewidth=3, zorder=10, label="Average Spectrum")
                    for i in range(len(wavs)):
                        ax.plot(wavs[i], fluxes[i], color="lightgray", linewidth=1)

                    # Labels / axis
                    ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} He I $\lambda$6678', fontsize=24)
                    ax.set_xlabel("Wavelength [Å]", fontsize=22)
                    ax.set_ylabel("Normalized Flux + offset", fontsize=22)

                    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                    ax.tick_params(axis='y', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.tick_params(axis='both', which='major', length=10, width=1)
                    ax.yaxis.get_offset_text().set_size(20)

                    ax.set_xlim(5000, 5030)

                    fig.savefig(
                        f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/he_I_6678/ME_he_I_6678_{self.star_name}_avg.pdf",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

        if na_1_doublet:
            csv_files = glob.glob(f"CHIRON_Spectra/StarSpectra/SpectraData/Na_I_Doublet/{self.star_name}*.csv")
            if len(csv_files) > 1:
                chiron_inventory = pd.read_csv("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt",
                                              header=None)
                wavs = []
                fluxes = []
                jds = []
                for f in csv_files:
                    ind = np.where((chiron_inventory[2] == f.split("/")[4].split("_")[1].split(".")[0]) &
                                   (chiron_inventory[0] == f.split("/")[4].split("_")[0]))[0]
                    # breakpoint()
                    jds.append(np.array(chiron_inventory[1][ind])[0])
                    dat = pd.read_csv(f)
                    wavs.append(np.array(dat["Wavelength"]))
                    fluxes.append(np.array(dat["Flux"]))

                sorted_inds = np.argsort(jds)
                jds = np.array(jds)[sorted_inds]
                wavs = np.array(wavs)[sorted_inds]
                fluxes = np.array(fluxes)[sorted_inds]

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                # colors = ["k", "r"]
                # Colormap
                cmap = cm.berlin
                norm = Normalize(vmin=(jds-2400000).min(), vmax=(jds-2400000).max())
                sm = mpcm.ScalarMappable(norm=norm, cmap=cmap)

                # Plot stacked spectra
                offset_step = 0.1
                offset = 0

                for i in range(len(wavs)):
                    color = sm.to_rgba((jds-2400000)[i])
                    ax.plot(wavs[i], fluxes[i] + offset, c=color)
                    offset += offset_step

                # Labels / axis
                ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} Na I Doublet', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux + offset", fontsize=22)

                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)

                ax.set_xlim(5886, 5900)

                # Add colorbar
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label("HJD-2400000", fontsize=20)
                cbar.ax.tick_params(labelsize=18)

                # Save
                fig.savefig(
                    f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/Na_I_Doublet/ME_Na_I_Doublet_{self.star_name}.pdf",
                    bbox_inches="tight", dpi=300
                )
                plt.close()

                if avg_na_1_doublet:
                    plt.rcParams['font.family'] = 'Geneva'
                    fig, ax = plt.subplots(figsize=(20, 10))

                    ax.plot(wavs[0],np.mean(np.stack(fluxes), axis=0), c="k", linewidth=3, zorder=10, label="Average Spectrum")
                    for i in range(len(wavs)):
                        ax.plot(wavs[i], fluxes[i], color="lightgray", linewidth=1)

                    # Labels / axis
                    ax.set_title(fr'Multi-epoch {clean_star_name3(self.star_name)} Na I Doublet', fontsize=24)
                    ax.set_xlabel("Wavelength [Å]", fontsize=22)
                    ax.set_ylabel("Normalized Flux + offset", fontsize=22)

                    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                    ax.tick_params(axis='y', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.tick_params(axis='both', which='major', length=10, width=1)
                    ax.yaxis.get_offset_text().set_size(20)

                    ax.set_xlim(5886, 5900)

                    fig.savefig(
                        f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/Na_I_Doublet/ME_Na_I_Doublet_{self.star_name}_avg.pdf",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()


                if dynamic_na_doublet:
                    phases = ((jds - t_0) / p) % 1
                    sort_idx = np.argsort(phases)
                    sorted_fluxes = fluxes[sort_idx]
                    sorted_phases = phases[sort_idx]

                    plt.rcParams['font.family'] = 'Geneva'
                    fig, ax = plt.subplots(figsize=(20, 10))
                    img = ax.imshow(
                        sorted_fluxes,
                        aspect='auto',
                        extent=[wavs.min(), wavs.max(), sorted_phases.min(), sorted_phases.max()],  # flips time axis so earliest is at top
                        cmap=cm.berlin,
                        origin="lower"
                    )

                    # Attach colorbar to image & axis
                    cbar = fig.colorbar(img, ax=ax, pad=0.01)
                    cbar.set_label('Normalized flux', fontsize=18)
                    cbar.ax.tick_params(labelsize=18, length=8, width=1)

                    ax.set_xlabel('Wavelength (Å)', fontsize=22)
                    ax.set_ylabel('Orbital Phase', fontsize=22)
                    ax.set_title(f'Dynamic spectrum of Na I Doublet – {self.star_name}', fontsize=24)
                    ax.set_xlim(5886, 5900)
                    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                    ax.tick_params(axis='y', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.tick_params(axis='both', which='major', length=10, width=1)
                    ax.yaxis.get_offset_text().set_size(20)

                    fig.savefig(
                        f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/Na_I_Doublet/ME_Na_I_Doublet_{self.star_name}_dynamic.pdf",
                        bbox_inches="tight", dpi=300
                    )
                    plt.close()

    def radial_velocity(self, print_rad_vel=False):
        """
        Obtains the radial velocity for a star by fitting a double Gaussian profile to its H Alpha line and finding the
        trough of fit (similar to the bisector method as described in Wang, L. et al. AJ, 2023, 165, 203). It also plots
        this fit onto the spectrum and transforms the wavelength axis to a radial velocity axis. It then applies a
        barycentric correction to the derived radial velocity, and writes it into a datafile.

        Parameters:
            print_rad_vel (bool): Default=True, prints the radial velocity with the barycentric correction applied

        Returns:
            None
        """
        if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/RV_HAlpha"):
            pass
        else:
            os.mkdir("CHIRON_Spectra/StarSpectra/Plots/RV_HAlpha")
            print("-->RV_HAlpha directory created, plots will be saved here!")

        with open(f'CHIRON_Spectra/StarSpectra/SpectraData/FullSpec/{self.star_name}_{self.obs_date}.json', 'r') as file:
            spec = json.load(file)

        h_alpha_order = 38
        wavs = np.array(spec[f"{h_alpha_order}"]["Wavelengths"])  # Read in H Alpha order wavelengths
        fluxes = np.array(spec[f"{h_alpha_order}"]["Fluxes"])  # Read in H Alpha order fluxes

        log_grid = np.arange(np.log10(min(wavs)), np.log10(max(wavs)), 3.33e-6)  # Create log(lambda) grid for resampling
        resample_grid = 10 ** log_grid * u.AA  # Convert log wavelengths to linear wavelengths

        wavs = wavs * u.AA  # Assign units to H Alpha wavelengths
        fluxes = fluxes * u.Unit('erg cm-2 s-1 AA-1')  # Assign units to H Alpha fluxes

        obs_spec = Spectrum1D(spectral_axis=wavs, flux=fluxes)  # Convert spectrum to Spectrum1D object
        obs_spec_resampled = fluxcon(obs_spec, resample_grid) # Resample spectrum to log grid

        ind = np.where((obs_spec_resampled.spectral_axis.value > 6556) &
                        (obs_spec_resampled.spectral_axis.value < 6572))[0]  # Find indices near H Alpha rest wavelength

        init_gauss_params = [6560.3, 1, 1, 6566.1, 1.2, 1]  # Set initial guesses for double gaussian fit
        popt = curve_fit(f=double_gaussian_fit, xdata=obs_spec_resampled.spectral_axis.value[ind],
                                   ydata=obs_spec_resampled.flux.value[ind] - 1, p0=init_gauss_params)[0]

        xArr = np.array(obs_spec_resampled.spectral_axis.value)
        yFit = double_gaussian_fit(obs_spec_resampled.spectral_axis.value, *popt)

        peak1 = popt[0]  # Set first peak to be mean value of first Gaussian
        peak2 = popt[3]  # Set second peak to be mean value of second Gaussian
        if peak1 < peak2:
            pmin = peak1
            pmax = peak2
        else:
            pmin = peak2
            pmax = peak1

        in_fitmin = np.argmin(np.abs(xArr - pmin))
        in_fitmax = np.argmin(np.abs(xArr - pmax))

        wlen_doppler = xArr[in_fitmin + np.argmin(yFit[in_fitmin:in_fitmax])]  # Find trough of double Gaussian fit

        rad_vel = ((wlen_doppler - 6562.8) / 6562.8) * 3e5  # Find radial velocity
        # rad_vel_bc_corrected = rad_vel - self.bc_corr/1000  # Apply barycentric correction

        if print_rad_vel:
            print(f"RV = \033[92m{rad_vel:.3f} km/s\033[0m")

        plt.rcParams['font.family'] = 'Geneva'
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(((obs_spec_resampled.spectral_axis.value - 6562.8) / 6562.8) * 3e5, obs_spec_resampled.flux.value,
                c="k")
        ax.plot(((obs_spec_resampled.spectral_axis.value - 6562.8) / 6562.8) * 3e5, yFit + 1, c="xkcd:periwinkle",
                linewidth=3)
        ax.plot([rad_vel, rad_vel], [-0.3, yFit[in_fitmin + np.argmin(yFit[in_fitmin:in_fitmax])] + 1], c="r")
        ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        ax.set_ylabel("Normalized Flux", fontsize=22)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        plt.tick_params(axis='y', which='major', labelsize=20)
        plt.tick_params(axis='x', which='major', labelsize=20)
        ax.tick_params(axis='both', which='major', length=10, width=1)
        ax.set_ylim(0.65, None)
        ax.set_xlim(-600, 800)
        ax.text(0.8, 0.8, f"{self.star_name}\nHJD {self.obs_jd:.4f}\nRV = {rad_vel:.3f} km/s",
                color="k", fontsize=18, transform=ax.transAxes)
        fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/RV_HAlpha/RV_Fit_{self.star_name}_{self.obs_date}.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        if not os.path.exists("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt"):
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "w") as file:
                file.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{rad_vel:.3f}\n")
        else:
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.obs_jd) in line for line in jds):
                with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "a") as f:
                    f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{rad_vel:.3f}\n")

    def radial_velocity_bisector(self, print_rad_vel=False, print_crossings=False):
        """
        Obtains the radial velocity for a star by cross correlating two oppositely signed Gaussians to the H Alpha profile
        to sample the wings (similar to the bisector method as described in Wang, L. et al. AJ, 2023, 165, 203). It also
        plots this bisector velocity onto the spectrum and transforms the wavelength axis to a radial velocity axis. It
        then applies a barycentric correction to the derived radial velocity, and writes it into a datafile.

        Parameters:
            print_rad_vel (bool): Default=False, prints the radial velocity with the barycentric correction applied
            print_crossings: Flag to check if the function should print out the zero crossings
        Returns:
            None
        """
        if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/RV_HAlpha_Bisector"):
            pass
        else:
            os.mkdir("CHIRON_Spectra/StarSpectra/Plots/RV_HAlpha_Bisector")
            print("-->RV_HAlpha_Bisector directory created, plots will be saved here!")

        with open(f'CHIRON_Spectra/StarSpectra/SpectraData/FullSpec/{self.star_name}_{self.obs_date}.json', 'r') as file:
            spec = json.load(file)

        h_alpha_order = 38
        wavs = np.array(spec[f"{h_alpha_order}"]["Wavelengths"])  # Read in H Alpha order wavelengths
        fluxes = np.array(spec[f"{h_alpha_order}"]["Fluxes"])  # Read in H Alpha order fluxes

        # v_bis, v_grid, ccf = shafter_bisector_velocity(wavs, fluxes, sep=10, sigma=5)
        # cross, threshold = find_crossings(wavs, fluxes - 1)

        v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs, fluxes, print_flag=print_crossings)

        sig_ind = np.where(wavs > 6600)[0]
        sig_cont = np.std(fluxes[sig_ind]-1)

        res = analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                                gaussian_width_kms=gaussian_width,
                                                p=0.25,
                                                Ntop=5,
                                                M_inject=400,
                                                MC_samples=10_000)

        # err_v_bis = (sig_cont * gaussian_width) / (max(np.array(fluxes) - 1) * 0.25 * np.sqrt(-2 * np.log(0.25)))
        err_v_bis = float(res["sigma_v_median"])

        rad_vel_bc_corrected = v_bis + self.bc_corr/1000  # self.bc_corr has a sign, so need to add (otherwise might add when negative)
        if print_rad_vel:
            print(f"Radial Velocity: \033[92m{rad_vel_bc_corrected:.3f} km/s\033[0m")

        # print(f"{self.obs_date} : Barycentric Correction = {self.bc_corr / 1000}")

        dlamb, coeff = wav_corr(np.array(wavs), self.bc_corr, v_bis)

        plot_ind = np.where((((np.array(wavs) - 6562.8) / 6562.8) * 3e5 > -500) &
                            (((np.array(wavs) - 6562.8) / 6562.8) * 3e5 < 500) &
                            ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]

        ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
        # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]

        fig, ax = plt.subplots(2,1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
        plt.subplots_adjust(hspace=0)
        fig.supxlabel("Wavelength [Å]", fontsize=22, y=0.03)
        ax[0].plot(dlamb, np.array(fluxes) - 1,
                color="black", label="CCF")
        # ax[0].plot(v_grid, ker, c="r")
        ax[0].vlines(6562.8 + 6562.8*rad_vel_bc_corrected/3e5, 0.15 * max(fluxes - 1), 0.35 * max(fluxes - 1), color="r", zorder=1, lw=3)
        ax[0].hlines(0.25 * max(fluxes - 1), dlamb[plot_ind[0]], dlamb[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
        ax[0].tick_params(axis='x', labelsize=20)
        ax[0].tick_params(axis='y', labelsize=20)
        ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=False, right=True, length=10,
                       width=1)
        # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        ax[0].set_ylabel("Normalized Flux", fontsize=22)
        ax[0].set_xlim(6562.8-6562.8*500/3e5, 6562.8+6562.8*500/3e5)
        ax[0].text(0.75, 0.8, fr"{clean_star_name3(self.star_name)} H$\alpha$"
                          f"\nBJD {self.bc_corr_time:.4f}\n"
                          fr"RV$_{{\text{{BC}}}}$ = {rad_vel_bc_corrected:.3f}±{err_v_bis:.3f} km/s",
                color="k", fontsize=18, transform=ax[0].transAxes,
                bbox=dict(
                    facecolor='white',  # Box background color
                    edgecolor='black',  # Box border color
                    boxstyle='square,pad=0.3',  # Rounded box with padding
                    alpha=0.9  # Slight transparency
                )
                )
        wav_to_vel, vel_to_wav = make_vel_wav_transforms(6562.8)  # H alpha
        secax_x = ax[0].secondary_xaxis("top", functions=(wav_to_vel, vel_to_wav))
        secax_x.set_xlabel(r"Radial Velocity [km/s]", fontsize=22)
        secax_x.tick_params(labelsize=22, which='both')
        secax_x.tick_params(axis='both', which='both', direction='in', length=10, width=1)

        ax[1].plot((6562.8+6562.8*(v_grid+self.bc_corr/1000)/3e5)[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=3)
        ax[1].set_ylabel("CCF", fontsize=22)
        ax[1].hlines(0, min((6562.8+6562.8*(v_grid+self.bc_corr/1000)/3e5)[ccf_ind]), max((6562.8+6562.8*(v_grid+self.bc_corr/1000)/3e5)[ccf_ind]), color="k", linestyle="--", zorder=0)
        ax[1].set_ylim(-1, 1)
        ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                          width=1)
        # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
        # ax.legend(loc="upper right", fontsize=22)
        # ax.set_ylim(-0.1, max(fluxes - 1)+0.2)
        fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/RV_HAlpha_Bisector/RV_{self.star_name}_{self.obs_date}.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        wavs = pd.Series(dlamb)
        fluxes = pd.Series(fluxes)

        df = pd.concat([wavs, fluxes], axis="columns")
        df.columns = ["Wavelength", "Flux"]
        df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/HAlpha/{self.star_name}_{self.obs_date}_BCCorrected.csv",
                  index=False)

        if not os.path.exists("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt"):
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "w") as file:
                file.write(f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
        else:
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.bc_corr_time) in line for line in jds):
                with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "a") as f:
                    f.write(f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

        if not os.path.exists(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV.txt"):
            with open(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV.txt", "w") as file:
                file.write(f"{self.bc_corr_time},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
        else:
            with open(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.bc_corr_time) in line for line in jds):
                with open(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV.txt", "a") as f:
                    f.write(f"{self.bc_corr_time},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    def radial_velocity_doublet(self, print_rad_vel=False):
        """
                Obtains the radial velocity for a star by cross correlating two oppositely signed Gaussians to the H Alpha profile
                to sample the wings (similar to the bisector method as described in Wang, L. et al. AJ, 2023, 165, 203). It also
                plots this bisector velocity onto the spectrum and transforms the wavelength axis to a radial velocity axis. It
                then applies a barycentric correction to the derived radial velocity, and writes it into a datafile.

                Parameters:
                    print_rad_vel (bool): Default=False, prints the radial velocity with the barycentric correction applied
                Returns:
                    None
                """
        if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/RV_Doublet"):
            pass
        else:
            os.mkdir("CHIRON_Spectra/StarSpectra/Plots/RV_Doublet")
            print("-->RV_Doublet directory created, plots will be saved here!")

        doublet_dat = pd.read_csv(f'CHIRON_Spectra/StarSpectra/SpectraData/Na_I_Doublet/{self.star_name}_{self.obs_date}.csv')

        wavs = np.array(doublet_dat["Wavelength"])  # Read in Na I Doublet wavelengths
        fluxes = np.array(doublet_dat["Flux"])  # Read in Na I Doublet fluxes

        ism_ind1 = np.where((wavs > 5889.95 - 1) & (wavs < 5889.95 + 1))[0]
        ism_ind2 = np.where((wavs > 5895.92 - 1) & (wavs < 5895.92 + 1))[0]

        # inv_flux = -(fluxes-1)
        # peaks, props = find_peaks(inv_flux, height=0.45)

        peaks = [np.argmin(fluxes[ism_ind1]), np.argmin(fluxes[ism_ind2])]

        # breakpoint()

        # if len(peaks) == 0:
        #     # print(self.filename)
        #     pass
        # else:
        #     if len(peaks) > 2:
        #         vel_d1 = (wavs[peaks[0]] - 5889.95) / 5889.95 * 3e5
        #         vel_d2 = (wavs[peaks[2]] - 5895.92 ) / 5895.92  * 3e5
        #     else:
        #         vel_d1 = (wavs[peaks[0]] - 5889.95) / 5889.95 * 3e5
        #         vel_d2 = (wavs[peaks[1]] - 5895.92) / 5895.92 * 3e5

        if len(peaks) == 0:
            # print(self.filename)
            pass
        else:
            if len(peaks) > 2:
                vel_d1 = (wavs[ism_ind1][peaks[0]] - 5889.95) / 5889.95 * 3e5
                vel_d2 = (wavs[ism_ind2][peaks[2]] - 5895.92) / 5895.92 * 3e5
                peak_check = 1
            else:
                vel_d1 = (wavs[ism_ind1][peaks[0]] - 5889.95) / 5889.95 * 3e5
                vel_d2 = (wavs[ism_ind2][peaks[1]] - 5895.92) / 5895.92 * 3e5
                peak_check = 0


            doublet_vel = np.mean([vel_d1, vel_d2])

            # print(f"Doublet Vel: {doublet_vel}")

            err_v_doublet = 0.5  # Arbitratry error amount, need to check this
            rad_vel_doublet_corrected = doublet_vel + (self.bc_corr / 1000)  # self.bc_corr has a sign, so need to add (otherwise might add when negative)
            # breakpoint()
            dlamb, coeff = wav_corr(np.array(wavs), self.bc_corr, doublet_vel)

            if print_rad_vel:
                print(f"Radial Velocity: \033[92m{rad_vel_doublet_corrected:.3f} km/s\033[0m")

            fig, ax = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
            plt.subplots_adjust(wspace=0)
            fig.supylabel("Normalized Flux", fontsize=22, x=0.06)
            ax[0].plot(dlamb, np.array(fluxes) - 1,
                       color="black", label="CCF")
            ax[0].scatter(dlamb[ism_ind1][peaks[0]], (np.array(fluxes) - 1)[ism_ind1][peaks[0]], c="r")
            ax[0].vlines(wavs[ism_ind1[0]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            ax[0].vlines(wavs[ism_ind1[-1]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                              width=1)
            ax[0].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[0].set_ylabel()
            ax[0].set_xlim(5889.95-5889.95*220/3e5, 5889.95+5889.95*220/3e5)
            ax[0].set_ylim(min(np.array(fluxes) - 1) - 0.1, max(np.array(fluxes) - 1) + 0.1)
            ax[0].text(0.55, 0.85, fr"{clean_star_name3(self.star_name)} Na I D$_1$"
                                  f"\nBJD {self.bc_corr_time:.4f}\nRV = {vel_d1 + self.bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[0].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )
            # wav_to_vel, vel_to_wav = make_vel_wav_transforms(5889.95)  # Na D1
            # secax_x = ax[0].secondary_xaxis("top", functions=(wav_to_vel, vel_to_wav))
            # secax_x.set_xlabel(r"Radial Velocity [km/s]", fontsize=22)
            # secax_x.tick_params(labelsize=22, which='both')
            # secax_x.tick_params(axis='both', which='both', direction='in', length=10, width=1)

            ax[1].plot(dlamb, np.array(fluxes) - 1,
                       color="black", label="CCF")
            ax[1].scatter(dlamb[ism_ind2][peaks[1]], (np.array(fluxes) - 1)[ism_ind2][peaks[1]], c="r")
            ax[1].vlines(wavs[ism_ind2[0]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            ax[1].vlines(wavs[ism_ind2[-1]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                              width=1)
            ax[1].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[1].set_ylabel()
            ax[1].set_xlim(5895.92-5895.92*220/3e5, 5895.92+5895.92*220/3e5)
            ax[1].set_ylim(min(np.array(fluxes) - 1) - 0.1, max(np.array(fluxes) - 1) + 0.1)
            ax[1].text(0.55, 0.85, fr"{clean_star_name3(self.star_name)} Na I D$_2$"
                                  f"\nBJD {self.bc_corr_time:.4f}\nRV = {vel_d2 + self.bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[1].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )
            # wav_to_vel, vel_to_wav = make_vel_wav_transforms(5895.92)  # Na D1
            # secax_x = ax[1].secondary_xaxis("top", functions=(wav_to_vel, vel_to_wav))
            # secax_x.set_xlabel(r"Radial Velocity [km/s]", fontsize=22)
            # secax_x.tick_params(labelsize=22, which='both')
            # secax_x.tick_params(axis='both', which='both', direction='in', length=10, width=1)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/RV_Na_I_Doublet/RV_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            if not os.path.exists("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Na_I_Doublet.txt"):
                with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Na_I_Doublet.txt", "w") as file:
                    file.write(
                        f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Na_I_Doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(self.bc_corr_time) in line for line in jds):
                    with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Na_I_Doublet.txt", "a") as f:
                        f.write(
                            f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

            if not os.path.exists(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV_doublet.txt"):
                with open(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV_doublet.txt", "w") as file:
                    file.write(f"{self.bc_corr_time},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV_doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(self.bc_corr_time) in line for line in jds):
                    with open(f"CHIRON_Spectra/StarSpectra/RV_Measurements/{self.star_name}_RV_doublet.txt", "a") as f:
                        f.write(f"{self.bc_corr_time},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

    @staticmethod
    def exp_time(v_mag_array, star_name_array):
        """
        Calculate and plot the Signal-to-Noise Ratio (SNR) as a function of exposure time for different stellar
        magnitudes based on the CHIRON spectrograph's parameters. Derived from http://www.astro.gsu.edu/~thenry/SMARTS/1.5m.chiron.etc.slicer

        This function computes the SNR for a range of exposure times (from 10 to 1000 seconds) for different stellar
        magnitudes, based on an equation that takes into account the flux, readout noise, and efficiency of the system.
        The results are plotted with logarithmic axes for both exposure time and SNR.

        Parameters:
            v_mag_array: a list of the V-band magnitudes of the stars you want to observe
            star_name_array (list): a list of the names of the stars you want to observe

        Returns:
            None. This function plots the results but does not return any values. It shows a plot with exposure time
            (log scale) on the x-axis and SNR (log scale) on the y-axis. The plot includes curves for different stellar
            magnitudes labels for each star.
        """
        ron = 5.5  # electrons
        npix = 9  # pixels across order
        # nbin = 1.0  # binning in dispersion direction
        eff = 0.05  # total efficiency
        f0 = 3.4e5  # photons/s/pixel unbinned, V=0-mag pixel 0.0202 A

        n = 50  # number of points
        time = 10 ** (1.0 + np.arange(n) / (n - 1) * np.log10(120))  # 10s to 1000s

        n_mag = len(v_mag_array)

        snr = np.zeros((n, n_mag))
        flux = f0 * eff * 10 ** (-0.4 * v_mag_array)  # flux, el/s/pixel

        # Calculate SNR
        for i in range(n_mag):
            snr[:, i] = time * flux[i] / np.sqrt(time * flux[i] + npix * ron ** 2)

        # Plotting SNR vs exposure time
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(n_mag):
            ax.loglog(time, snr[:, i], label=f'{star_name_array[i]}', linestyle='-', color="k", linewidth=3)

        plt.rcParams['font.family'] = 'Geneva'
        ax.yaxis.get_offset_text().set_size(20)
        ax.hlines(130, min(time), max(time), color="xkcd:emerald", label="SNR=130", linewidth=3, alpha=0.8)
        ax.hlines(100, min(time), max(time), color="xkcd:goldenrod", label="SNR=100", linewidth=3, alpha=0.8)
        ax.hlines(80, min(time), max(time), color="xkcd:orange red", label="SNR=80", linewidth=3, alpha=0.8)
        ax.set_xlabel('Exposure Time [s]', fontsize=18)
        ax.set_ylabel('SNR', fontsize=18)
        ax.set_ylim(50, 300)
        ax.set_xlim(10, 1000)
        plt.tick_params(axis='y', which='major', labelsize=18)
        plt.tick_params(axis='x', which='major', labelsize=18)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='major', length=10, width=1)
        ax.tick_params(axis='both', which='minor', length=5, width=1)
        ax.grid(True, which="both", ls="--")
        ax.legend(fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
        plt.close()

    def chiron_log(self):
        logging.info(f'CHIRON Spectrum Analyzed: {self.star_name}, Observation Date: {self.obs_date}')