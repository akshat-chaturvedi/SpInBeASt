import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import mplcursors
from tqdm import tqdm
from astropy.time import Time
import sys
import glob
import json
from scipy.optimize import curve_fit
import scipy.signal as sig
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
fluxcon = FluxConservingResampler()
from barycorrpy import get_BC_vel
from cmcrameri import cm
from chironHelperFunctions import *
from hstHelperFunctions import *

#TODO: Add logging capabilities to each of the classes

class CHIRONSpectrum:
    """
    A class that represents a pipeline for analyzing CTIO/CHIRON slicer mode spectra to obtain radial velocities. It
    follows the steps laid out by §3.3 of Paredes, L. A., et al. 2021, AJ, 162, 176.

    Attributes:
        filename (str): the filepath to the CHIRON FITS file, in the form "CHIRON_Spectra/StarSpectra/[star name]_
        [observation number].fits"
        observation_number (str): a number signifying which observation number the spectrum is from (assuming multiple
        observations were taken)
        dat (numpy.ndarray): a NumPy array containing the spectroscopic data
        hdr (astropy.io.fits.header.Header): the FITS file header for the spectrum, containing important metadata
        star_name (str): the name of the star being observed
        obs_jd (float): the UTC Julian date of the observation
        bc_corr (float): the barycentric correction that needs to be applied to the radial velocities derived from this spectrum

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
        self.observation_number = self.filename.split("/")[2].split(".fits")[0].split("_")[1]

        with fits.open(self.filename) as hdul:
            self.dat = hdul[0].data
            self.hdr = hdul[0].header

        # Getting name of star from FITS file header
        self.star_name = self.hdr["OBJECT"]

        # Obtaining JD UTC time of observation
        self.obs_date = self.hdr["EMMNWOB"].split("T")[0]
        self.obs_jd = Time(self.hdr["EMMNWOB"], format='isot', scale='utc').jd

        # Get BC correction using barycorrpy package in units of m/s
        # TODO: This isn't the actual correction, need to understand this better!
        self.bc_corr = get_BC_vel(JDUTC=self.obs_jd, starname=self.star_name, obsname="CTIO", ephemeris="de430")[0][0]

        # Append to inventory file containing star name and observation JD time
        with open("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(self.obs_jd) in line for line in jds):
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventory.txt", "a") as f:
                f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{self.bc_corr/1000:.3f}\n")

    def blaze_corrected_plotter(self, h_alpha=True, h_beta=True, full_spec=False):
        """
        Plots the full blaze-corrected CHIRON spectra as well as just the H Alpha and Beta orders (orders 37 and 7)

        Parameters:
            h_alpha (bool): Default=True, plots the order of the spectrum containing H Alpha 6563 Å (order 37)
            h_beta (bool): Default=True, plots the order of the spectrum containing H Beta 4862 Å (order 7)
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

            continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, degree=5, sigma_threshold=3)
            wavs = np.array(wavs)
            fluxes = np.array(fluxes)

            # Plotting H Alpha order
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(wavs, fluxes / continuum_fit, c='k')
            ax.set_title(f'{self.star_name}' + fr' {self.obs_date} H$\alpha$', fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/HAlpha/HAlpha_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)

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

            continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, degree=5, sigma_threshold=3)
            wavs = np.array(wavs)
            fluxes = np.array(fluxes)

            # Plotting H Beta order
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(wavs, fluxes / continuum_fit, c='k')
            ax.set_title(f'{self.star_name}' + fr' {self.obs_date} H$\beta$', fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/HBeta/HBeta_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)

            wavs = pd.Series(wavs)
            fluxes = pd.Series(fluxes / continuum_fit)
            df = pd.concat([wavs, fluxes], axis="columns")
            df.columns = ["Wavelength", "Flux"]
            df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/HBeta/{self.star_name}_{self.obs_date}.csv",
                      index=False)

        if full_spec:
            if os.path.exists("CHIRON_Spectra/StarSpectra/Plots/FullSpec"):
                pass
            else:
                os.mkdir("CHIRON_Spectra/StarSpectra/Plots/FullSpec")
                print("-->FullSpec directory created, plots will be saved here!")
            total_wavs = []
            blaze_fluxes = []
            for i in tqdm(range(59), colour='#8e82fe', file=sys.stdout):
                wavs = []
                fluxes = []
                for j in range(3200):
                    wavs.append(self.dat[i][j][0])
                    fluxes.append(self.dat[i][j][1])

                continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, degree=5, sigma_threshold=3)
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
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.tick_params(axis='y', which='major', labelsize=20)
            ax.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/FullSpec/fullSpec_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)

            # wavs = pd.Series(total_wavelengths)
            # fluxes = pd.Series(blaze_flux)
            # df = pd.concat([wavs, fluxes], axis="columns")
            # df.columns = ["Wavelength", "Flux"]
            # df.to_csv(f"CHIRON_Spectra/StarSpectra/SpectraData/FullSpec/{self.star_name}_{self.obs_date}.csv",
            #           index=False)

    def multi_epoch_spec(self, h_alpha=True, h_beta=True):
        """
        Plots the multi-epoch H Alpha and Beta orders (orders 37 and 7) for stars with multiple observations

        Parameters:
            h_alpha (bool): Default=True, plots the multi-epoch orders of the spectra containing H Alpha 6563 Å (order 37)
            h_beta (bool): Default=True, plots the multi-epoch orders of the spectra containing H Beta 4862 Å (order 7)

        Returns:
            None
        """
        if h_alpha:
            csv_files = glob.glob(f"CHIRON_Spectra/StarSpectra/SpectraData/HAlpha/{self.star_name}*.csv")
            if len(csv_files) > 1:
                chiron_inventory = pd.read_csv("CHIRON_Spectra/StarSpectra/analysisInventory.txt",
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

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                colors = ["k", "r"]
                for i in range(len(wavs)):
                    ax.plot(wavs[i], fluxes[i], c=colors[i], label=f"HJD={jds[i]:.3f}")
                ax.set_title(fr'Multi-epoch {self.star_name} H$\alpha$', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)
                ax.legend(loc="upper right", fontsize=18)
                fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/HAlpha/ME_HAlpha_{self.star_name}.pdf",
                            bbox_inches="tight", dpi=300)


        if h_beta:
            csv_files = glob.glob(f"CHIRON_Spectra/StarSpectra/SpectraData/HBeta/{self.star_name}*.csv")
            if len(csv_files) > 1:
                chiron_inventory = pd.read_csv("CHIRON_Spectra/StarSpectra/analysisInventory.txt",
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

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                colors = ["k", "r"]
                for i in range(len(wavs)):
                    ax.plot(wavs[i], fluxes[i], c=colors[i], label=f"HJD={jds[i]:.3f}")
                ax.set_title(fr'Multi-epoch {self.star_name} H$\beta$', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)
                ax.legend(loc="upper right", fontsize=18)
                fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/Multi_Epoch/HBeta/ME_HBeta_{self.star_name}.pdf",
                            bbox_inches="tight", dpi=300)

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
        ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
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

        if not os.path.exists("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt"):
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "w") as file:
                file.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{rad_vel:.3f}\n")
        else:
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.obs_jd) in line for line in jds):
                with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "a") as f:
                    f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{rad_vel:.3f}\n")

    def radial_velocity_bisector(self, print_rad_vel=False):
        """
        Obtains the radial velocity for a star by cross correlating two oppositely signed Gaussians to the H Alpha profile
        to sample the wings (similar to the bisector method as described in Wang, L. et al. AJ, 2023, 165, 203). It also
        plots this bisector velocity onto the spectrum and transforms the wavelength axis to a radial velocity axis. It
        then applies a barycentric correction to the derived radial velocity, and writes it into a datafile.

        Parameters:
            print_rad_vel (bool): Default=True, prints the radial velocity with the barycentric correction applied

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

        v_bis, v_grid, ccf = shafter_bisector_velocity(wavs, fluxes, sep=10, sigma=5)

        # rad_vel_bc_corrected = v_bis - self.bc_corr/1000
        if print_rad_vel:
            print(f"Radial Velocity: \033[92m{v_bis:.3f} km/s\033[0m")

        plot_ind = np.where((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1))[0]

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(((np.array(wavs) - 6562.8) / 6562.8) * 3e5, np.array(fluxes) - 1,
                color="black", label="CCF")
        ax.vlines(v_bis, 0.23 * max(fluxes - 1), 0.27 * max(fluxes - 1), color="r", zorder=1)
        ax.hlines(0.25 * max(fluxes - 1), (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[0]],
                  (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
        ax.set_xlim(-500, 500)
        ax.text(0.8, 0.8, fr"{self.star_name} H$\alpha$"
                          f"\nHJD {self.obs_jd:.4f}\nRV = {v_bis:.3f} km/s",
                color="k", fontsize=18, transform=ax.transAxes)
        # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
        # ax.legend(loc="upper right", fontsize=22)
        fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/RV_HAlpha_Bisector/RV_{self.star_name}_{self.obs_date}.pdf",
                    bbox_inches="tight", dpi=300)

        if not os.path.exists("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt"):
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "w") as file:
                file.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{v_bis:.3f}\n")
        else:
            with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.obs_jd) in line for line in jds):
                with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt", "a") as f:
                    f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{v_bis:.3f}\n")

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
            ax.loglog(time, snr[:, i], label=f'{star_name_array[i]}', linestyle='-')

        plt.rcParams['font.family'] = 'Geneva'
        ax.set_xlabel('Exposure Time [s]', fontsize=18)
        ax.set_ylabel('SNR', fontsize=18)
        ax.set_ylim(None, 300)
        plt.tick_params(axis='y', which='major', labelsize=18)
        plt.tick_params(axis='x', which='major', labelsize=18)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='major', length=10, width=1)
        ax.grid(True, which="both", ls="--")
        ax.legend(fontsize=15, ncols=3)
        plt.show()


class ARCESSpectrum:
    """
    A class that represents a pipeline for analyzing APO ARCES spectra to obtain radial velocities.

    Attributes:
        filename (str): the filepath to the APO FITS file, in the form "APO_Spectra/FitsFiles/tell[star_name].[4-digit-number].ex.ec.fits"
        dat (numpy.ndarray): a NumPy array containing the spectroscopic data
        hdr (astropy.io.fits.header.Header): the FITS file header for the spectrum, containing important metadata
        star_name (str): the name of the star being observed
        obs_date (str): the human-readable date of the observation
        obs_jd (float): the UTC Julian date of the observation
        bc_corr (float): the barycentric correction that needs to be applied to the radial velocities derived from this spectrum

    Methods:
        spec_plot(): Plots the H Alpha-centered and APO spectrum
    """
    def __init__(self, filename: str):
        """
        Initializes a new APOSpectrum instance

        Parameters:
            filename (str): the filepath to the APO fits file, in the form "APO_Spectra/FitsFiles/tell[star_name].[4-digit-number].ex.ec.fits"
        """
        self.filename = filename

        with fits.open(self.filename) as hdul:
            self.dat = hdul[0].data
            self.hdr = hdul[0].header

        # Getting name of star from FITS file header
        self.star_name = self.hdr["OBJNAME"]

        # Handling naming edge cases (KOI names need a dash in between the KOI and the number, BD names need a + sign
        # and space after initial coordinate)
        if self.star_name == "BD_14_3887":
            self.star_name = "BD +14 3887"

        if "KOI" in self.star_name:
            self.star_name = 'KOI' + '-' + self.star_name.split("KOI")[1]

        # Obtaining JD UTC time of observation
        self.obs_date = self.hdr["DATE-OBS"].split("T")[0]
        self.obs_jd = Time(self.hdr["DATE-OBS"], format='isot', scale='utc').jd

        # Get BC correction using barycorrpy package in units of m/s
        try:
            self.bc_corr = get_BC_vel(JDUTC=self.obs_jd, starname=self.star_name, obsname="Apache Point Observatory",
                                      ephemeris="de430")[0][0]
        except Exception as e:
            print(f"{e}")
            pass

        with open("APO_Spectra/APOInventory.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(self.obs_jd) in line for line in jds):
            with open("APO_Spectra/APOInventory.txt", "a") as f:
                f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{self.bc_corr/1000:.3f}\n")

    def spec_plot(self, h_alpha=True, h_beta=True, full_spec=False):
        """
        Plots the H Alpha- and H Beta-centered APO spectra

        Parameters:
            h_alpha (bool): Default=True, plots the spectrum centered on H Alpha 6563 Å
            h_beta (bool): Default=True, plots the spectrum centered on H Beta 4862 Å
            full_spec (bool): Default=False, plots the whole spectrum

        Returns:
            None
        """
        if os.path.exists("APO_Spectra/SpectrumPlots"):
            pass
        else:
            os.mkdir("APO_Spectra/SpectrumPlots")
            print("-->SpectrumPlots directory created, plots will be saved here!")

        # Read in FITS file containing calibrated wavelengths
        with fits.open("APO_Spectra/woned.fits") as wavelengths:
            w = wavelengths[0].data

        if h_alpha:
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(w, self.dat, c="k")
            ax.set_title(fr"{self.star_name} {self.obs_date} H$\alpha$", fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            plt.tick_params(axis='y', which='major', labelsize=20)
            plt.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.set_xlim(6550, 6620)
            mask = (w >= 6550) & (w <= 6620)
            ax.set_ylim(0.5, np.max(self.dat[mask])+0.25)
            # ax.vlines(6562.8, -0.05, self.dat[np.argmin(abs(w - 6562.8))])
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"APO_Spectra/SpectrumPlots/HAlpha/{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)

            wavs = pd.Series(w)
            fluxes = pd.Series(self.dat)
            df = pd.concat([wavs, fluxes], axis="columns")
            df.columns = ["Wavelength", "Flux"]
            df.to_csv(f"APO_Spectra/SpectraData/{self.star_name}_{self.obs_date}.csv",
                      index=False)

        if h_beta:
            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(w, self.dat, c="k")
            ax.set_title(fr"{self.star_name} {self.obs_date} H$\beta$", fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            plt.tick_params(axis='y', which='major', labelsize=20)
            plt.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.set_xlim(4800, 5000)
            mask = (w >= 4800) & (w <= 5000)
            ax.set_ylim(0.5, np.max(self.dat[mask]) + 0.25)
            # ax.vlines(6562.8, -0.05, self.dat[np.argmin(abs(w - 6562.8))])
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"APO_Spectra/SpectrumPlots/HBeta/{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)

            if not h_alpha:
                wavs = pd.Series(w)
                fluxes = pd.Series(self.dat)
                df = pd.concat([wavs, fluxes], axis="columns")
                df.columns = ["Wavelength", "Flux"]
                df.to_csv(f"APO_Spectra/SpectraData/{self.star_name}_{self.obs_date}.csv",
                          index=False)

        if full_spec:
            if os.path.exists("APO_Spectra/SpectrumPlots/FullSpec"):
                pass
            else:
                os.mkdir("APO_Spectra/SpectrumPlots/FullSpec")
                print("-->FullSpec directory created, plots will be saved here!")

            plt.rcParams['font.family'] = 'Geneva'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(w, self.dat, c="k")
            ax.set_title(fr"{self.star_name} {self.obs_date}", fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            plt.tick_params(axis='y', which='major', labelsize=20)
            plt.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            # ax.set_xlim(4800, 5000)
            # mask = (w >= 4800) & (w <= 5000)
            # ax.set_ylim(0.5, np.max(self.dat[mask]) + 0.25)
            # ax.vlines(6562.8, -0.05, self.dat[np.argmin(abs(w - 6562.8))])
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"APO_Spectra/SpectrumPlots/FullSpec/{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)


    def multi_epoch_spec(self, h_alpha=True, h_beta=True):
        """
        Plots the multi-epoch H Alpha and Beta orders for stars with multiple observations

        Parameters:
            h_alpha (bool): Default=True, plots the multi-epoch orders of the spectra containing H Alpha 6563 Å
            h_beta (bool): Default=True, plots the multi-epoch orders of the spectra containing H Beta 4862 Å

        Returns:
            None
        """
        if h_alpha:
            csv_files = glob.glob(f"APO_Spectra/SpectraData/{self.star_name}*.csv")
            #APO_Spectra/SpectraData/7 Vul_2024-11-11.csv
            if len(csv_files) > 1:
                apo_inventory = pd.read_csv("APO_Spectra/APOInventory.txt", header=None)
                wavs = []
                fluxes = []
                jds = []
                for f in csv_files:
                    ind = np.where((apo_inventory[2] == f.split("/")[2].split("_")[1].split(".")[0]) &
                                   (apo_inventory[0] == f.split("/")[2].split("_")[0]))[0]
                    jds.append(np.array(apo_inventory[1][ind])[0])
                    dat = pd.read_csv(f)
                    wavs.append(np.array(dat["Wavelength"]))
                    fluxes.append(np.array(dat["Flux"]))

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                colors = ["k", "r"]
                for i in range(len(wavs)):
                    ax.plot(wavs[i], fluxes[i], c=colors[i], label=f"HJD={jds[i]:.3f}")
                ax.set_title(fr'Multi-epoch {self.star_name} H$\alpha$', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)
                ax.legend(loc="upper right", fontsize=18)
                ax.set_xlim(6500, 6700)
                mask = (wavs[1] >= 6500) & (wavs[1] <= 6700)
                ax.set_ylim(0.5, np.max(fluxes[1][mask]) + 0.25)
                fig.savefig(f"APO_Spectra/SpectrumPlots/Multi_Epoch/HAlpha/ME_HAlpha_{self.star_name}.pdf",
                            bbox_inches="tight", dpi=300)

        if h_beta:
            csv_files = glob.glob(f"APO_Spectra/SpectraData/{self.star_name}*.csv")
            if len(csv_files) > 1:
                apo_inventory = pd.read_csv("APO_Spectra/APOInventory.txt", header=None)
                wavs = []
                fluxes = []
                jds = []
                for f in csv_files:
                    ind = np.where((apo_inventory[2] == f.split("/")[2].split("_")[1].split(".")[0]) &
                                   (apo_inventory[0] == f.split("/")[2].split("_")[0]))[0]
                    jds.append(np.array(apo_inventory[1][ind])[0])
                    dat = pd.read_csv(f)
                    wavs.append(np.array(dat["Wavelength"]))
                    fluxes.append(np.array(dat["Flux"]))

                plt.rcParams['font.family'] = 'Geneva'
                fig, ax = plt.subplots(figsize=(20, 10))
                colors = ["k", "r"]
                for i in range(len(wavs)):
                    ax.plot(wavs[i], fluxes[i], c=colors[i], label=f"HJD={jds[i]:.3f}")
                ax.set_title(fr'Multi-epoch {self.star_name} H$\beta$', fontsize=24)
                ax.set_xlabel("Wavelength [Å]", fontsize=22)
                ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                ax.tick_params(axis='y', which='major', labelsize=20)
                ax.tick_params(axis='x', which='major', labelsize=20)
                ax.tick_params(axis='both', which='major', length=10, width=1)
                ax.yaxis.get_offset_text().set_size(20)
                ax.legend(loc="upper right", fontsize=18)
                ax.set_xlim(4800, 5000)
                mask = (wavs[1] >= 4800) & (wavs[1] <= 5000)
                ax.set_ylim(0.5, np.max(fluxes[1][mask]) + 0.25)
                fig.savefig(f"APO_Spectra/SpectrumPlots/Multi_Epoch/HBeta/ME_HBeta_{self.star_name}.pdf",
                            bbox_inches="tight", dpi=300)

    def radial_velocity_bisector(self, print_rad_vel=False):
        """
        Obtains the radial velocity for a star by cross correlating two oppositely signed Gaussians to the H Alpha profile
        to sample the wings (similar to the bisector method as described in Wang, L. et al. AJ, 2023, 165, 203). It also
        plots this bisector velocity onto the spectrum and transforms the wavelength axis to a radial velocity axis. It
        then applies a barycentric correction to the derived radial velocity, and writes it into a datafile.

        Parameters:
            print_rad_vel (bool): Default=True, prints the radial velocity with the barycentric correction applied

        Returns:
            None
        """
        if os.path.exists("APO_Spectra/SpectrumPlots/RV_HAlpha_Bisector"):
            pass
        else:
            os.mkdir("APO_Spectra/SpectrumPlots/RV_HAlpha_Bisector")
            print("-->RV_HAlpha_Bisector directory created, plots will be saved here!")

        with fits.open("APO_Spectra/woned.fits") as wavelengths:
            wavs = wavelengths[0].data

        mask = (wavs >= 6550) & (wavs <= 6620)
        wavs = wavs[mask]
        fluxes = self.dat[mask]

        v_bis, v_grid, ccf = shafter_bisector_velocity(wavs, fluxes, sep=10, sigma=5)

        # rad_vel_bc_corrected = v_bis - self.bc_corr/1000
        if print_rad_vel:
            print(f"Radial Velocity: \033[92m{v_bis:.3f} km/s\033[0m")

        plot_ind = np.where((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1))[0]

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(((np.array(wavs) - 6562.8) / 6562.8) * 3e5, np.array(fluxes) - 1,
                color="black", label="CCF")
        ax.vlines(v_bis, 0.23 * max(fluxes - 1), 0.27 * max(fluxes - 1), color="r", zorder=1)
        ax.hlines(0.25 * max(fluxes - 1), (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[0]],
                  (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
        ax.set_xlim(-500, 500)
        ax.text(0.8, 0.8, fr"{self.star_name} H$\alpha$"
                          f"\nHJD {self.obs_jd:.4f}\nRV = {v_bis:.3f} km/s",
                color="k", fontsize=18, transform=ax.transAxes)
        # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
        # ax.legend(loc="upper right", fontsize=22)
        fig.savefig(f"APO_Spectra/SpectrumPlots/RV_HAlpha_Bisector/RV_{self.star_name}_{self.obs_date}.pdf",
                    bbox_inches="tight", dpi=300)

        if not os.path.exists("APO_Spectra/APOInventoryRV_Bisector.txt"):
            with open("APO_Spectra/APOInventoryRV_Bisector.txt", "w") as file:
                file.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{v_bis:.3f}\n")
        else:
            with open("APO_Spectra/APOInventoryRV_Bisector.txt", "r") as f:
                jds = f.read().splitlines()
                print(jds)

            if not any(str(self.obs_jd) in line for line in jds):
                with open("APO_Spectra/APOInventoryRV_Bisector.txt", "a") as f:
                    f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{v_bis:.3f}\n")

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

        plt.rcParams['font.family'] = 'Geneva'
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

        if full_spec:
            if os.path.exists("HST_Spectra/Plots/FullSpec"):
                pass
            else:
                os.mkdir("HST_Spectra/Plots/FullSpec")
                print("-->FullSpec directory created, plots will be saved here!")

            plt.rcParams['font.family'] = 'Geneva'
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


def sky_plot(interactive=False):
    dat = pd.read_fwf("sim-id", header=None)

    ra = np.array(dat[0])  # Right Ascension in degrees
    dec = np.array(dat[1])  # Declination in degrees
    star_names = np.array(dat[2], dtype=str)  # Assuming the 3rd column contains star names

    chiron_inventory = pd.read_csv("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV_Bisector.txt",
                                   header=None)
    apo_inventory = pd.read_csv("APO_Spectra/APOInventoryRV_Bisector.txt", header=None)

    hst_inventory = pd.read_csv("HST_Spectra/HSTInventory.txt", header=None)

    overall_obs_count = []
    for name in star_names:
        individual_obs_count = 0
        # breakpoint()
        if name in np.array(chiron_inventory[0]):
            individual_obs_count += np.count_nonzero(np.array(chiron_inventory[0]) == name)
        if name in np.array(apo_inventory[0]):
            individual_obs_count += np.count_nonzero(np.array(apo_inventory[0]) == name)
        overall_obs_count.append(individual_obs_count)

    overall_target_inventory = pd.concat([dat[2], dat[0], dat[1], pd.Series(overall_obs_count)], axis="columns")
    overall_target_inventory.columns = ["Name", "RA", "Dec", "Number of Observations"]
    overall_target_inventory.to_csv("Be_sdO_Target_Inventory.txt", sep="\t", index=False)

    eq = SkyCoord(ra, dec, unit=u.deg)

    eq_ra, eq_dec = -eq.ra.wrap_at('180d').radian, eq.dec.radian

    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "aitoff"})
    ax.set_xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]),
               labels=['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°'])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    scatter = ax.scatter(eq_ra, eq_dec, s=35, zorder=2, c=overall_obs_count, cmap=cm.roma)
    ax.tick_params(axis='both', which='major', labelsize=18)
    cbar = fig.colorbar(scatter, orientation='vertical', ticks=[0, 1, 2])
    cbar.set_label('Number of Observations', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    # Add interactivity with mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(star_names[sel.index]))
    ax.set_title("Be+sdO Targets", fontsize=20)

    if interactive:
        plt.show()

    else:
        fig.savefig("skyplot.pdf", bbox_inches="tight", dpi=300)


def apo_main():
    apo_fits_files = list_fits_files("APO_Spectra/FitsFiles")
    for file in apo_fits_files:
        star = ARCESSpectrum(file)
        star.spec_plot()
        star.multi_epoch_spec()
        star.radial_velocity_bisector()

def hst_main():
    hst_fits_files = list_fits_files_hst("HST_Spectra")
    for file in hst_fits_files:
        star = HSTSpectrum(file)
        star.spec_plot(full_spec=True)


def chiron_main():
    chiron_fits_files = list_fits_files("CHIRON_Spectra/StarSpectra")
    # chiron_fits_files = ["CHIRON_Spectra/StarSpectra/ANCol_First.fits"]
    for file in chiron_fits_files:
        star = CHIRONSpectrum(file)
        star.blaze_corrected_plotter(full_spec=True)
        star.multi_epoch_spec()
        star.radial_velocity()
        star.radial_velocity_bisector()


if __name__ == '__main__':
    pass
    # apo_main()
    # hst_main()
    # chiron_main()
    # with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "r") as f:
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
    # with open("CHIRON_Spectra/StarSpectra/CHIRONInventoryRV.txt", "w") as file:
    #     file.write("\n".join(star_names))

    # sky_plot()
