import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.time import Time
import glob
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()
from specutils.manipulation import FluxConservingResampler
fluxcon = FluxConservingResampler()
from barycorrpy import get_BC_vel
from cmcrameri import cm
from chironHelperFunctions import *

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
            plt.close()

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
            plt.close()

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
            plt.close()


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
                cmap = cm.roma  # or cm.roma, cm.lajolla, etc.
                N = len(wavs)  # Number of colors (e.g., for 10 lines)
                colors = [cmap(i / N) for i in range(N)]
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
                plt.close()

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
                plt.close()

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
                color="k", fontsize=18, transform=ax.transAxes,
                bbox=dict(
                    facecolor='white',  # Box background color
                    edgecolor='black',  # Box border color
                    boxstyle='square,pad=0.3',  # Rounded box with padding
                    alpha=0.9  # Slight transparency
                )
                )
        # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
        # ax.legend(loc="upper right", fontsize=22)
        fig.savefig(f"APO_Spectra/SpectrumPlots/RV_HAlpha_Bisector/RV_{self.star_name}_{self.obs_date}.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        if not os.path.exists("APO_Spectra/APOInventoryRV_Bisector.txt"):
            with open("APO_Spectra/APOInventoryRV_Bisector.txt", "w") as file:
                file.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{v_bis:.3f}\n")
        else:
            with open("APO_Spectra/APOInventoryRV_Bisector.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.obs_jd) in line for line in jds):
                with open("APO_Spectra/APOInventoryRV_Bisector.txt", "a") as f:
                    f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{v_bis:.3f}\n")