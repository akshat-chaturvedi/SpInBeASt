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
from barycorrpy import get_BC_vel, JDUTC_to_BJDTDB
from cmcrameri import cm
from chironHelperFunctions import *
from arces_time_correcter import time_correcter

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

        time_correcter(filename)

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
            self.bc_corr = 0
            # self.bc_corr = get_BC_vel(JDUTC=self.obs_jd, starname=self.star_name, obsname="Apache Point Observatory",
            #                           ephemeris="de430")[0][0]
            self.bc_corr_time = JDUTC_to_BJDTDB(JDUTC=self.obs_jd, starname=self.star_name, obsname="Apache Point Observatory",
                                      ephemeris="de430")[0][0]
        except:
            self.bc_corr = 0
            self.bc_corr_time = self.obs_jd
            print(f"\033[91mWARNING: BC Not Found for\033[0m \033[93m{self.star_name}!\033[0m")

        with open("APO_Spectra/APOInventory.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(self.obs_jd) in line for line in jds):
            with open("APO_Spectra/APOInventory.txt", "a") as f:
                f.write(f"{self.star_name},{self.obs_jd},{self.obs_date},{self.bc_corr/1000:.3f}\n")

    def spec_plot(self, h_alpha=True, h_beta=False, na_1_doublet=False, full_spec=False):
        """
        Plots the H Alpha- and H Beta-centered APO spectra

        Parameters:
            h_alpha (bool): Default=True, plots the spectrum centered on H Alpha 6563 Å
            h_beta (bool): Default=False, plots the spectrum centered on H Beta 4862 Å
            na_1_doublet (bool): Default=False, plots the spectrum centered on the Na I D doublet ~5890 Å
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
            # plt.rcParams['font.family'] = 'Trebuchet MS'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(w, self.dat, c="k")
            ax.set_title(fr"{self.star_name} {self.obs_date} H$\alpha$", fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            plt.tick_params(axis='y', which='major', labelsize=20)
            plt.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.set_xlim(6550, 6575)
            mask = (w >= 6550) & (w <= 6575)
            ax.set_ylim(np.min(self.dat[mask]-0.25), np.max(self.dat[mask])+0.25)
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
            # plt.rcParams['font.family'] = 'Trebuchet MS'
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

        if na_1_doublet:
            # plt.rcParams['font.family'] = 'Trebuchet MS'
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(w, self.dat, c="k")
            ax.set_title(fr"{self.star_name} {self.obs_date} Na I Doublet", fontsize=24)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.set_ylabel("Normalized Flux [ergs s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=22)
            plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            plt.tick_params(axis='y', which='major', labelsize=20)
            plt.tick_params(axis='x', which='major', labelsize=20)
            ax.tick_params(axis='both', which='major', length=10, width=1)
            ax.set_xlim(5886, 5900)
            mask = (w >= 5886) & (w <= 5900)
            ax.set_ylim(np.min(self.dat[mask]) - 0.25, np.max(self.dat[mask]) + 0.25)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"APO_Spectra/SpectrumPlots/Na_I_Doublet/{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            wavs = pd.Series(w[mask])
            fluxes = pd.Series(self.dat[mask])
            df = pd.concat([wavs, fluxes], axis="columns")
            df.columns = ["Wavelength", "Flux"]
            df.to_csv(f"APO_Spectra/SpectraData/Na_I_Doublet/{self.star_name}_{self.obs_date}.csv",
                      index=False)

        if full_spec:
            if os.path.exists("APO_Spectra/SpectrumPlots/FullSpec"):
                pass
            else:
                os.mkdir("APO_Spectra/SpectrumPlots/FullSpec")
                print("-->FullSpec directory created, plots will be saved here!")

            # plt.rcParams['font.family'] = 'Trebuchet MS'
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

                # plt.rcParams['font.family'] = 'Trebuchet MS'
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

                # plt.rcParams['font.family'] = 'Trebuchet MS'
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

    def radial_velocity_bisector(self, print_rad_vel=False, print_crossings=False):
        """
        Obtains the radial velocity for a star by cross correlating two oppositely signed Gaussians to the H Alpha profile
        to sample the wings (similar to the bisector method as described in Wang, L. et al. AJ, 2023, 165, 203). It also
        plots this bisector velocity onto the spectrum and transforms the wavelength axis to a radial velocity axis. It
        then applies a barycentric correction to the derived radial velocity, and writes it into a datafile.

        Parameters:
            print_rad_vel (bool): Default=True, prints the radial velocity with the barycentric correction applied
            print_crossings: Flag to check if the function should print out the zero crossings
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

        v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs, fluxes, print_flag=print_crossings)

        sig_ind = np.where(wavs > 6600)[0]
        sig_cont = np.std(fluxes[sig_ind] - 1)

        res = analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                                gaussian_width_kms=gaussian_width,
                                                p=0.25,
                                                Ntop=5,
                                                M_inject=400,
                                                MC_samples=10_000)

        # err_v_bis = (sig_cont * gaussian_width) / (max(np.array(fluxes) - 1) * 0.25 * np.sqrt(-2 * np.log(0.25)))
        err_v_bis = float(res["sigma_v_median"])

        rad_vel_bc_corrected = v_bis + self.bc_corr / 1000  # self.bc_corr has a sign, so need to add (otherwise might add when negative)
        if print_rad_vel:
            print(f"Radial Velocity: \033[92m{rad_vel_bc_corrected:.3f} km/s\033[0m")

        # print(f"{self.obs_date} : Barycentric Correction = {self.bc_corr / 1000}")

        dlamb, coeff = wav_corr(np.array(wavs), self.bc_corr, v_bis)

        plot_ind = np.where((((np.array(wavs) - 6562.8) / 6562.8) * 3e5 > -500) &
                            (((np.array(wavs) - 6562.8) / 6562.8) * 3e5 < 500) &
                            ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]


        ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
        # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
        plt.subplots_adjust(hspace=0)
        fig.supxlabel("Wavelength [Å]", fontsize=22, y=0.03)
        ax[0].plot(dlamb, np.array(fluxes) - 1,
                   color="black", label="CCF")
        # ax[0].plot(v_grid, ker, c="r")
        ax[0].vlines(6562.8 + 6562.8 * rad_vel_bc_corrected / 3e5, 0.15 * max(fluxes - 1), 0.35 * max(fluxes - 1),
                     color="r", zorder=1, lw=3)
        ax[0].hlines(0.25 * max(fluxes - 1), dlamb[plot_ind[0]], dlamb[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
        ax[0].tick_params(axis='x', labelsize=20)
        ax[0].tick_params(axis='y', labelsize=20)
        ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=False, right=True, length=10,
                          width=1)
        # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        ax[0].set_ylabel("Normalized Flux", fontsize=22)
        ax[0].set_xlim(6562.8 - 6562.8 * 500 / 3e5, 6562.8 + 6562.8 * 500 / 3e5)
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

        ax[1].plot((6562.8 + 6562.8 * (v_grid + self.bc_corr / 1000) / 3e5)[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle",
                   zorder=10, linewidth=3)
        ax[1].set_ylabel("CCF", fontsize=22)
        ax[1].hlines(0, 6562.8 - 6562.8 * 500 / 3e5, 6562.8 + 6562.8 * 500 / 3e5, color="k", linestyle="--", zorder=0)
        ax[1].set_ylim(-1, 1)
        ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                          width=1)
        # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
        # ax.legend(loc="upper right", fontsize=22)
        # ax.set_ylim(-0.1, max(fluxes - 1)+0.2)
        fig.savefig(f"APO_Spectra/SpectrumPlots/RV_HAlpha_Bisector/RV_{self.star_name}_{self.obs_date}.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        wavs = pd.Series(dlamb)
        fluxes = pd.Series(fluxes)

        df = pd.concat([wavs, fluxes], axis="columns")
        df.columns = ["Wavelength", "Flux"]
        df.to_csv(f"APO_Spectra/SpectraData/HAlpha/{self.star_name}_{self.obs_date}_BCCorrected.csv",
                  index=False)

        if not os.path.exists("APO_Spectra/APOInventoryRV_Bisector.txt"):
            with open("APO_Spectra/APOInventoryRV_Bisector.txt", "w") as file:
                file.write(
                    f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
        else:
            with open("APO_Spectra/APOInventoryRV_Bisector.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.bc_corr_time) in line for line in jds):
                with open("APO_Spectra/APOInventoryRV_Bisector.txt", "a") as f:
                    f.write(
                        f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

        if not os.path.exists(f"APO_Spectra/RV_Measurements/{self.star_name}_RV.txt"):
            with open(f"APO_Spectra/RV_Measurements/{self.star_name}_RV.txt", "w") as file:
                file.write(f"{self.bc_corr_time},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
        else:
            with open(f"APO_Spectra/RV_Measurements/{self.star_name}_RV.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(self.bc_corr_time) in line for line in jds):
                with open(f"APO_Spectra/RV_Measurements/{self.star_name}_RV.txt", "a") as f:
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
        if os.path.exists("APO_Spectra/SpectrumPlots/RV_Doublet"):
            pass
        else:
            os.mkdir("APO_Spectra/SpectrumPlots/RV_Doublet")
            print("-->RV_Doublet directory created, plots will be saved here!")

        doublet_dat = pd.read_csv(f'APO_Spectra/SpectraData/Na_I_Doublet/{self.star_name}_{self.obs_date}.csv')

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

            err_v_doublet = 0.5  # Arbitrary error amount, need to check this
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
            fig.savefig(f"APO_Spectra/SpectrumPlots/RV_Doublet/RV_{self.star_name}_{self.obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            if not os.path.exists("APO_Spectra/APOInventoryRV_Na_I_Doublet.txt"):
                with open("APO_Spectra/APOInventoryRV_Na_I_Doublet.txt", "w") as file:
                    file.write(
                        f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open("APO_Spectra/APOInventoryRV_Na_I_Doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(self.bc_corr_time) in line for line in jds):
                    with open("APO_Spectra/APOInventoryRV_Na_I_Doublet.txt", "a") as f:
                        f.write(
                            f"{self.star_name},{self.bc_corr_time},{self.obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

            if not os.path.exists(f"APO_Spectra/RV_Measurements/{self.star_name}_RV_doublet.txt"):
                with open(f"APO_Spectra/RV_Measurements/{self.star_name}_RV_doublet.txt", "w") as file:
                    file.write(f"{self.bc_corr_time},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open(f"APO_Spectra/RV_Measurements/{self.star_name}_RV_doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(self.bc_corr_time) in line for line in jds):
                    with open(f"APO_Spectra/RV_Measurements/{self.star_name}_RV_doublet.txt", "a") as f:
                        f.write(f"{self.bc_corr_time},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

    @staticmethod
    def exp_time(v_mag_array, star_name_array):
        """
        Calculate and plot the Signal-to-Noise Ratio (SNR) as a function of exposure time for different stellar
        magnitudes based on the ARCES spectrograph's parameters. Derived from the IDL code by Doug Gies and James
        Davenport

        This function computes the required exposure times for a range of desired SNRs (from 10 to 500) for different
        stellar magnitudes. The results are plotted with logarithmic axes for both exposure time and SNR.

        Parameters:
            v_mag_array: a list of the V-band magnitudes of the stars you want to observe
            star_name_array (list): a list of the names of the stars you want to observe

        Returns:
            None. This function plots the results but does not return any values. It shows a plot with exposure time
            (log scale) on the y-axis and SNR (log scale) on the x-axis. The plot includes curves for different stellar
            magnitudes labels for each star.
        """
        snr_array = np.linspace(10, 150, 1000)
        echelle_c = 11.88

        exp_times = []
        for i in range(len(v_mag_array)):
            echelle_flux = 10 ** ((v_mag_array[i] - echelle_c) / (-2.5))
            exp_times.append(snr_array**2 / echelle_flux)

        exp_times = np.array(exp_times)
        # breakpoint()

        # Plotting SNR vs exposure time
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(len(v_mag_array)):
            ax.loglog(snr_array, exp_times[i], label=f'{star_name_array[i]}', linestyle='-', color='k', linewidth=3)

        ax.yaxis.get_offset_text().set_size(20)
        ax.vlines(130, min(exp_times[0]), max(exp_times[0]), color="xkcd:emerald", label="SNR=130", linewidth=3, alpha=0.8)
        ax.vlines(100, min(exp_times[0]), max(exp_times[0]), color="xkcd:goldenrod", label="SNR=100", linewidth=3, alpha=0.8)
        ax.vlines(80, min(exp_times[0]), max(exp_times[0]), color="xkcd:orange red", label="SNR=80", linewidth=3, alpha=0.8)
        ax.set_ylabel('Exposure Time [s]', fontsize=18)
        ax.set_xlabel('SNR', fontsize=18)
        ax.set_ylim(-1, 300)
        ax.set_xlim(10, 160)
        plt.tick_params(axis='y', which='major', labelsize=18)
        plt.tick_params(axis='x', which='major', labelsize=18)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='major', length=10, width=1)
        ax.tick_params(axis='both', which='minor', length=5, width=1)
        ax.grid(True, which="both", ls="--")
        ax.legend(fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
        # plt.close()

