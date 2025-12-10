import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import CubicSpline
from chironHelperFunctions import (shafter_bisector_velocity, clean_star_name,
                                   list_fits_files, clean_star_name_2, analytic_sigma_v_mc_from_nonparam,
                                   clean_star_name3, wav_corr, vel_to_wav_halpha, wav_to_vel_halpha)
from astropy.time import Time
from barycorrpy import get_BC_vel
import os
from SpectrumAnalyzer import BLUE, GREEN, RESET, YELLOW
import pandas as pd
import glob

plt.rcParams['font.family'] = 'Trebuchet MS'

def uves_spec(file_name):
    print(f"{BLUE}{file_name}{RESET}")
    with fits.open(file_name) as hdul:
        dat = hdul[1].data
        hdr = hdul[0].header
        # breakpoint()

    # XShooter spectra have wavelengths in nm instead of Å
    if hdr["INSTRUME"] == "XSHOOTER":
        dat["WAVE"][0] *= 10

    star_name = clean_star_name_2(hdr["OBJECT"])
    print(f"{YELLOW}{star_name}{RESET}")
    obs_jd = Time(hdr['DATE-OBS'], format='isot', scale='utc').jd
    obs_date = hdr['DATE-OBS'].split("T")[0]

    bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, lat=-24.627222, longi=-70.404167, alt=2635,
                         ephemeris="de430")[0][0]
    print(bc_corr)

    rec_list = [6410, 6425, 6500, 6540, 6600, 6615, 6650, 6695]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 6400) & (dat["WAVE"][0] < 6700))[0]

    # print(file_name)
    # print(ind)
    # print(dat["WAVE"][0][ind])

    x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
    cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX_REDUCED"][0][rec_points])
    y = cs(x)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX_REDUCED"][0][ind], c="k")
    ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX_REDUCED"][0][rec_points], facecolor='none', edgecolor="xkcd:goldenrod",
               linewidth=5, zorder=10, s=300)
    ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"UVES/Plots/RawSpec/UVES_{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted raw spectrum!")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX_REDUCED"][0][ind]/y, c="k")
    # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
    # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Normalized Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"UVES/Plots/NormSpec/UVES_{star_name}_{obs_date}_normalized_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    wavs = pd.Series(dat["WAVE"][0][ind])
    fluxes = pd.Series((dat["FLUX_REDUCED"][0][ind])/y)
    df = pd.concat([wavs, fluxes], axis="columns")
    df.columns = ["Wavelength", "Flux"]
    df.to_csv(f"UVES/SpectrumData/HAlpha/{star_name}_{obs_date}_{obs_jd:.3f}.csv",
              index=False)

    print("Plotted normalized spectrum!")

    wavs = dat["WAVE"][0][ind]
    fluxes = dat["FLUX_REDUCED"][0][ind]/y

    v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs, fluxes, print_flag=False)

    sig_ind = np.where((wavs > 6600) & (wavs < 6625))[0]

    res = analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                            gaussian_width_kms=gaussian_width,
                                            p=0.25,
                                            Ntop=5,
                                            M_inject=400,
                                            MC_samples=10_000)


    err_v_bis = float(res["sigma_v_median"])

    rad_vel_bc_corrected = v_bis + bc_corr/1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

    dlamb, coeff = wav_corr(np.array(wavs), bc_corr, v_bis)

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
    ax[0].vlines(6562.8 + 6562.8 * v_bis / 3e5, 0.15 * max(fluxes - 1), 0.35 * max(fluxes - 1), color="r", zorder=1,
                 lw=3)
    ax[0].hlines(0.25 * max(fluxes - 1), dlamb[plot_ind[0]], dlamb[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=False, right=True, length=10,
                      width=1)
    # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].set_ylabel("Normalized Flux", fontsize=22)
    ax[0].set_xlim(6562.8 - 6562.8 * 500 / 3e5, 6562.8 + 6562.8 * 500 / 3e5)
    ax[0].text(0.73, 0.8, fr"{clean_star_name3(star_name)} H$\alpha$"
                          f"\nHJD {obs_jd:.4f}\n"
                          fr"RV$_{{\text{{BC}}}}$ = {rad_vel_bc_corrected:.3f}±{err_v_bis:.3f} km/s",
               color="k", fontsize=18, transform=ax[0].transAxes,
               bbox=dict(
                   facecolor='white',  # Box background color
                   edgecolor='black',  # Box border color
                   boxstyle='square,pad=0.3',  # Rounded box with padding
                   alpha=0.9  # Slight transparency
               )
               )

    secax_x = ax[0].secondary_xaxis("top", functions=(wav_to_vel_halpha, vel_to_wav_halpha))
    secax_x.set_xlabel(r"Radial Velocity [km/s]", fontsize=22)
    secax_x.tick_params(labelsize=22, which='both')
    secax_x.tick_params(axis='both', which='both', direction='in', length=10, width=1)

    ax[1].plot((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=2)
    ax[1].set_ylabel("CCF", fontsize=22)
    ax[1].hlines(0, min((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind]), max((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind]),
                 color="k", linestyle="--", zorder=0)
    ax[1].set_ylim(-1, 1)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    fig.savefig(f"UVES/Plots/RV_HAlpha_Bisector/UVES_RV_{star_name}_{obs_date}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if not os.path.exists("UVES/UVES_RV_Bisector.txt"):
        with open("UVES/UVES_RV_Bisector.txt", "w") as file:
            file.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open("UVES/UVES_RV_Bisector.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open("UVES/UVES_RV_Bisector.txt", "a") as f:
                f.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"UVES/RV_Measurements/{star_name}_RV.txt"):
        with open(f"UVES/RV_Measurements/{star_name}_RV.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"UVES/RV_Measurements/{star_name}_RV.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"UVES/RV_Measurements/{star_name}_RV.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    print("Plotted bisector!")

    # He I 5016 Line
    # breakpoint()
    rec_list = [5002, 5005, 5010, 5012, 5025, 5030, 5035, 5040, 5045, 5048]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 5000) & (dat["WAVE"][0] < 5050))[0]

    if len(ind) > 0:

        print(f"{GREEN}Running He 5016 Analysis{RESET}")

        x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
        cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX_REDUCED"][0][rec_points])
        y = cs(x)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX_REDUCED"][0][ind], c="k")
        ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX_REDUCED"][0][rec_points], facecolor='none',
                   edgecolor="xkcd:goldenrod",
                   linewidth=5, zorder=10, s=300)
        ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)
        fig.savefig(f"UVES/Plots/RawSpec/UVES_{star_name}_{obs_date}_raw_spec_He5016.pdf", bbox_inches="tight", dpi=300)
        plt.close()

        print("Plotted raw spectrum!")

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX_REDUCED"][0][ind] / y, c="k")
        ax.scatter(dat["WAVE"][0][ind][np.argmin(dat["FLUX_REDUCED"][0][ind] / y)], (dat["FLUX_REDUCED"][0][ind] / y)[np.argmin(dat["FLUX_REDUCED"][0][ind] / y)], c="r")
        ax.vlines(5015.6783, min(dat["FLUX_REDUCED"][0][ind] / y), max(dat["FLUX_REDUCED"][0][ind] / y), color='dodgerblue', lw=2)
        # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
        # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Normalized Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)

        rv = (dat["WAVE"][0][ind][np.argmin(dat["FLUX_REDUCED"][0][ind] / y)] - 5015.6783) / 5015.6783 * 3e5
        print(f"RV = {YELLOW}{rv:.3f}{RESET}")

        rad_vel_he_corrected = rv + bc_corr / 1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

        print(f"{obs_jd},{rad_vel_bc_corrected:.3f},5")

        err_v_he = 5

        ax.text(0.76, 0.15, fr"{clean_star_name3(star_name)} He I 5016"
                            f"\nHJD {obs_jd:.4f}\nRV = {rv:.3f}±{err_v_he:.3f} km/s",
                color="k", fontsize=18, transform=ax.transAxes,
                bbox=dict(
                    facecolor='white',  # Box background color
                    edgecolor='black',  # Box border color
                    boxstyle='square,pad=0.3',  # Rounded box with padding
                    alpha=0.9  # Slight transparency
                )
                )

        fig.savefig(f"UVES/Plots/NormSpec/UVES_{star_name}_{obs_date}_normalized_spec_He5016.pdf", bbox_inches="tight", dpi=300)
        plt.close()

        # Na I Doublet for offset calculation
        # breakpoint()
        rec_list = [5886.5, 5888, 5888.4, 5889.4, 5894, 5894.735, 5895.5, 5898.7, 5899.5]
        rec_points = []
        for item in rec_list:
            rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

        ind = np.where((dat["WAVE"][0] > 5886) & (dat["WAVE"][0] < 5900))[0]

        if len(ind) > 0:
            print(f"{GREEN}Running Na I Doublet Analysis{RESET}")

            x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
            cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX_REDUCED"][0][rec_points])
            y = cs(x)

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(dat["WAVE"][0][ind], dat["FLUX_REDUCED"][0][ind], c="k")
            ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX_REDUCED"][0][rec_points], facecolor='none',
                       edgecolor="xkcd:goldenrod",
                       linewidth=5, zorder=10, s=300)
            ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
            ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                           width=1)
            ax.set_ylabel("Flux", fontsize=22)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.yaxis.get_offset_text().set_size(20)
            fig.savefig(f"UVES/Plots/RawSpec/UVES_{star_name}_{obs_date}_raw_spec_NaIDoublet.pdf", bbox_inches="tight",
                        dpi=300)
            plt.close()

            print("Plotted raw spectrum!")

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(dat["WAVE"][0][ind], dat["FLUX_REDUCED"][0][ind] / y, c="k")
            ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                           width=1)
            ax.set_ylabel("Normalized Flux", fontsize=22)
            ax.set_xlabel("Wavelength [Å]", fontsize=22)
            ax.yaxis.get_offset_text().set_size(20)

            fig.savefig(f"UVES/Plots/NormSpec/UVES_{star_name}_{obs_date}_normalized_spec_NaIDoublet.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

        print("Plotted normalized spectrum!")

        wavs = dat["WAVE"][0][ind]
        fluxes = dat["FLUX_REDUCED"][0][ind] / y

        ism_ind1 = np.where((wavs > 5889.95-1) & (wavs < 5889.95+1))[0]
        ism_ind2 = np.where((wavs > 5895.92 - 1) & (wavs < 5895.92 + 1))[0]
        ism_ind = np.concat([ism_ind1, ism_ind2])

        inv_flux = -(fluxes - 1)
        # peaks, props = find_peaks(inv_flux, height=0.2)

        peaks = [np.argmin(fluxes[ism_ind1]), np.argmin(fluxes[ism_ind2])]

        # breakpoint()

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
            rad_vel_doublet_corrected = doublet_vel + (bc_corr / 1000)  # self.bc_corr has a sign, so need to add (otherwise might add when negative)

            dlamb, coeff = wav_corr(np.array(wavs), bc_corr, doublet_vel)

            # breakpoint()

            fig, ax = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
            plt.subplots_adjust(wspace=0)
            fig.supylabel("Normalized Flux", fontsize=22, x=0.06)
            ax[0].plot(dlamb, np.array(fluxes) - 1,
                       color="black", label="CCF")
            ax[0].scatter(dlamb[ism_ind1][peaks[0]], (np.array(fluxes) - 1)[ism_ind1][peaks[0]], c="r")
            ax[0].vlines(wavs[ism_ind1[0]], min(np.array(fluxes) - 1)-0.2, max(np.array(fluxes) - 1)+0.2, color="xkcd:periwinkle")
            ax[0].vlines(wavs[ism_ind1[-1]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            # ax[0].axvspan(wavs[ism_ind1[0]], wavs[ism_ind1[-1]], color="xkcd:periwinkle")
            # ax[0].plot(v_grid, ker, c="r")
            # ax[0].vlines(5889.95 + 5889.95*vel_d1/3e5 - 5889.95*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                              width=1)
            ax[0].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[0].set_ylabel()
            ax[0].set_xlim(5889.95 - 5889.95 * 220 / 3e5, 5889.95 + 5889.95 * 220 / 3e5)
            ax[0].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
            ax[0].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_1$"
                                   f"\nHJD {obs_jd:.4f}\nRV = {vel_d1 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[0].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )

            ax[1].plot(dlamb, np.array(fluxes) - 1, color="black", label="CCF")
            ax[1].scatter(dlamb[ism_ind2][peaks[1]], (np.array(fluxes) - 1)[ism_ind2][peaks[1]], c="r")
            ax[1].vlines(wavs[ism_ind2[0]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            ax[1].vlines(wavs[ism_ind2[-1]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            # ax[1].axvspan(wavs[ism_ind2[0]], wavs[ism_ind2[-1]], color="xkcd:periwinkle")
            # ax[1].vlines(5895.92 + 5895.92*vel_d2/3e5 - 5895.92*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                              width=1)
            ax[1].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[1].set_ylabel()
            ax[1].set_xlim(5895.92 - 5895.92 * 220 / 3e5, 5895.92 + 5895.92 * 220 / 3e5)
            ax[1].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
            ax[1].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_2$"
                                   f"\nHJD {obs_jd:.4f}\nRV = {vel_d2 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[1].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )
            fig.savefig(f"UVES/Plots/RV_Na_I_Doublet/RV_{star_name}_{obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            if not os.path.exists("UVES/UVESInventoryRV_Na_I_Doublet.txt"):
                with open("UVES/UVESInventoryRV_Na_I_Doublet.txt", "w") as file:
                    file.write(
                        f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open("UVES/UVESInventoryRV_Na_I_Doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(obs_jd) in line for line in jds):
                    with open("UVES/UVESInventoryRV_Na_I_Doublet.txt", "a") as f:
                        f.write(
                            f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

            if not os.path.exists(f"UVES/RV_Measurements/{star_name}_RV_doublet.txt"):
                with open(f"UVES/RV_Measurements/{star_name}_RV_doublet.txt", "w") as file:
                    file.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open(f"UVES/RV_Measurements/{star_name}_RV_doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(obs_jd) in line for line in jds):
                    with open(f"UVES/RV_Measurements/{star_name}_RV_doublet.txt", "a") as f:
                        f.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

    else:
        pass

    print(f"{GREEN}={RESET}"*10 + f"{GREEN}Done!{RESET}" + f"{GREEN}={RESET}"*10)


def iacob_spec(file_name):
    print(f"{BLUE}{file_name}{RESET}")
    with fits.open(file_name) as hdul:
        dat = hdul[1].data
        hdr = hdul[0].header
        # breakpoint()

    # XShooter spectra have wavelengths in nm instead of Å
    if hdr["INSTRUME"] == "XSHOOTER":
        dat["WAVE"][0] *= 10

    star_name = clean_star_name(hdr["OBJECT"])
    obs_jd = Time(hdr['DATE-OBS'], format='isot', scale='utc').jd
    obs_date = hdr['DATE-OBS'].split("T")[0]

    bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, obsname="La Silla Observatory (ESO)", ephemeris="de430")[0][0]

    rec_list = [6410, 6425, 6500, 6540, 6600, 6615, 6650, 6695]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 6400) & (dat["WAVE"][0] < 6700))[0]

    # print(file_name)
    # print(ind)
    # print(dat["WAVE"][0][ind])

    x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
    cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
    y = cs(x)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind], c="k")
    ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none', edgecolor="xkcd:goldenrod",
               linewidth=5, zorder=10, s=300)
    ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"UVES/Plots/RawSpec/UVES_{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted raw spectrum!")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind]/y, c="k")
    # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
    # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Normalized Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"UVES/Plots/NormSpec/UVES_{star_name}_{obs_date}_normalized_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted normalized spectrum!")

    wavs = dat["WAVE"][0][ind]
    fluxes = dat["FLUX"][0][ind]/y

    v_bis, v_grid, ccf = shafter_bisector_velocity(wavs, fluxes, print_flag=False)

    plot_ind = np.where((((np.array(wavs) - 6562.8) / 6562.8) * 3e5 > -500) &
                                (((np.array(wavs) - 6562.8) / 6562.8) * 3e5 < 500) &
                                ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]
    ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
    # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]

    sig_ind = np.where((wavs > 6600) & (wavs < 6625))[0]
    sig_cont = np.std(fluxes[sig_ind] - 1)

    err_v_bis = 0.5
    rad_vel_bc_corrected = v_bis + bc_corr/1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

    fig, ax = plt.subplots(2,1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
    plt.subplots_adjust(hspace=0)
    fig.supxlabel("Radial Velocity [km s$^{-1}$]", fontsize=22, y=0.05)
    ax[0].plot(((np.array(wavs) - 6562.8) / 6562.8) * 3e5, np.array(fluxes) - 1,
            color="black", label="CCF")
    # ax[0].plot(v_grid, ker, c="r")
    ax[0].vlines(v_bis, 0.23 * max(fluxes - 1), 0.27 * max(fluxes - 1), color="r", zorder=1)
    ax[0].hlines(0.25 * max(fluxes - 1), (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[0]],
              (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].set_ylabel("Normalized Flux", fontsize=22)
    ax[0].set_xlim(-500, 500)
    ax[0].text(0.78, 0.8, fr"{hdr['OBJECT']} H$\alpha$"
                      f"\nHJD {obs_jd:.4f}\nRV = {v_bis:.3f}±{err_v_bis:.3f} km/s",
            color="k", fontsize=18, transform=ax[0].transAxes,
            bbox=dict(
                facecolor='white',  # Box background color
                edgecolor='black',  # Box border color
                boxstyle='square,pad=0.3',  # Rounded box with padding
                alpha=0.9  # Slight transparency
            )
            )
    ax[1].plot(v_grid[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=2)
    ax[1].set_ylabel("CCF", fontsize=22)
    ax[1].hlines(0, min(v_grid[ccf_ind]), max(v_grid[ccf_ind]), color="k", linestyle="--", zorder=0)
    ax[1].set_ylim(-1, 1)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
    # ax.legend(loc="upper right", fontsize=22)
    # ax.set_ylim(-0.1, max(fluxes - 1)+0.2)
    fig.savefig(f"UVES/Plots/RV_HAlpha_Bisector/UVES_RV_{star_name}_{obs_date}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if not os.path.exists("UVES/UVES_RV_Bisector.txt"):
        with open("UVES/UVES_RV_Bisector.txt", "w") as file:
            file.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open("UVES/UVES_RV_Bisector.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open("UVES/UVES_RV_Bisector.txt", "a") as f:
                f.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"UVES/RV_Measurements/{star_name}_RV.txt"):
        with open(f"UVES/RV_Measurements/{star_name}_RV.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"UVES/RV_Measurements/{star_name}_RV.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"UVES/RV_Measurements/{star_name}_RV.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    print("Plotted bisector!")

    print(f"{GREEN}={RESET}"*10 + f"{GREEN}Done!{RESET}" + f"{GREEN}={RESET}"*10)


def feros_spec(file_name):
    print(f"{BLUE}{file_name}{RESET}")
    with fits.open(file_name) as hdul:
        dat = hdul[1].data
        hdr = hdul[0].header

    # XShooter spectra have wavelengths in nm instead of Å
    if hdr["INSTRUME"] == "XSHOOTER":
        dat["WAVE"][0] *= 10

    star_name = clean_star_name_2(hdr["OBJECT"])
    obs_jd = Time(hdr['DATE-OBS'], format='isot', scale='utc').jd
    obs_date = hdr['DATE-OBS'].split("T")[0]

    # star_name = "HD 63462"

    bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, obsname="La Silla Observatory (ESO)", ephemeris="de430")[0][0]

    rec_list = [6410, 6425, 6500, 6540, 6600, 6615, 6650, 6695]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 6400) & (dat["WAVE"][0] < 6700))[0]

    # print(file_name)
    # print(ind)
    # print(dat["WAVE"][0][ind])

    x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
    cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
    y = cs(x)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind], c="k")
    ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none', edgecolor="xkcd:goldenrod",
               linewidth=5, zorder=10, s=300)
    ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"FEROS/Plots/RawSpec/FEROS_{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted raw spectrum!")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind]/y, c="k")
    # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
    # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Normalized Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"FEROS/Plots/NormSpec/FEROS_{star_name}_{obs_date}_normalized_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted normalized spectrum!")

    wavs = dat["WAVE"][0][ind]
    fluxes = dat["FLUX"][0][ind]/y

    v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs, fluxes, print_flag=False)

    plot_ind = np.where((((np.array(wavs) - 6562.8) / 6562.8) * 3e5 > -500) &
                                (((np.array(wavs) - 6562.8) / 6562.8) * 3e5 < 500) &
                                ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]
    ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
    # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]

    sig_ind = np.where((wavs > 6600) & (wavs < 6625))[0]
    sig_cont = np.std(fluxes[sig_ind] - 1)

    res = analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                            gaussian_width_kms=gaussian_width,
                                            p=0.25,
                                            Ntop=5,
                                            M_inject=400,
                                            MC_samples=10_000)

    # err_v_bis = (sig_cont * gaussian_width) / (max(np.array(fluxes) - 1) * 0.25 * np.sqrt(-2 * np.log(0.25)))
    err_v_bis = float(res["sigma_v_median"])

    # err_v_bis = 0.5
    rad_vel_bc_corrected = v_bis + bc_corr/1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

    fig, ax = plt.subplots(2,1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
    plt.subplots_adjust(hspace=0)
    fig.supxlabel("Radial Velocity [km s$^{-1}$]", fontsize=22, y=0.05)
    ax[0].plot(((np.array(wavs) - 6562.8) / 6562.8) * 3e5, np.array(fluxes) - 1,
            color="black", label="CCF")
    # ax[0].plot(v_grid, ker, c="r")
    ax[0].vlines(v_bis, 0.23 * max(fluxes - 1), 0.27 * max(fluxes - 1), color="r", zorder=1)
    ax[0].hlines(0.25 * max(fluxes - 1), (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[0]],
              (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].set_ylabel("Normalized Flux", fontsize=22)
    ax[0].set_xlim(-500, 500)
    ax[0].text(0.78, 0.8, fr"{hdr['OBJECT']} H$\alpha$"
                      f"\nHJD {obs_jd:.4f}\nRV = {rad_vel_bc_corrected:.3f}±{err_v_bis:.3f} km/s",
            color="k", fontsize=18, transform=ax[0].transAxes,
            bbox=dict(
                facecolor='white',  # Box background color
                edgecolor='black',  # Box border color
                boxstyle='square,pad=0.3',  # Rounded box with padding
                alpha=0.9  # Slight transparency
            )
            )
    ax[1].plot(v_grid[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=2)
    ax[1].set_ylabel("CCF", fontsize=22)
    ax[1].hlines(0, min(v_grid[ccf_ind]), max(v_grid[ccf_ind]), color="k", linestyle="--", zorder=0)
    ax[1].set_ylim(-1, 1)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
    # ax.legend(loc="upper right", fontsize=22)
    # ax.set_ylim(-0.1, max(fluxes - 1)+0.2)
    fig.savefig(f"FEROS/Plots/RV_HAlpha_Bisector/FEROS_RV_{star_name}_{obs_date}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if not os.path.exists("FEROS/FEROS_RV_Bisector.txt"):
        with open("FEROS/FEROS_RV_Bisector.txt", "w") as file:
            file.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open("FEROS/FEROS_RV_Bisector.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open("FEROS/FEROS_RV_Bisector.txt", "a") as f:
                f.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"FEROS/RV_Measurements/{star_name}_RV.txt"):
        with open(f"FEROS/RV_Measurements/{star_name}_RV.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"FEROS/RV_Measurements/{star_name}_RV.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"FEROS/RV_Measurements/{star_name}_RV.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    print("Plotted bisector!")

    # Na I Doublet for offset calculation
    # breakpoint()
    rec_list = [5886.5, 5888, 5888.4, 5889.4, 5894, 5894.735, 5895.5, 5898.7, 5899.5]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 5886) & (dat["WAVE"][0] < 5900))[0]

    if len(ind) > 0:
        print(f"{GREEN}Running Na I Doublet Analysis{RESET}")

        x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
        cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
        y = cs(x)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind], c="k")
        ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none',
                   edgecolor="xkcd:goldenrod",
                   linewidth=5, zorder=10, s=300)
        ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)
        fig.savefig(f"FEROS/Plots/RawSpec/FEROS_{star_name}_{obs_date}_raw_spec_NaIDoublet.pdf", bbox_inches="tight",
                    dpi=300)
        plt.close()

        print("Plotted raw spectrum!")

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind] / y, c="k")
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Normalized Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)

        fig.savefig(f"FEROS/Plots/NormSpec/FEROS_{star_name}_{obs_date}_normalized_spec_NaIDoublet.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        print("Plotted normalized spectrum!")

        wavs = dat["WAVE"][0][ind]
        fluxes = dat["FLUX"][0][ind] / y

        ism_ind1 = np.where((wavs > 5889.95 - 1) & (wavs < 5889.95 + 1))[0]
        ism_ind2 = np.where((wavs > 5895.92 - 1) & (wavs < 5895.92 + 1))[0]
        ism_ind = np.concat([ism_ind1, ism_ind2])

        inv_flux = -(fluxes - 1)
        # peaks, props = find_peaks(inv_flux, height=0.2)

        peaks = [np.argmin(fluxes[ism_ind1]), np.argmin(fluxes[ism_ind2])]

        # breakpoint()

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
            rad_vel_doublet_corrected = doublet_vel + bc_corr / 1000  # self.bc_corr has a sign, so need to add (otherwise might add when negative)

            dlamb, coeff = wav_corr(np.array(wavs), bc_corr, doublet_vel)

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
            # ax[0].axvspan(wavs[ism_ind1[0]], wavs[ism_ind1[-1]], color="xkcd:periwinkle")
            # ax[0].plot(v_grid, ker, c="r")
            # ax[0].vlines(5889.95 + 5889.95*vel_d1/3e5 - 5889.95*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                              width=1)
            ax[0].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[0].set_ylabel()
            ax[0].set_xlim(5889.95 - 5889.95 * 220 / 3e5, 5889.95 + 5889.95 * 220 / 3e5)
            ax[0].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
            ax[0].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_1$"
                                   f"\nHJD {obs_jd:.4f}\nRV = {vel_d1 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[0].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )

            ax[1].plot(dlamb, np.array(fluxes) - 1, color="black", label="CCF")
            ax[1].scatter(dlamb[ism_ind2][peaks[1]], (np.array(fluxes) - 1)[ism_ind2][peaks[1]], c="r")
            ax[1].vlines(wavs[ism_ind2[0]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            ax[1].vlines(wavs[ism_ind2[-1]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            # ax[1].axvspan(wavs[ism_ind2[0]], wavs[ism_ind2[-1]], color="xkcd:periwinkle")
            # ax[1].vlines(5895.92 + 5895.92*vel_d2/3e5 - 5895.92*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                              width=1)
            ax[1].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[1].set_ylabel()
            ax[1].set_xlim(5895.92 - 5895.92 * 220 / 3e5, 5895.92 + 5895.92 * 220 / 3e5)
            ax[1].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
            ax[1].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_2$"
                                   f"\nHJD {obs_jd:.4f}\nRV = {vel_d2 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[1].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )
            fig.savefig(f"FEROS/Plots/RV_Na_I_Doublet/RV_{star_name}_{obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            if not os.path.exists("FEROS/FEROSInventoryRV_Na_I_Doublet.txt"):
                with open("FEROS/FEROSInventoryRV_Na_I_Doublet.txt", "w") as file:
                    file.write(
                        f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open("FEROS/FEROSInventoryRV_Na_I_Doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(obs_jd) in line for line in jds):
                    with open("FEROS/FEROSInventoryRV_Na_I_Doublet.txt", "a") as f:
                        f.write(
                            f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

            if not os.path.exists(f"FEROS/RV_Measurements/{star_name}_RV_doublet.txt"):
                with open(f"FEROS/RV_Measurements/{star_name}_RV_doublet.txt", "w") as file:
                    file.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open(f"FEROS/RV_Measurements/{star_name}_RV_doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(obs_jd) in line for line in jds):
                    with open(f"FEROS/RV_Measurements/{star_name}_RV_doublet.txt", "a") as f:
                        f.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

    print(f"{GREEN}={RESET}"*10 + f"{GREEN}Done!{RESET}" + f"{GREEN}={RESET}"*10)


def harps_spec(file_name):
    print(f"{BLUE}{file_name}{RESET}")
    with fits.open(file_name) as hdul:
        dat = hdul[1].data
        hdr = hdul[0].header

    # XShooter spectra have wavelengths in nm instead of Å
    if hdr["INSTRUME"] == "XSHOOTER":
        dat["WAVE"][0] *= 10

    star_name = clean_star_name_2(hdr["OBJECT"])
    obs_jd = Time(hdr['DATE-OBS'], format='isot', scale='utc').jd
    obs_date = hdr['DATE-OBS'].split("T")[0]

    bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, obsname="La Silla Observatory (ESO)", ephemeris="de430")[0][0]

    rec_list = [6410, 6425, 6500, 6540, 6600, 6615, 6650, 6695]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 6400) & (dat["WAVE"][0] < 6700))[0]

    # print(file_name)
    # print(ind)
    # print(dat["WAVE"][0][ind])

    x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
    cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
    y = cs(x)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind], c="k")
    ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none', edgecolor="xkcd:goldenrod",
               linewidth=5, zorder=10, s=300)
    ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"HARPS/Plots/RawSpec/HARPS_{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted raw spectrum!")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind]/y, c="k")
    # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
    # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Normalized Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"HARPS/Plots/NormSpec/HARPS_{star_name}_{obs_date}_normalized_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted normalized spectrum!")

    wavs = dat["WAVE"][0][ind]
    fluxes = dat["FLUX"][0][ind]/y

    v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs, fluxes, print_flag=False)

    plot_ind = np.where((((np.array(wavs) - 6562.8) / 6562.8) * 3e5 > -500) &
                                (((np.array(wavs) - 6562.8) / 6562.8) * 3e5 < 500) &
                                ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]
    ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
    # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]

    sig_ind = np.where((wavs > 6600) & (wavs < 6625))[0]
    sig_cont = np.std(fluxes[sig_ind] - 1)

    res = analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                            gaussian_width_kms=gaussian_width,
                                            p=0.25,
                                            Ntop=5,
                                            M_inject=400,
                                            MC_samples=10_000)

    # err_v_bis = (sig_cont * gaussian_width) / (max(np.array(fluxes) - 1) * 0.25 * np.sqrt(-2 * np.log(0.25)))
    err_v_bis = float(res["sigma_v_median"])

    rad_vel_bc_corrected = v_bis + bc_corr/1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

    fig, ax = plt.subplots(2,1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
    plt.subplots_adjust(hspace=0)
    fig.supxlabel("Radial Velocity [km s$^{-1}$]", fontsize=22, y=0.05)
    ax[0].plot(((np.array(wavs) - 6562.8) / 6562.8) * 3e5, np.array(fluxes) - 1,
            color="black", label="CCF")
    # ax[0].plot(v_grid, ker, c="r")
    ax[0].vlines(v_bis, 0.23 * max(fluxes - 1), 0.27 * max(fluxes - 1), color="r", zorder=1)
    ax[0].hlines(0.25 * max(fluxes - 1), (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[0]],
              (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].set_ylabel("Normalized Flux", fontsize=22)
    ax[0].set_xlim(-500, 500)
    ax[0].text(0.78, 0.8, fr"{hdr['OBJECT']} H$\alpha$"
                      f"\nHJD {obs_jd:.4f}\nRV = {v_bis:.3f}±{err_v_bis:.3f} km/s",
            color="k", fontsize=18, transform=ax[0].transAxes,
            bbox=dict(
                facecolor='white',  # Box background color
                edgecolor='black',  # Box border color
                boxstyle='square,pad=0.3',  # Rounded box with padding
                alpha=0.9  # Slight transparency
            )
            )
    ax[1].plot(v_grid[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=2)
    ax[1].set_ylabel("CCF", fontsize=22)
    ax[1].hlines(0, min(v_grid[ccf_ind]), max(v_grid[ccf_ind]), color="k", linestyle="--", zorder=0)
    ax[1].set_ylim(-1, 1)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
    # ax.legend(loc="upper right", fontsize=22)
    # ax.set_ylim(-0.1, max(fluxes - 1)+0.2)
    fig.savefig(f"HARPS/Plots/RV_HAlpha_Bisector/HARPS_RV_{star_name}_{obs_date}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if not os.path.exists("HARPS/HARPS_RV_Bisector.txt"):
        with open("HARPS/HARPS_RV_Bisector.txt", "w") as file:
            file.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open("HARPS/HARPS_RV_Bisector.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open("HARPS/HARPS_RV_Bisector.txt", "a") as f:
                f.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"HARPS/RV_Measurements/{star_name}_RV.txt"):
        with open(f"HARPS/RV_Measurements/{star_name}_RV.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"HARPS/RV_Measurements/{star_name}_RV.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"HARPS/RV_Measurements/{star_name}_RV.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    print("Plotted bisector!")

    # Na I Doublet for offset calculation
    # breakpoint()
    rec_list = [5886.5, 5888, 5888.4, 5889.4, 5894, 5894.735, 5895.5, 5898.7, 5899.5]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 5886) & (dat["WAVE"][0] < 5900))[0]

    if len(ind) > 0:
        print(f"{GREEN}Running Na I Doublet Analysis{RESET}")

        x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
        cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
        y = cs(x)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind], c="k")
        ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none',
                   edgecolor="xkcd:goldenrod",
                   linewidth=5, zorder=10, s=300)
        ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)
        fig.savefig(f"HARPS/Plots/RawSpec/HARPS_{star_name}_{obs_date}_raw_spec_NaIDoublet.pdf", bbox_inches="tight",
                    dpi=300)
        plt.close()

        print("Plotted raw spectrum!")

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind] / y, c="k")
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Normalized Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)

        fig.savefig(f"HARPS/Plots/NormSpec/HARPS_{star_name}_{obs_date}_normalized_spec_NaIDoublet.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

    print("Plotted normalized spectrum!")

    wavs = dat["WAVE"][0][ind]
    fluxes = dat["FLUX"][0][ind] / y

    ism_ind1 = np.where((wavs > 5889.95 - 3) & (wavs < 5889.95 + 3))[0]
    ism_ind2 = np.where((wavs > 5895.92 - 3) & (wavs < 5895.92 + 3))[0]
    ism_ind = np.concat([ism_ind1, ism_ind2])

    inv_flux = -(fluxes - 1)
    # peaks, props = find_peaks(inv_flux, height=0.2)

    peaks = [np.argmin(fluxes[ism_ind1]), np.argmin(fluxes[ism_ind2])]

    # breakpoint()

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
        rad_vel_doublet_corrected = doublet_vel + (
                    bc_corr / 1000)  # self.bc_corr has a sign, so need to add (otherwise might add when negative)

        dlamb, coeff = wav_corr(np.array(wavs), bc_corr, doublet_vel)

        # breakpoint()

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
        # ax[0].axvspan(wavs[ism_ind1[0]], wavs[ism_ind1[-1]], color="xkcd:periwinkle")
        # ax[0].plot(v_grid, ker, c="r")
        # ax[0].vlines(5889.95 + 5889.95*vel_d1/3e5 - 5889.95*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
        ax[0].tick_params(axis='x', labelsize=20)
        ax[0].tick_params(axis='y', labelsize=20)
        ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                          width=1)
        ax[0].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        # ax[0].set_ylabel()
        ax[0].set_xlim(5889.95 - 5889.95 * 220 / 3e5, 5889.95 + 5889.95 * 220 / 3e5)
        ax[0].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
        ax[0].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_1$"
                               f"\nHJD {obs_jd:.4f}\nRV = {vel_d1 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                   color="k", fontsize=18, transform=ax[0].transAxes,
                   bbox=dict(
                       facecolor='white',  # Box background color
                       edgecolor='black',  # Box border color
                       boxstyle='square,pad=0.3',  # Rounded box with padding
                       alpha=0.9  # Slight transparency
                   )
                   )

        ax[1].plot(dlamb, np.array(fluxes) - 1, color="black", label="CCF")
        ax[1].scatter(dlamb[ism_ind2][peaks[1]], (np.array(fluxes) - 1)[ism_ind2][peaks[1]], c="r")
        ax[1].vlines(wavs[ism_ind2[0]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                     color="xkcd:periwinkle")
        ax[1].vlines(wavs[ism_ind2[-1]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                     color="xkcd:periwinkle")
        # ax[1].axvspan(wavs[ism_ind2[0]], wavs[ism_ind2[-1]], color="xkcd:periwinkle")
        # ax[1].vlines(5895.92 + 5895.92*vel_d2/3e5 - 5895.92*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
        ax[1].tick_params(axis='x', labelsize=20)
        ax[1].tick_params(axis='y', labelsize=20)
        ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                          width=1)
        ax[1].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        # ax[1].set_ylabel()
        ax[1].set_xlim(5895.92 - 5895.92 * 220 / 3e5, 5895.92 + 5895.92 * 220 / 3e5)
        ax[1].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
        ax[1].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_2$"
                               f"\nHJD {obs_jd:.4f}\nRV = {vel_d2 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                   color="k", fontsize=18, transform=ax[1].transAxes,
                   bbox=dict(
                       facecolor='white',  # Box background color
                       edgecolor='black',  # Box border color
                       boxstyle='square,pad=0.3',  # Rounded box with padding
                       alpha=0.9  # Slight transparency
                   )
                   )
        fig.savefig(f"HARPS/Plots/RV_Na_I_Doublet/RV_{star_name}_{obs_date}.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        if not os.path.exists("HARPS/HARPSInventoryRV_Na_I_Doublet.txt"):
            with open("HARPS/HARPSInventoryRV_Na_I_Doublet.txt", "w") as file:
                file.write(
                    f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
        else:
            with open("HARPS/HARPSInventoryRV_Na_I_Doublet.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(obs_jd) in line for line in jds):
                with open("HARPS/HARPSInventoryRV_Na_I_Doublet.txt", "a") as f:
                    f.write(
                        f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

        if not os.path.exists(f"HARPS/RV_Measurements/{star_name}_RV_doublet.txt"):
            with open(f"HARPS/RV_Measurements/{star_name}_RV_doublet.txt", "w") as file:
                file.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
        else:
            with open(f"HARPS/RV_Measurements/{star_name}_RV_doublet.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(obs_jd) in line for line in jds):
                with open(f"HARPS/RV_Measurements/{star_name}_RV_doublet.txt", "a") as f:
                    f.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

    print(f"{GREEN}={RESET}"*10 + f"{GREEN}Done!{RESET}" + f"{GREEN}={RESET}"*10)


def stitch_echelle_orders(wavelengths, fluxes, tol=1e-3):
    """
    Stitch echelle orders that are individually ascending but
    globally decreasing in wavelength order.
    Keeps flux aligned and removes overlaps.
    """

    # Reverse order sequence so we go from low λ → high λ
    if wavelengths[0][0] > wavelengths[-1][0]:
        wavelengths = wavelengths[::-1]
        fluxes = fluxes[::-1]

    stitched_wave = [wavelengths[0]]
    stitched_flux = [fluxes[0]]

    for w, f in zip(wavelengths[1:], fluxes[1:]):
        prev_max = stitched_wave[-1][-1]
        mask = w > prev_max + tol  # assume within-order increasing
        if np.any(mask):
            stitched_wave.append(w[mask])
            stitched_flux.append(f[mask])

    final_wave = np.concatenate(stitched_wave)
    final_flux = np.concatenate(stitched_flux)

    return final_wave, final_flux


def trim_order_edges(wavelengths, fluxes, ntrim=4):
    """
    Trim ntrim pixels from both ends of each order.
    """
    w_trimmed, f_trimmed = [], []
    for w, f in zip(wavelengths, fluxes):
        if len(w) > 2 * ntrim:
            w_trimmed.append(w[ntrim:-ntrim])
            f_trimmed.append(f[ntrim:-ntrim])
    return w_trimmed, f_trimmed


def hst_spec(file_name):
    print(file_name)
    with fits.open(file_name) as hdul:
        dat = hdul[1].data
        hdr = hdul[0].header

    star_name = clean_star_name_2(hdr["TARGNAME"])
    obs_jd = Time("T".join((hdr['TDATEOBS'], hdr['TTIMEOBS'])), format='isot', scale='utc').jd
    obs_date = hdr['TDATEOBS']

    wavs = dat["WAVELENGTH"]
    fluxes = dat["FLUX"]

    waves_trimmed, fluxes_trimmed = trim_order_edges(wavs, fluxes, ntrim=2)
    final_wavs, final_fluxes = stitch_echelle_orders(waves_trimmed, fluxes_trimmed, tol=1e-3)
    # breakpoint()

    # rec_list = [6410, 6425, 6500, 6540, 6600, 6615, 6650, 6695]
    # rec_points = []
    # for item in rec_list:
    #     rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))
    #
    # ind = np.where((dat["WAVE"][0] > 6400) & (dat["WAVE"][0] < 6700))[0]
    #
    # # print(file_name)
    # # print(ind)
    # # print(dat["WAVE"][0][ind])
    #
    # x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
    # cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
    # y = cs(x)

    fig, ax = plt.subplots(figsize=(20, 10))
    for i in range(len(waves_trimmed)):
        ax.plot(waves_trimmed[i], fluxes_trimmed[i], c="k")
    ax.plot(final_wavs, final_fluxes, c="r", alpha=0.5)
    # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none', edgecolor="xkcd:goldenrod",
    #            linewidth=5, zorder=10, s=300)
    # ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"HST_Archival/Plots/RawSpec/STIS_{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # print("Plotted raw spectrum!")
    #
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind] / y, c="k")
    # # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
    # # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
    #                width=1)
    # ax.set_ylabel("Normalized Flux", fontsize=22)
    # ax.set_xlabel("Wavelength [Å]", fontsize=22)
    # ax.yaxis.get_offset_text().set_size(20)
    # fig.savefig(f"HARPS/Plots/NormSpec/HARPS_{star_name}_{obs_date}_normalized_spec.pdf", bbox_inches="tight", dpi=300)
    # plt.close()
    #
    # print("Plotted normalized spectrum!")


def besos_spec(file_name):
    print(file_name)

    with fits.open(file_name) as hdul:
        dat = hdul[0].data
        hdr = hdul[0].header

    # XShooter spectra have wavelengths in nm instead of Å
    if hdr["INSTRUME"] == "XSHOOTER":
        dat["WAVE"][0] *= 10


    star_name = clean_star_name_2(hdr["NAME"])
    obs_jd = Time(hdr['DATE-OBS'], format='isot', scale='utc').jd
    obs_date = hdr['DATE-OBS'].split("T")[0]

    bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, lat=-33.2691666667, longi=-70.5344444444, alt=1450, ephemeris="de430")[0][0]
    print(bc_corr)

    wavs = np.linspace(hdr["CRVAL1"],
                       hdr["CRVAL1"] + hdr["CDELT1"] * hdr["NAXIS1"],
                       hdr["NAXIS1"])

    fluxes = dat

    ind = np.where((wavs > 6400) & (wavs < 6700))[0]

    wavs = wavs[ind]
    fluxes = fluxes[ind]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(wavs, fluxes, c="k")
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"BESOS/Plots/NormSpec/BESOS_{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted normalized spectrum!")

    v_bis, v_grid, ccf = shafter_bisector_velocity(wavs, fluxes, print_flag=False)

    plot_ind = np.where((((np.array(wavs) - 6562.8) / 6562.8) * 3e5 > -500) &
                                (((np.array(wavs) - 6562.8) / 6562.8) * 3e5 < 500) &
                                ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]
    ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
    # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]

    sig_ind = np.where((wavs > 6600) & (wavs < 6625))[0]
    sig_cont = np.std(fluxes[sig_ind] - 1)

    err_v_bis = 0.5
    rad_vel_bc_corrected = v_bis + bc_corr/1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

    fig, ax = plt.subplots(2,1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
    plt.subplots_adjust(hspace=0)
    fig.supxlabel("Radial Velocity [km s$^{-1}$]", fontsize=22, y=0.05)
    ax[0].plot(((np.array(wavs) - 6562.8) / 6562.8) * 3e5, np.array(fluxes) - 1,
            color="black", label="CCF")
    # ax[0].plot(v_grid, ker, c="r")
    ax[0].vlines(v_bis, 0.23 * max(fluxes - 1), 0.27 * max(fluxes - 1), color="r", zorder=1)
    ax[0].hlines(0.25 * max(fluxes - 1), (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[0]],
              (((wavs - 6562.8) / 6562.8) * 3e5)[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].set_ylabel("Normalized Flux", fontsize=22)
    ax[0].set_xlim(-500, 500)
    ax[0].text(0.78, 0.8, fr"{star_name} H$\alpha$"
                      f"\nHJD {obs_jd:.4f}\nRV = {v_bis:.3f}±{err_v_bis:.3f} km/s",
            color="k", fontsize=18, transform=ax[0].transAxes,
            bbox=dict(
                facecolor='white',  # Box background color
                edgecolor='black',  # Box border color
                boxstyle='square,pad=0.3',  # Rounded box with padding
                alpha=0.9  # Slight transparency
            )
            )
    ax[1].plot(v_grid[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=2)
    ax[1].set_ylabel("CCF", fontsize=22)
    ax[1].hlines(0, min(v_grid[ccf_ind]), max(v_grid[ccf_ind]), color="k", linestyle="--", zorder=0)
    ax[1].set_ylim(-1, 1)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    # ax.set_title("Cross Correlation Function w/ Gaussian Fit", fontsize=26)
    # ax.legend(loc="upper right", fontsize=22)
    # ax.set_ylim(-0.1, max(fluxes - 1)+0.2)
    fig.savefig(f"BESOS/Plots/RV_HAlpha_Bisector/BESOS_RV_{star_name}_{obs_date}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if not os.path.exists("BESOS/BESOS_RV_Bisector.txt"):
        with open("BESOS/BESOS_RV_Bisector.txt", "w") as file:
            file.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open("BESOS/BESOS_RV_Bisector.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open("BESOS/BESOS_RV_Bisector.txt", "a") as f:
                f.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"BESOS/RV_Measurements/{star_name}_RV.txt"):
        with open(f"BESOS/RV_Measurements/{star_name}_RV.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"BESOS/RV_Measurements/{star_name}_RV.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"BESOS/RV_Measurements/{star_name}_RV.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    print("Plotted bisector!")

    print("="*10 + "Done!" + "="*10)


def espresso_spec(file_name):
    print(f"{BLUE}{file_name}{RESET}")
    with fits.open(file_name) as hdul:
        dat = hdul[1].data
        hdr = hdul[0].header
        # breakpoint()

    # XShooter spectra have wavelengths in nm instead of Å
    if hdr["INSTRUME"] == "XSHOOTER":
        dat["WAVE"][0] *= 10

    star_name = clean_star_name_2(hdr["OBJECT"])
    print(f"{YELLOW}{star_name}{RESET}")
    obs_jd = Time(hdr['DATE-OBS'], format='isot', scale='utc').jd
    obs_date = hdr['DATE-OBS'].split("T")[0]

    bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, obsname="Cerro Paranal", ephemeris="de430")[0][0]
    print(bc_corr)

    rec_list = [6410, 6425, 6500, 6540, 6600, 6615, 6650, 6695]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 6400) & (dat["WAVE"][0] < 6700))[0]

    # print(file_name)
    # print(ind)
    # print(dat["WAVE"][0][ind])

    x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
    cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
    y = cs(x)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind], c="k")
    ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none', edgecolor="xkcd:goldenrod",
               linewidth=5, zorder=10, s=300)
    ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"ESPRESSO/Plots/RawSpec/ESPRESSO_{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted raw spectrum!")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind]/y, c="k")
    # ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
    # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax.set_ylabel("Normalized Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"ESPRESSO/Plots/NormSpec/ESPRESSO_{star_name}_{obs_date}_normalized_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted normalized spectrum!")

    wavs = dat["WAVE"][0][ind]
    fluxes = dat["FLUX"][0][ind]/y

    v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs, fluxes, print_flag=False)

    plot_ind = np.where((((np.array(wavs) - 6562.8) / 6562.8) * 3e5 > -500) &
                                (((np.array(wavs) - 6562.8) / 6562.8) * 3e5 < 500) &
                                ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]
    ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
    # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]

    sig_ind = np.where((wavs > 6600) & (wavs < 6625))[0]
    sig_cont = np.std(fluxes[sig_ind] - 1)

    res = analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                            gaussian_width_kms=gaussian_width,
                                            p=0.25,
                                            Ntop=5,
                                            M_inject=400,
                                            MC_samples=10_000)

    # err_v_bis = (sig_cont * gaussian_width) / (max(np.array(fluxes) - 1) * 0.25 * np.sqrt(-2 * np.log(0.25)))
    err_v_bis = float(res["sigma_v_median"])

    # err_v_bis = 0.5
    rad_vel_bc_corrected = v_bis + bc_corr/1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

    dlamb, coeff = wav_corr(np.array(wavs), bc_corr, v_bis)

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
    ax[0].vlines(6562.8 + 6562.8 * v_bis / 3e5, 0.15 * max(fluxes - 1), 0.35 * max(fluxes - 1), color="r", zorder=1,
                 lw=3)
    ax[0].hlines(0.25 * max(fluxes - 1), dlamb[plot_ind[0]], dlamb[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=False, right=True, length=10,
                      width=1)
    # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].set_ylabel("Normalized Flux", fontsize=22)
    ax[0].set_xlim(6562.8 - 6562.8 * 500 / 3e5, 6562.8 + 6562.8 * 500 / 3e5)
    ax[0].text(0.77, 0.8, fr"{clean_star_name3(star_name)} H$\alpha$"
                          f"\nHJD {obs_jd:.4f}\n"
                          fr"RV$_{{\text{{BC}}}}$ = {rad_vel_bc_corrected:.3f}±{err_v_bis:.3f} km/s",
               color="k", fontsize=18, transform=ax[0].transAxes,
               bbox=dict(
                   facecolor='white',  # Box background color
                   edgecolor='black',  # Box border color
                   boxstyle='square,pad=0.3',  # Rounded box with padding
                   alpha=0.9  # Slight transparency
               )
               )

    secax_x = ax[0].secondary_xaxis("top", functions=(wav_to_vel_halpha, vel_to_wav_halpha))
    secax_x.set_xlabel(r"Radial Velocity [km/s]", fontsize=22)
    secax_x.tick_params(labelsize=22, which='both')
    secax_x.tick_params(axis='both', which='both', direction='in', length=10, width=1)

    ax[1].plot((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=2)
    ax[1].set_ylabel("CCF", fontsize=22)
    ax[1].hlines(0, min((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind]), max((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind]),
                 color="k", linestyle="--", zorder=0)
    ax[1].set_ylim(-1, 1)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    fig.savefig(f"ESPRESSO/Plots/RV_HAlpha_Bisector/ESPRESSO_RV_{star_name}_{obs_date}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    rad_vel_bc_corrected -= 100.717

    if not os.path.exists("ESPRESSO/ESPRESSO_RV_Bisector.txt"):
        with open("ESPRESSO/ESPRESSO_RV_Bisector.txt", "w") as file:
            file.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open("ESPRESSO/ESPRESSO_RV_Bisector.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open("ESPRESSO/ESPRESSO_RV_Bisector.txt", "a") as f:
                f.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"ESPRESSO/RV_Measurements/{star_name}_RV.txt"):
        with open(f"ESPRESSO/RV_Measurements/{star_name}_RV.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"ESPRESSO/RV_Measurements/{star_name}_RV.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"ESPRESSO/RV_Measurements/{star_name}_RV.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    print("Plotted bisector!")

    # Na I Doublet for offset calculation
    # breakpoint()
    rec_list = [5886.5, 5888, 5888.4, 5889.4, 5894, 5894.735, 5895.5, 5898.7, 5899]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(dat["WAVE"][0] - item)))

    ind = np.where((dat["WAVE"][0] > 5886) & (dat["WAVE"][0] < 5900))[0]

    if len(ind) > 0:
        print(f"{GREEN}Running Na I Doublet Analysis{RESET}")

        x = np.linspace(min(dat["WAVE"][0][ind]), max(dat["WAVE"][0][ind]), len(dat["WAVE"][0][ind]))
        cs = CubicSpline(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points])
        y = cs(x)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind], c="k")
        ax.scatter(dat["WAVE"][0][rec_points], dat["FLUX"][0][rec_points], facecolor='none',
                   edgecolor="xkcd:goldenrod",
                   linewidth=5, zorder=10, s=300)
        ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)
        fig.savefig(f"ESPRESSO/Plots/RawSpec/ESPRESSO_{star_name}_{obs_date}_raw_spec_NaIDoublet.pdf",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

        print("Plotted raw spectrum!")

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(dat["WAVE"][0][ind], dat["FLUX"][0][ind] / y, c="k")
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        ax.set_ylabel("Normalized Flux", fontsize=22)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.yaxis.get_offset_text().set_size(20)

        fig.savefig(f"ESPRESSO/Plots/NormSpec/ESPRESSO_{star_name}_{obs_date}_normalized_spec_NaIDoublet.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        wavs = dat["WAVE"][0][ind]
        fluxes = dat["FLUX"][0][ind] / y

        ism_ind1 = np.where((wavs > 5889.95 - 1) & (wavs < 5889.95 + 3))[0]
        ism_ind2 = np.where((wavs > 5895.92 - 1) & (wavs < 5895.92 + 3))[0]
        ism_ind = np.concat([ism_ind1, ism_ind2])

        inv_flux = -(fluxes - 1)
        # peaks, props = find_peaks(inv_flux, height=0.2)

        peaks = [np.argmin(fluxes[ism_ind1]), np.argmin(fluxes[ism_ind2])]

        # breakpoint()

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
            # breakpoint()

            # print(f"Doublet Vel: {doublet_vel}")

            err_v_doublet = 0.5  # Arbitratry error amount, need to check this
            rad_vel_doublet_corrected = doublet_vel + bc_corr / 1000  # self.bc_corr has a sign, so need to add (otherwise might add when negative)

            dlamb, coeff = wav_corr(np.array(wavs), bc_corr, doublet_vel)

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
            # ax[0].axvspan(wavs[ism_ind1[0]], wavs[ism_ind1[-1]], color="xkcd:periwinkle")
            # ax[0].plot(v_grid, ker, c="r")
            # ax[0].vlines(5889.95 + 5889.95*vel_d1/3e5 - 5889.95*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True,
                              length=10,
                              width=1)
            ax[0].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[0].set_ylabel()
            ax[0].set_xlim(5889.95 - 5889.95 * 220 / 3e5, 5889.95 + 5889.95 * 220 / 3e5)
            ax[0].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
            ax[0].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_1$"
                                   f"\nHJD {obs_jd:.4f}\nRV = {vel_d1 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[0].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )

            ax[1].plot(dlamb, np.array(fluxes) - 1, color="black", label="CCF")
            ax[1].scatter(dlamb[ism_ind2][peaks[1]], (np.array(fluxes) - 1)[ism_ind2][peaks[1]], c="r")
            ax[1].vlines(wavs[ism_ind2[0]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            ax[1].vlines(wavs[ism_ind2[-1]], min(np.array(fluxes) - 1) - 0.2, max(np.array(fluxes) - 1) + 0.2,
                         color="xkcd:periwinkle")
            # ax[1].axvspan(wavs[ism_ind2[0]], wavs[ism_ind2[-1]], color="xkcd:periwinkle")
            # ax[1].vlines(5895.92 + 5895.92*vel_d2/3e5 - 5895.92*rad_vel_doublet_corrected/3e5, 0.2 * max(fluxes) - 0.2 , 0.25 * max(fluxes) - 0.2, color="r", zorder=1, lw=3)
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True,
                              length=10,
                              width=1)
            ax[1].set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
            # ax[1].set_ylabel()
            ax[1].set_xlim(5895.92 - 5895.92 * 220 / 3e5, 5895.92 + 5895.92 * 220 / 3e5)
            ax[1].set_ylim(min(np.array(fluxes) - 1) - 0.02, max(np.array(fluxes) - 1) + 0.02)
            ax[1].text(0.55, 0.05, fr"{clean_star_name3(star_name)} Na I D$_2$"
                                   f"\nHJD {obs_jd:.4f}\nRV = {vel_d2 + bc_corr / 1000:.3f}±{err_v_doublet:.3f} km/s",
                       color="k", fontsize=18, transform=ax[1].transAxes,
                       bbox=dict(
                           facecolor='white',  # Box background color
                           edgecolor='black',  # Box border color
                           boxstyle='square,pad=0.3',  # Rounded box with padding
                           alpha=0.9  # Slight transparency
                       )
                       )
            fig.savefig(f"ESPRESSO/Plots/RV_Na_I_Doublet/RV_{star_name}_{obs_date}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.close()

            if not os.path.exists("ESPRESSO/ESPRESSOInventoryRV_Na_I_Doublet.txt"):
                with open("ESPRESSO/ESPRESSOInventoryRV_Na_I_Doublet.txt", "w") as file:
                    file.write(
                        f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open("ESPRESSO/ESPRESSOInventoryRV_Na_I_Doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(obs_jd) in line for line in jds):
                    with open("ESPRESSO/ESPRESSOInventoryRV_Na_I_Doublet.txt", "a") as f:
                        f.write(
                            f"{star_name},{obs_jd},{obs_date},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

            if not os.path.exists(f"ESPRESSO/RV_Measurements/{star_name}_RV_doublet.txt"):
                with open(f"ESPRESSO/RV_Measurements/{star_name}_RV_doublet.txt", "w") as file:
                    file.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")
            else:
                with open(f"ESPRESSO/RV_Measurements/{star_name}_RV_doublet.txt", "r") as f:
                    jds = f.read().splitlines()

                if not any(str(obs_jd) in line for line in jds):
                    with open(f"ESPRESSO/RV_Measurements/{star_name}_RV_doublet.txt", "a") as f:
                        f.write(f"{obs_jd},{rad_vel_doublet_corrected:.3f},{err_v_doublet:.5f}\n")

    print(rad_vel_doublet_corrected)
    print(f"{GREEN}={RESET}"*10 + f"{GREEN}Done!{RESET}" + f"{GREEN}={RESET}"*10)


def omi_pup_koubsky(file_name):
    with fits.open(f"{file_name}") as hdul:
        hdr = hdul[0].header
        wavs = np.linspace(hdul[0].header['CRVAL1'],
                           hdul[0].header['CRVAL1'] + hdul[0].header['CDELT1'] * hdul[0].header['NAXIS1'],
                           hdul[0].header['NAXIS1'])
        fluxes = hdul[0].data

    star_name = "omi Pup"
    print(f"{YELLOW}{star_name}{RESET}")
    obs_jd = Time(hdr['DATE-OBS'], format='isot', scale='utc').jd
    obs_date = hdr['DATE-OBS'].split("T")[0]

    bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, lat=49.91019913005359, longi=14.780646225824176, alt=528,
                         ephemeris="de430")[0][0]
    print(bc_corr)

    rec_list = [6410, 6425, 6500, 6540, 6600, 6615, 6650, 6695]
    rec_points = []
    for item in rec_list:
        rec_points.append(np.argmin(abs(wavs - item)))

    ind = np.where((wavs > 6400) & (wavs < 6700))[0]

    # print(file_name)
    # print(ind)
    # print(dat["WAVE"][0][ind])

    x = np.linspace(min(wavs[ind]), max(wavs[ind]), len(wavs[ind]))
    cs = CubicSpline(wavs[rec_points], fluxes[rec_points])
    y = cs(x)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(wavs[ind], fluxes[ind], c="k")
    ax.scatter(wavs[rec_points], fluxes[rec_points], facecolor='none',
               edgecolor="xkcd:goldenrod",
               linewidth=5, zorder=10, s=300)
    ax.plot(x, y, c="r", alpha=0.75, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    ax.set_ylabel("Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"omiPup_Koubsky/Plots/RawSpec/{star_name}_{obs_date}_raw_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("Plotted raw spectrum!")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(wavs[ind], fluxes[ind] / y, c="k")
    # ax.scatter(wavs[rec_points], dat["FLUX"][0][rec_points], c="xkcd:goldenrod", zorder=10, s=300)
    # ax.plot(x, y, c="r", alpha=0.5, linewidth=3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    ax.set_ylabel("Normalized Flux", fontsize=22)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"omiPup_Koubsky/Plots/NormSpec/{star_name}_{obs_date}_normalized_spec.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    wavs = pd.Series(wavs[ind])
    fluxes = pd.Series((fluxes[ind]) / y)
    df = pd.concat([wavs, fluxes], axis="columns")
    df.columns = ["Wavelength", "Flux"]
    df.to_csv(f"omiPup_Koubsky/SpectrumData/{star_name}_{obs_date}_{obs_jd:.3f}.csv",
              index=False)

    print("Plotted normalized spectrum!")
    wavs = np.array(wavs)
    fluxes = np.array(fluxes)

    v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs, fluxes, print_flag=False)

    sig_ind = np.where((wavs > 6600) & (wavs < 6625))[0]

    res = analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                            gaussian_width_kms=gaussian_width,
                                            p=0.25,
                                            Ntop=5,
                                            M_inject=400,
                                            MC_samples=10_000)

    err_v_bis = float(res["sigma_v_median"])

    rad_vel_bc_corrected = v_bis + bc_corr / 1000  # bc_corr has a sign, so need to add (otherwise might add when negative)

    dlamb, coeff = wav_corr(np.array(wavs), bc_corr, v_bis)

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
    ax[0].vlines(6562.8 + 6562.8 * v_bis / 3e5, 0.15 * max(fluxes - 1), 0.35 * max(fluxes - 1), color="r", zorder=1,
                 lw=3)
    ax[0].hlines(0.25 * max(fluxes - 1), dlamb[plot_ind[0]], dlamb[plot_ind[-1]], color="k", alpha=0.8, zorder=0)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=False, right=True, length=10,
                      width=1)
    # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].set_ylabel("Normalized Flux", fontsize=22)
    ax[0].set_xlim(6562.8 - 6562.8 * 500 / 3e5, 6562.8 + 6562.8 * 500 / 3e5)
    ax[0].text(0.73, 0.8, fr"{clean_star_name3(star_name)} H$\alpha$"
                          f"\nHJD {obs_jd:.4f}\n"
                          fr"RV$_{{\text{{BC}}}}$ = {rad_vel_bc_corrected:.3f}±{err_v_bis:.3f} km/s",
               color="k", fontsize=18, transform=ax[0].transAxes,
               bbox=dict(
                   facecolor='white',  # Box background color
                   edgecolor='black',  # Box border color
                   boxstyle='square,pad=0.3',  # Rounded box with padding
                   alpha=0.9  # Slight transparency
               )
               )

    secax_x = ax[0].secondary_xaxis("top", functions=(wav_to_vel_halpha, vel_to_wav_halpha))
    secax_x.set_xlabel(r"Radial Velocity [km/s]", fontsize=22)
    secax_x.tick_params(labelsize=22, which='both')
    secax_x.tick_params(axis='both', which='both', direction='in', length=10, width=1)

    ax[1].plot((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle", zorder=10, linewidth=2)
    ax[1].set_ylabel("CCF", fontsize=22)
    ax[1].hlines(0, min((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind]), max((6562.8 + 6562.8 * v_grid / 3e5)[ccf_ind]),
                 color="k", linestyle="--", zorder=0)
    ax[1].set_ylim(-1, 1)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    fig.savefig(f"omiPup_Koubsky/Plots/RV_HAlpha_Bisector/RV_{star_name}_{obs_date}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if not os.path.exists("omiPup_Koubsky/RV_Bisector.txt"):
        with open("omiPup_Koubsky/RV_Bisector.txt", "w") as file:
            file.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open("omiPup_Koubsky/RV_Bisector.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open("omiPup_Koubsky/RV_Bisector.txt", "a") as f:
                f.write(f"{star_name},{obs_jd},{obs_date},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"omiPup_Koubsky/RV_Measurements/{star_name}_RV.txt"):
        with open(f"omiPup_Koubsky/RV_Measurements/{star_name}_RV.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"omiPup_Koubsky/RV_Measurements/{star_name}_RV.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"omiPup_Koubsky/RV_Measurements/{star_name}_RV.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    print("Plotted bisector!")


uves_files = list_fits_files("UVES/HD088661")
# iacob_files = list_fits_files("IACOB/HD91120")
iacob_files = list_fits_files("IACOB/HD63462")
feros_files = list_fits_files("FEROS/HD063462")
harps_files = list_fits_files("HARPS")
hst_files = list_fits_files("HST_Archival")
besos_files = list_fits_files("BESOS/HD217891")
espresso_files = list_fits_files("ESPRESSO/HD063462")
koubsky_files = glob.glob("omiPup_Koubsky/*.fit")


for file in koubsky_files:
    omi_pup_koubsky(file)