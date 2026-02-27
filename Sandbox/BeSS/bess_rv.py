import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from chironHelperFunctions import *
from barycorrpy import get_BC_vel, JDUTC_to_BJDTDB

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

def bess_rv(star_name, file_name):
    with fits.open(f"{file_name}") as hdul:
        hdr = hdul[0].header
        wavs = np.linspace(hdul[0].header['CRVAL1'],
                           hdul[0].header['CRVAL1'] + hdul[0].header['CDELT1'] * hdul[0].header['NAXIS1'],
                           hdul[0].header['NAXIS1'])
        fluxes = hdul[0].data
        if hdr["BSS_RQVH"] != 0:
            bc_corr = hdr['BSS_RQVH']
        else:
            bc_corr = 0
        wavs_corr = barycentric_correct(wavs, bc_corr)
    # breakpoint()

    if hdr["BSS_ITRP"] >= 10000:
        print(f"Resolving Power: {GREEN}{hdr["BSS_ITRP"]}{RESET}")
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(wavs_corr, fluxes, color="k")
        ax.set_ylabel("Normalized Flux", fontsize=22)
        ax.set_xlabel("Wavelength", fontsize=22)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                       width=1)
        fig.savefig("bess_spec.pdf", bbox_inches='tight', dpi=300)

        v_bis, v_grid, ccf, gaussian_width = shafter_bisector_velocity(wavs_corr, fluxes, v_step=5, sep=600)

        res = analytic_sigma_v_mc_from_nonparam(wavs_corr, fluxes,
                                                gaussian_width_kms=gaussian_width,
                                                p=0.25,
                                                Ntop=5,
                                                M_inject=400,
                                                MC_samples=10_000)

        # err_v_bis = (sig_cont * gaussian_width) / (max(np.array(fluxes) - 1) * 0.25 * np.sqrt(-2 * np.log(0.25)))
        # err_v_bis = float(res["sigma_v_median"])
        dx = 300  # Half the separation of the double Gaussians from above
        err_v_bis = (np.sqrt(7) / np.log(4)) * (1 / der_snr(fluxes)) * np.sqrt(
            ((1 + 0.25 * max(np.array(fluxes - 1))) * dx * 5))

        # if hdr['BSS_VHEL'] == 0:
        #     rad_vel_bc_corrected = v_bis + hdr['BSS_RQVH']
        # else:
        rad_vel_bc_corrected = v_bis

        dlamb = stellar_rest_frame(np.array(wavs), v_bis)
        # try:
        #     bc_corr = get_BC_vel(JDUTC=obs_jd, starname=star_name, obsname="CTIO", ephemeris="de430")[0][0]
        # except:
        #     self.bc_corr = 0
        #     print(f"WARNING: BC Not Found for {self.star_name}!")
        #
        # print(f"Radial Velocity: \033[92m{rad_vel_bc_corrected:.3f} km/s\033[0m")

        plot_ind = np.where((((np.array(wavs_corr) - 6562.8) / 6562.8) * 3e5 > -500) &
                            (((np.array(wavs_corr) - 6562.8) / 6562.8) * 3e5 < 500) &
                            ((np.array(fluxes) - 1) > 0.25 * max(np.array(fluxes) - 1)))[0]
        ccf_ind = np.where((v_grid > -500) & (v_grid < 500))[0]
        # plot_ind = np.where((np.array(fluxes[plot_ind1]) - 1) > 0.25 * max(np.array(fluxes[plot_ind1]) - 1))[0]
        try:
            obs_jd = hdr['JD-MID']
        except:
            obs_jd = hdr['MID-HJD']

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
        plt.subplots_adjust(hspace=0)
        fig.supxlabel("Radial Velocity [km s$^{-1}$]", fontsize=22, y=0.05)
        ax[0].plot(dlamb, np.array(fluxes),color="black", label="CCF")
        # ax[0].vlines(6562.8 + 6562.8 * rad_vel_bc_corrected / 3e5, 1 + 0.15 * max(fluxes - 1),
        #              1 + 0.35 * max(fluxes - 1), color="r", zorder=1, lw=3) # Barycentric rest frame
        ax[0].vlines(6562.8, 1 + 0.15 * max(fluxes - 1),
                     1 + 0.35 * max(fluxes - 1), color="r", zorder=1, lw=3) # Stellar rest frame
        ax[0].hlines(1 + 0.25 * max(fluxes - 1), dlamb[plot_ind[0]], dlamb[plot_ind[-1]], color="k", alpha=0.8,
                     zorder=0)
        ax[0].tick_params(axis='x', labelsize=20)
        ax[0].tick_params(axis='y', labelsize=20)
        ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=False, right=True, length=10,
                          width=1)
        # ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
        ax[0].set_ylabel("Normalized Flux", fontsize=22)
        ax[0].set_xlim(6562.8 - 6562.8 * 500 / 3e5, 6562.8 + 6562.8 * 500 / 3e5)
        ax[0].text(0.75, 0.8, fr"{clean_star_name3(star_name)} H$\alpha$"
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
        wav_to_vel, vel_to_wav = make_vel_wav_transforms(6562.8)  # H alpha
        secax_x = ax[0].secondary_xaxis("top", functions=(wav_to_vel, vel_to_wav))
        secax_x.set_xlabel(r"Radial Velocity [km/s]", fontsize=22)
        secax_x.tick_params(labelsize=22, which='both')
        secax_x.tick_params(axis='both', which='both', direction='in', length=10, width=1)

        # ax[1].plot((6562.8 + 6562.8 * (v_grid + hdr['BSS_RQVH'] / 1000) / 3e5)[ccf_ind], ccf[ccf_ind], c="xkcd:periwinkle",
        #            zorder=10, linewidth=3)
        ax[1].plot((6562.8 + 6562.8 * (v_grid - v_bis) / 3e5)[ccf_ind], ccf[ccf_ind],
                   c="xkcd:periwinkle",
                   zorder=10, linewidth=3)
        ax[1].set_ylabel("CCF", fontsize=22)
        ax[1].hlines(0, 6562.8 - 6562.8 * 500 / 3e5, 6562.8 + 6562.8 * 500 / 3e5, color="gray", linestyle="--",
                     zorder=0)
        ax[1].set_ylim(-2, 2)
        ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                          width=1)
        fig.savefig(f"BeSS_RVs/{star_name}/bess_spec_rv_bisector_{hdr['OBJNAME'].strip()}_{hdr['DATE-OBS'].split('T')[0]}.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()

        if not os.path.exists(f"BeSS_RVs/{star_name}/RV_Bisector.txt"):
            with open(f"BeSS_RVs/{star_name}/RV_Bisector.txt", "w") as file:
                file.write(f"{star_name},{obs_jd},{hdr['DATE-OBS'].split('T')[0]},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
        else:
            with open(f"BeSS_RVs/{star_name}/RV_Bisector.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(obs_jd) in line for line in jds):
                with open(f"BeSS_RVs/{star_name}/RV_Bisector.txt", "a") as f:
                    f.write(f"{star_name},{obs_jd},{hdr['DATE-OBS'].split('T')[0]},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

        if not os.path.exists(f"BeSS_RVs/{star_name}/{star_name}_RV.txt"):
            with open(f"BeSS_RVs/{star_name}/{star_name}_RV.txt", "w") as file:
                file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
        else:
            with open(f"BeSS_RVs/{star_name}/{star_name}_RV.txt", "r") as f:
                jds = f.read().splitlines()

            if not any(str(obs_jd) in line for line in jds):
                with open(f"BeSS_RVs/{star_name}/{star_name}_RV.txt", "a") as f:
                    f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    else:
        print(f"Resolving Power: {RED}{hdr["BSS_ITRP"]}{RESET}")
        pass


def bess_rv_he(star_name, file_name):
    with fits.open(f"{file_name}") as hdul:
        hdr = hdul[0].header
        wavs = np.linspace(hdul[0].header['CRVAL1'],
                           hdul[0].header['CRVAL1'] + hdul[0].header['CDELT1'] * hdul[0].header['NAXIS1'],
                           hdul[0].header['NAXIS1'])
        fluxes = hdul[0].data

    try:
        obs_jd = hdr['JD-MID']
    except:
        obs_jd = hdr['MID-HJD']

    ind = np.where((wavs > 5012) & (wavs < 5025))

    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(wavs, fluxes, c='k')
    ax.set_title(f'{clean_star_name3(star_name)}' + fr' {hdr['DATE-OBS'].split('T')[0]} He I $\lambda$5016', fontsize=24)
    ax.set_xlabel("Wavelength [Å]", fontsize=22)
    ax.set_ylabel("Normalized Flux", fontsize=22)
    ax.scatter(wavs[ind][np.argmin(fluxes[ind])], fluxes[ind][np.argmin(fluxes[ind])], c="r")
    ax.vlines(5015.6783, min(fluxes), max(fluxes), color='dodgerblue')
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='both', which='major', length=10, width=1)
    ax.set_xlim(5010, 5030)
    ax.yaxis.get_offset_text().set_size(20)

    rv = (wavs[ind][np.argmin(fluxes[ind])] - 5015.6783) / 5015.6783 * 3e5
    print(f"RV = {YELLOW}{rv:.3f}{RESET}")

    if hdr['BSS_VHEL'] == 0:
        rad_vel_bc_corrected = rv - hdr['BSS_RQVH']
    else:
        rad_vel_bc_corrected = rv

    # rv = (wavs[np.argmin(fluxes)] - 5015.6783) / 5015.6783 * 3e5
    print(f"{obs_jd},{rad_vel_bc_corrected:.3f},5")

    err_v_bis = 5

    ax.text(0.76, 0.85, fr"{clean_star_name3(hdr['OBJNAME'].strip())} He I 5016"
                          f"\nHJD {obs_jd:.4f}\nRV = {rv:.3f}±{err_v_bis:.3f} km/s",
               color="k", fontsize=18, transform=ax.transAxes,
               bbox=dict(
                   facecolor='white',  # Box background color
                   edgecolor='black',  # Box border color
                   boxstyle='square,pad=0.3',  # Rounded box with padding
                   alpha=0.9  # Slight transparency
               )
               )
    fig.savefig(f"BeSS_RVs/{star_name}/bess_spec_rv_He5016_{hdr['OBJNAME'].strip()}_{hdr['DATE-OBS'].split('T')[0]}.pdf",
                bbox_inches="tight", dpi=300)

    # plt.show()
    plt.close()


    if not os.path.exists(f"BeSS_RVs/{star_name}/RV_He5016.txt"):
        with open(f"BeSS_RVs/{star_name}/RV_He5016.txt", "w") as file:
            file.write(
                f"{star_name},{obs_jd},{hdr['DATE-OBS'].split('T')[0]},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"BeSS_RVs/{star_name}/RV_He5016.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"BeSS_RVs/{star_name}/RV_He5016.txt", "a") as f:
                f.write(
                    f"{star_name},{obs_jd},{hdr['DATE-OBS'].split('T')[0]},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")

    if not os.path.exists(f"BeSS_RVs/{star_name}/{star_name}_RV_He.txt"):
        with open(f"BeSS_RVs/{star_name}/{star_name}_RV_He.txt", "w") as file:
            file.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")
    else:
        with open(f"BeSS_RVs/{star_name}/{star_name}_RV_He.txt", "r") as f:
            jds = f.read().splitlines()

        if not any(str(obs_jd) in line for line in jds):
            with open(f"BeSS_RVs/{star_name}/{star_name}_RV_He.txt", "a") as f:
                f.write(f"{obs_jd},{rad_vel_bc_corrected:.3f},{err_v_bis:.5f}\n")



list_of_files = list_fits_files("BeSS_HD060855")

for file in list_of_files:
    # print(file)
    try:
        bess_rv('HD060855', file)
    except:
        print(f"Couldn't analyze: {file}")
        pass
    # with open("BeSS_RVs/HD191610/RV_Bisector.txt", "r") as f:
    #     # Read the names
    #     star_names = sorted(f.read().splitlines())
    #
    # with open("BeSS_RVs/HD191610/HD191610_RV.txt", "w") as f:
    #     f.write("\n".join(star_names))

    # read the file (comma-separated)
    # df = pd.read_csv("BeSS_RVs/HD191610/HD191610_RV.txt", header=None, names=["HJD", "RV", "RV_err"])
    #
    # # sort by HJD
    # df_sorted = df.sort_values(by="HJD")
    #
    # # save back to file if you want
    # df_sorted.to_csv("BeSS_RVs/HD191610/HD191610_RV.txt", index=False, header=False)