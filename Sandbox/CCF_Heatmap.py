import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import scipy.signal as sig
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
import glob
from cmcrameri import cm
import concurrent.futures
import time
from astropy.time import Time
from SpectrumAnalyzer import spin_beast

model_path = '../HST_Spectra/Models/HD191610_Models_Be/TLUSTY*_rec.fits'

def hst_ccf(obs_spec, model_file):
    """
    Computes the cross-correlation function (CCF) of an observed HST/STIS spectrum with a provided model spectrum

    :param obs_spec: Observed spectrum, provided as tab-separated text file with first column containing wavelengths,
    and the subsequent columns containing fluxes
    :param model_file: Filename of the model. Should be a fits file of the format 'TLUSTY15000_rec.fits' where the
    number represents the model temperature

    :returns: Interpolated velocity shift increments, interpolated CCF values at corresponding velocity shifts, and
    temperature of the model as read in from model filename
    """
    model_temp = model_file.split("/")[-1].split('_')[0].split('TLUSTY')[-1]
    with fits.open(model_file) as hdul:
        model_spec = np.array(hdul[0].data, dtype=float)
    # breakpoint()
    omit_inds = [[0, 3012],
                 [4592, 5207],
                 [7490, 8147],
                 [11802, 12060],
                 [14431, 15833]]

    for ind in omit_inds:
        x = np.arange(ind[0], ind[1])
        obs_spec[ind[0]:ind[1]] = CubicSpline([ind[0], ind[1]],[obs_spec[ind[0]], obs_spec[ind[1]]])(x)

    obs_spec = (obs_spec - np.mean(obs_spec)) / np.std(obs_spec)
    model_spec = (model_spec - np.mean(model_spec)) / np.std(model_spec)

    ccf = sig.correlate(obs_spec, model_spec)
    lag_pixel = np.arange(len(ccf))-(len(ccf)-1)/2
    lag_loglam = 2.5e-5 * lag_pixel

    ind = np.where((-150 < lag_pixel) & (lag_pixel < 150))[0]
    x = np.linspace(lag_loglam[ind][0],lag_loglam[ind][-1], 10_000)
    cs = CubicSpline(lag_loglam, ccf)

    return x, cs(x), model_temp

    # ccf_raw = sig.correlate(obs_spec, model_spec, mode='full')
    #
    # # compute overlap counts per lag by correlating ones
    # overlap = sig.correlate(np.ones_like(obs_spec), np.ones_like(model_spec), mode='full')
    #
    # # avoid division by zero (shouldn't happen for central lags)
    # overlap = np.where(overlap == 0, 1, overlap)
    #
    # # normalized CCF -> correlation coefficient r(lag) in [-1,1]
    # ccf_norm = ccf_raw / overlap
    #
    # lag_pixel = np.arange(len(ccf_raw)) - (len(ccf_raw) - 1) / 2
    # lag_loglam = 2.5e-5 * lag_pixel
    #
    # # choose your lag window as before
    # ind = np.where((-150 < lag_pixel) & (lag_pixel < 150))[0]
    # x = np.linspace(lag_loglam[ind][0], lag_loglam[ind][-1], 10_000)
    # cs = CubicSpline(lag_loglam, ccf_norm)
    #
    # return x, cs(x), model_temp

def ccf_plot_initial(obs_spectrum, star_name, obs_date):
    model = "../HST_Spectra/Models/Generic_Models_sdO/TLUSTY45000_rec.fits"
    x_val, cs_val, temp = hst_ccf(obs_spectrum, model)

    velocities = x_val * 3e5
    ccf_vals = cs_val - min(cs_val)
    # ccf_vals = cs_val

    peaks, properties = sig.find_peaks(ccf_vals, height=0.5 * np.max(ccf_vals))
    main_peak = peaks[np.argmax(properties["peak_heights"])]

    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(15, 10))
    # ax.plot(lag_loglam[ind], ccf[ind]-min(ccf[ind]), linewidth=3, c="k")
    ax.plot(x_val * 3e5, cs_val - min(cs_val), linewidth=3, label=f"45 kK CCF", c="xkcd:periwinkle")
    ax.plot(velocities[main_peak], ccf_vals[main_peak], 'kx', markersize=10,
            label=f'CCF peak = {velocities[main_peak]:.2f} km/s')
    ax.vlines(velocities[main_peak], -100, ccf_vals[main_peak], color="k", linestyle="--", lw=2, zorder=-10)
        # ax.plot(sorted_x[i] * 3e5, sorted_cs[i] - min(sorted_cs[i]), linewidth=3,
        #         label=f"{sorted_temps[i] / 1000}", c=colors[i])
    # ax.plot(be_sdo_ccf[0] * 3e5, be_sdo_ccf[1]*1.4, linewidth=3, c="k")
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=18, top=True, right=True, length=10,
                   width=1)
    ax.set_xlabel(r"Velocity", fontsize=20)
    ax.set_ylabel("Correlation Value", fontsize=20)
    ax.set_title("Cross Correlation Function", fontsize=24)
    ax.legend(title=f"{obs_date}", fontsize=18, title_fontsize=22)
    ax.set_ylim(-10, 1.1 * np.max(ccf_vals))
    fig.savefig(f"Contours/CCF_{star_name}_{obs_date}_initial.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    return velocities, ccf_vals, obs_date

def ccf_plot_heatmap(obs_spectrum, plot_index, star_name, obs_date, be_flag=False):
    """
    Plots the CCF as a function of velocity shift, the CCF peak height as a function of model temperature, and a CCF
    heatmap to identify best-fit model

    :param obs_spectrum: Observed spectrum, provided as tab-separated text file with first column containing
    wavelengths, and the subsequent columns containing fluxes
    :param plot_index: Plot file index, starting with 1, corresponding to each observation of target
    :param star_name: Name of the observed target star
    :param obs_date: UT date of the observation
    :param be_flag: Flag to check if function is being run with Be star models. If true, the heatmap won't be plotted.
    False by default

    :returns: None
    """
    models = glob.glob(model_path)

    x_arr = []
    cs_arr = []
    temp_arr = []
    for model in models:
        x_val, cs_val, temp = hst_ccf(obs_spectrum, model)
        x_arr.append(x_val)
        cs_arr.append(cs_val)
        temp_arr.append(temp)

    # Sorting all arrays based on temperature in ascending order
    paired = [
        (int(t), wl, fl) for t,wl,fl in zip(temp_arr, x_arr, cs_arr) if t.isdigit()
    ]

    paired.sort(key=lambda x: x[0])

    sorted_temps = [t for t, _, _ in paired]
    sorted_x = [wl for _, wl, _ in paired]
    sorted_cs = [fl for _, _, fl in paired]

    cmap = cm.managua  # or cm.roma, cm.lajolla, etc.
    N = len(sorted_x)  # Number of colors (e.g., for 10 lines)
    colors = [cmap(i / N) for i in range(N)]

    # Plotting CCF plots
    # Read in broadened Be+sdO CCF
    # be_sdo_ccfs = glob.glob(f"../HST_Spectra/Models/{star_name}_Models_Be/Be_sdO_CCF*.fits")

    be_sdo_ccf_vals = []
    final_ccf_vals = []
    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(15,10))
    # ax.plot(lag_loglam[ind], ccf[ind]-min(ccf[ind]), linewidth=3, c="k")
    for i in range(len(sorted_temps)):
        with fits.open(f"../HST_Spectra/Models/{star_name}_Models_Be/Be_sdO_CCF_{sorted_temps[i]}.fits") as hdul:
            be_sdo_ccf = hdul[0].data[1]

        be_sdo_ccf_vals.append(be_sdo_ccf)
        final_ccf = sorted_cs[i]-min(sorted_cs[i])-be_sdo_ccf
        final_ccf_vals.append(final_ccf)
        # breakpoint()
        ax.plot(sorted_x[i]*3e5, final_ccf, linewidth=3,
                label=f"{sorted_temps[i]/1000}", c=colors[i])
        # ax.plot(sorted_x[i] * 3e5, sorted_cs[i] - min(sorted_cs[i]), linewidth=3,
        #         label=f"{sorted_temps[i] / 1000}", c=colors[i])
    # ax.plot(be_sdo_ccf[0] * 3e5, be_sdo_ccf[1]*1.4, linewidth=3, c="k")
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=18, top=True, right=True, length=10,
                   width=1)
    ax.set_xlabel(r"Velocity", fontsize=20)
    ax.set_ylabel("Correlation Value", fontsize=20)
    ax.set_title("Cross Correlation Function", fontsize=24)
    ax.legend(title="Model Temperature [kK]", title_fontsize=16, ncols=3, fontsize=12)
    fig.savefig(f"Contours/CCF_{star_name}_{plot_index}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    cs_peak_max_arr = []
    for idx, cs in enumerate(final_ccf_vals):
        ind = np.where((sorted_x[idx]*3e5 > -150) & (sorted_x[idx]*3e5 < 150))[0]
        # breakpoint()
        cs_peak_max_arr.append(max(cs[ind]))
    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(sorted_temps, cs_peak_max_arr, fc="r", zorder=10, ec="k", lw=2, s=60)
    rel_fit = CubicSpline(sorted_temps, cs_peak_max_arr)
    sorted_temp_arr = np.linspace(min(sorted_temps), max(sorted_temps), 5000)
    ax.plot(sorted_temp_arr, rel_fit(sorted_temp_arr), c="k")
    # ax.set_box_aspect(0.3)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=18, top=True, right=True, length=10,
                   width=1)
    ax.set_xlabel(r"Effective Temperature", fontsize=20)
    ax.set_ylabel("Maximum Correlation Value", fontsize=20)
    ax.text(
        0.83, 0.93,  # (x, y) in axes coordinates (1.0 is right/top)
        fr"T$_{{\text{{eff}}}}$ = {round(sorted_temp_arr[np.argmax(rel_fit(sorted_temp_arr))])} K",  # Text string
        ha='left', va='bottom',  # Horizontal and vertical alignment
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=16,
        fontweight='bold',
        bbox=dict(
            facecolor='white',  # Box background color
            edgecolor='black',  # Box border color
            boxstyle='square,pad=0.3',  # Rounded box with padding
            alpha=0.9  # Slight transparency
        )
    )
    # ax.set_title("Cross Correlation Function", fontsize=24)
    # ax.legend(title="Model Temperature", title_fontsize=16, ncols=3, fontsize=12)
    fig.savefig(f"Contours/CCF_Temp_{star_name}_{plot_index}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if not be_flag:
        # Plotting coarse CCF heatmap
        vel_grid, temp_grid = np.meshgrid(sorted_x[0]*3e5, sorted_temps)
        ccf_grid = np.vstack(sorted_cs)

        # Normalizing CCF values for clarity
        ccf_norm_rows = np.array([
            (row - np.min(row))
            for row in ccf_grid
        ])

        # breakpoint()

        # # Plot on the existing axis 'ax'
        # fig, ax = plt.subplots(figsize=(15, 10))
        # mesh = ax.pcolormesh(vel_grid, temp_grid, ccf_norm_rows,
        #                      shading='auto', cmap=cm.managua, rasterized=True)
        #
        # cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        # cbar.set_label('CCF Value', fontsize=18)
        # cbar.ax.tick_params(labelsize=18, length=8, width=1)
        # ax.tick_params(axis='x', labelsize=18)
        # ax.tick_params(axis='y', labelsize=18)
        # ax.set_xlabel(r"Velocity Shift [km/s]", fontsize=20)
        # ax.set_ylabel("Model Temperature [K]", fontsize=20)
        # ax.set_title("Cross Correlation Function Heatmap", fontsize=24)
        # fig.savefig("Contours/Contour_{star_name}_{plot_index}.pdf", bbox_inches='tight')
        #plt.close()

        # Plotting smoothed CCF heatmap
        points = np.array([(v, t) for t, v_array in zip(sorted_temps, vel_grid) for v in v_array])
        values = ccf_norm_rows.flatten()

        # Create finer grid
        vel_fine = np.linspace(vel_grid.min(), vel_grid.max(), 5000)
        temp_fine = np.linspace(min(sorted_temps), max(sorted_temps), 100)
        vel_fine_grid, temp_fine_grid = np.meshgrid(vel_fine, temp_fine)

        # Interpolate onto fine grid
        ccf_fine = griddata(points, values, (vel_fine_grid, temp_fine_grid), method='cubic')

        plt.rcParams['font.family'] = 'Geneva'
        fig, ax = plt.subplots(figsize=(15, 10))
        mesh = ax.pcolormesh(vel_fine_grid, temp_fine_grid/1000, ccf_fine,
                       shading='auto', cmap=cm.managua, rasterized=True)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label('CCF Value', fontsize=18)
        cbar.ax.tick_params(labelsize=18, length=8, width=1)
        ax.set_xlabel(r"Velocity Shift [km/s]", fontsize=20)
        ax.set_ylabel("Model Temperature [kK]", fontsize=20)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=18, top=True, right=True, length=10,
                           width=1)
        ax.set_title(f"{star_name} Cross Correlation Function Heatmap", fontsize=24)
        ax.text(
            0.05, 0.05,  # (x, y) in axes coordinates (1.0 is right/top)
            f"{obs_date}",  # Text string
            ha='left', va='bottom',  # Horizontal and vertical alignment
            transform=ax.transAxes,  # Use axes coordinates
            fontsize=16,
            fontweight='bold',
            bbox=dict(
                facecolor='white',  # Box background color
                edgecolor='black',  # Box border color
                boxstyle='square,pad=0.3',  # Rounded box with padding
                alpha=0.9  # Slight transparency
            )
        )
        fig.savefig(f"Contours/Contour_Smoothened_{star_name}_{plot_index}.pdf", bbox_inches='tight')
        plt.close()
    return


if __name__ == '__main__':
    spec = pd.read_fwf("../HST_Spectra/HST_STIS_Spectra/HD191610_spectrum_data.txt", skiprows=1, header=None)

    obs_spec_arr = []
    col_arr = []
    for col in spec.columns[1:]:
        obs_spect = np.array(spec[col], dtype=float)
        obs_spec_arr.append(obs_spect)
        col_arr.append(col)

    star_name_arr = ['HD191610_Be'] * len(obs_spec_arr)
    be_flag_arr = [False] * len(obs_spec_arr)  # Comment out when running sdO models
    obs_jd_arr = [2458718.1438, 2458726.9457, 2458753.7662]
    obs_date_arr = []
    for obs_time in obs_jd_arr:
        obs_date_arr.append(Time(obs_time, format='jd').fits.split('T')[0])
    print(f"{spin_beast}")
    t1 = time.perf_counter()

    ccf_val_arr = []
    velo_arr = []
    dates_arr = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(ccf_plot_heatmap, obs_spec_arr, col_arr, star_name_arr, obs_date_arr, be_flag_arr)
            results = executor.map(ccf_plot_initial, obs_spec_arr, star_name_arr, obs_date_arr)
    # ccf_plot_heatmap(obs_spec_arr[0], col_arr[0], star_name_arr[0], obs_date_arr[0], be_flag_arr[0])

    # for val1, val2, val3 in results:
    #     velo_arr.append(val1)
    #     ccf_val_arr.append(val2)
    #     dates_arr.append(val3)
    #
    # cmap = cm.roma  # or cm.roma, cm.lajolla, etc.
    # N = len(ccf_val_arr)  # Number of colors (e.g., for 10 lines)
    # colors = [cmap(i / N) for i in range(N)]
    #
    # plt.rcParams['font.family'] = 'Geneva'
    # fig, ax = plt.subplots(figsize=(15, 10))
    # for i in range(len(ccf_val_arr)):
    #     ax.plot(velo_arr[i], ccf_val_arr[i], linewidth=2, label=f"{dates_arr[i]}", c=colors[i])
    # # ax.plot(sorted_x[i] * 3e5, sorted_cs[i] - min(sorted_cs[i]), linewidth=3,
    # #         label=f"{sorted_temps[i] / 1000}", c=colors[i])
    # # ax.plot(be_sdo_ccf[0] * 3e5, be_sdo_ccf[1]*1.4, linewidth=3, c="k")
    # ax.tick_params(axis='x', labelsize=18)
    # ax.tick_params(axis='y', labelsize=18)
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=18, top=True, right=True, length=10,
    #                width=1)
    # ax.set_xlabel(r"Velocity", fontsize=20)
    # ax.set_ylabel("Correlation Value", fontsize=20)
    # ax.set_title("Cross Correlation Function", fontsize=24)
    # ax.legend(title=f"Observation Date", fontsize=18, title_fontsize=22)
    # # ax.set_ylim(-10, 1.1 * np.max(ccf_vals))
    # fig.savefig(f"Contours/CCF_{star_name_arr[0]}_all_initial.pdf", bbox_inches="tight", dpi=300)
    # plt.close()
    # # ccf_plot_heatmap(obs_spec_arr[0], col_arr[0], star_name_arr[0], obs_date_arr[0], be_flag=False)
    # # model = "../HST_Spectra/Models/Generic_Models_sdO/TLUSTY45000_rec.fits"
    # # x_val, cs_val, temp = hst_ccf(obs_spec_arr[0], model)
    # # breakpoint()
    # t2 = time.perf_counter()
    # print(f'Finished in \033[94m{round(t2 - t1, 2)}\033[0m second(s)')

########################################################################################################################

    # obs_spectra = pd.read_fwf("../HST_Spectra/HST_STIS_Spectra/HD041335_spectrum_data.txt", skiprows=1, header=None)
    # model = '../HST_Spectra/Models/Generic_Models_Be/TLUSTY23000_rec.fits'
    #
    # x_val_arr = []
    # cs_val_arr = []
    # temp_arr = []
    # peak_ind_arr = []
    # for col in obs_spectra.columns[1:]:
    #     x_val, cs_val, temp = hst_ccf(obs_spectra[col], model)
    #     peak_ind, _ = sig.find_peaks(cs_val)
    #     x_val_arr.append(x_val)
    #     cs_val_arr.append(cs_val)
    #     temp_arr.append(temp)
    #     peak_ind_arr.append(peak_ind)
    #
    # cmap = cm.managua  # or cm.roma, cm.lajolla, etc.
    # N = len(x_val_arr)  # Number of colors (e.g., for 10 lines)
    # colors = ['xkcd:grass green', 'r', 'dodgerblue']
    # offset = np.arange(len(x_val_arr))
    #
    # plt.rcParams['font.family'] = 'Geneva'
    # fig, ax = plt.subplots(figsize=(15, 10))
    # for i in range(len(x_val_arr)):
    #     ax.plot(x_val_arr[i]*3e5, cs_val_arr[i]-min(cs_val_arr[i]), c=colors[i])
    #     ax.vlines(x_val_arr[i][peak_ind_arr[i]] * 3e5, 0.25*max(cs_val_arr[-1] - min(cs_val_arr[-1]))+offset[i], 0.35*max(cs_val_arr[-1] - min(cs_val_arr[-1]))+offset[i], color=colors[i])
    #     ax.text(
    #         0.8, 0.93-(offset[i]/10),  # (x, y) in axes coordinates (1.0 is right/top)
    #         f"RV = {x_val_arr[i][peak_ind_arr[i]][0] * 3e5:.3f} km/s",  # Text string
    #         ha='left', va='bottom',  # Horizontal and vertical alignment
    #         transform=ax.transAxes,  # Use axes coordinates
    #         fontsize=16,
    #         fontweight='bold',
    #         bbox=dict(
    #             facecolor='white',  # Box background color
    #             edgecolor=colors[i],  # Box border color
    #             boxstyle='square,pad=0.3',  # Rounded box with padding
    #             alpha=0.9  # Slight transparency
    #         )
    #     )
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=18, top=True, right=True, length=10,
    #                width=1)
    # ax.set_xlabel(r"Velocity Shift", fontsize=20)
    # ax.set_ylabel("CCF", fontsize=20)
    # # ax.set_title("Cross Correlation Function", fontsize=24)
    # # ax.legend(title="Model Temperature", title_fontsize=16, ncols=3, fontsize=12)
    # fig.savefig(f"ccf_peak_ident.pdf", bbox_inches="tight", dpi=300)
    # plt.close()


# print("\a")
# ccf_plot_heatmap(obs_spec, col, 'HD152478')
