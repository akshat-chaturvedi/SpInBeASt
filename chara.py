import numpy as np
import pmoired
import matplotlib.pyplot as plt
from astropy.time import Time
import json
import logging
import glob
import time
from cmcrameri import cm
import pandas as pd
from charaHelperFunctions import confidence_ellipse, pa, comp_sep
import concurrent.futures

YELLOW = '\033[93m'
GREEN = '\033[92m'
RESET = '\033[0m'

def binary_fit(filename, star_name, star_diam):
    band = 'K'  # MIRCX = H; MYSTIC = K
    oi = pmoired.OI(filename)
    obs_date = Time(np.mean(oi.data[0]['MJD']), format='mjd').fits.split("T")[0]
    obs_time = Time(np.mean(oi.data[0]['MJD']), format='mjd').value

    # -- smallest lambda/B in mas (first data set)
    step = 180 * 3600 * 1000e-6 / np.pi / max([np.max(oi.data[0]['OI_VIS2'][k]['B/wl']) for k in oi.data[0]['OI_VIS2']])

    # -- spectral resolution (first data set)
    R = np.mean(oi.data[0]['WL'] / oi.data[0]['dWL'])

    print('step: %.1fmas, range: +- %.1fmas' % (step, R / 2 * step))

    # -- initial model dict: 'c,x' and 'c,y' do not matter, as they will be explored in the fit
    param = {'*,ud': star_diam, '*,f': 1, 'c,f': 0.01, 'c,x': 0, 'c,y': 0, 'c,ud': 0.0}

    # -- define the exploration pattern
    expl = {'grid': {'c,x': (-R / 2 * step, R / 2 * step, step), 'c,y': (-R / 2 * step, R / 2 * step, step)}}

    # -- set up the fit, as usual
    oi.setupFit({'obs': ['T3PHI']})

    # -- reference fit (no companion)
    oi.doFit({'ud': star_diam})
    bestUD = oi.bestfit['best']

    # -- actual grid fit
    oi.gridFit(expl, model=param, doNotFit=['*,f', 'c,ud'], prior=[('c,f', '<', 1)],
               constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R * step / 2),
                          ('np.sqrt(c,x**2+c,y**2)', '>', step / 2)])

    oi.showGrid(interpolate=True, tight=True)
    plt.savefig(f"CHARA/Figures/{obs_date}_{star_name}_{band}_figure_1.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    oi.showGrid(interpolate=True, tight=True, significance=bestUD)
    plt.savefig(f"CHARA/Figures/{obs_date}_{star_name}_{band}_figure_2.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{GREEN}={RESET}" * 100 + "\n" +
          f"Best-fit Parameters:\nx={oi.bestfit['best']['c,x']:.4f}\ny={oi.bestfit['best']['c,y']:.4f}" +
          "\n" + f"{GREEN}={RESET}" * 100)

    best_fit_params = oi.bestfit['best']
    best_fit_cov_mat = oi.bestfit['covd']

    oi.show()
    plt.savefig(f"CHARA/Figures/{obs_date}_{star_name}_{band}_figure_3.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    oi.bootstrapFit(300)
    oi.showBootstrap()
    plt.savefig(f"CHARA/Figures/{obs_date}_{star_name}_{band}_figure_4.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    for index in best_fit_params.keys():
        best_fit_params[index] = format(best_fit_params[index], '.5e')

    for index in best_fit_cov_mat.keys():
        for key in best_fit_cov_mat[index].keys():
            best_fit_cov_mat[index][key] = format(best_fit_cov_mat[index][key], '.5e')

    if band:
        if band == "H":
            combiner = "MIRCX"
        else:
            combiner = "MYSTIC"

    else:
        combiner = "Unknown"

    comp = {'companion_params': best_fit_params, 'cov_mat': best_fit_cov_mat, 'obs_time': obs_time,
            'combiner': combiner}
    with open(f"CHARA/CompanionParams/{obs_date}_{star_name}_{band}.json", "w") as f:
        json.dump(comp, f)

    print(f'{GREEN}#{RESET}' * 75 + f"{GREEN}FINISHED!{RESET}" + f'{GREEN}#{RESET}' * 75)
    logging.info(f'OIFITS Analyzed: {star_name}, Observation Date: {obs_date}, Band: {band}')

    return

def companion_position(file_name):
    with open(f"{file_name}") as f:
        a = json.load(f)
        # cov_mat = np.array([[a["cov_mat"]["x"]["x"], a["cov_mat"]["x"]["y"]],
        #                     [a["cov_mat"]["y"]["x"], a["cov_mat"]["y"]["y"]]], dtype=float)
        # ra, dec = (float(a["companion_params"]["x"]), float(a["companion_params"]["y"]))
        cov_mat = np.array([[a["cov_mat"]["c,x"]["c,x"], a["cov_mat"]["c,x"]["c,y"]],
                            [a["cov_mat"]["c,y"]["c,x"], a["cov_mat"]["c,y"]["c,y"]]], dtype=float)
        ra, dec = (float(a["companion_params"]["c,x"]), float(a["companion_params"]["c,y"]))
        err_ra, err_dec = np.sqrt(np.diag(cov_mat))

        obs_date = file_name.split("/")[-1].split("_")[0]
        obs_time = a["obs_time"]
        combiner = a["combiner"]

    return ra, err_ra, dec, err_dec, cov_mat, obs_date, obs_time, combiner

def orbit_plotter(comp_data, star_name):
    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(0, 0, color="k", marker="+", s=200)

    cmap = cm.roma  # or cm.roma, cm.lajolla, etc.
    n = len(comp_data)  # Number of colors (e.g., for 10 lines)
    colors = [cmap(i / n) for i in range(n)]

    print("=" * 50 + f" {star_name} " + "=" * 50)
    pas = []
    pa_errs = []
    separations = []
    separation_errs = []
    obs_times = []
    for i, dat in enumerate(comp_data):
        pos_angles = pa(dat[0], dat[1], dat[2], dat[3])
        seps = comp_sep(dat[0], dat[1], dat[2], dat[3])
        pas.append(pa(dat[0], dat[1], dat[2], dat[3])[0])
        pa_errs.append(pa(dat[0], dat[1], dat[2], dat[3])[1])
        separations.append(comp_sep(dat[0], dat[1], dat[2], dat[3])[0])
        separation_errs.append(comp_sep(dat[0], dat[1], dat[2], dat[3])[1])
        obs_times.append(dat[6])

        print(" "*18 + f"{dat[5]}\t" + f"{dat[6]:.4f}\t" +f"PA = {pos_angles[0]:.3f} ± {pos_angles[1]:.3e}\t" +
              f"\tSep = {seps[0]:.3f} ± {seps[1]:.3e}")
        confidence_ellipse(np.array([dat[0]]), np.array([dat[2]]), ax, n_std=3, cov=dat[4], edgecolor='k', zorder=0)
        ax.scatter(dat[0], dat[2], color=colors[i], marker="x", s=100, label=f"{dat[7]} {dat[5]}")
        # ax.errorbar(dat[0], dat[2], color=colors[i], xerr=dat[1], yerr=dat[3], marker="x")

    ax.set_xlim(-max(separations)-0.2,max(separations)+0.2)
    ax.set_ylim(-max(separations)-0.2,max(separations)+0.2)
    ax.xaxis.set_inverted(True)
    ax.set_title(fr'{star_name} Interferometric Orbit', fontsize=20)
    ax.set_xlabel(r"E $\leftarrow\, \Delta \alpha$ (mas)", fontsize=18)
    ax.set_ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)', fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=8, width=1)
    ax.yaxis.get_offset_text().set_size(20)
    ax.legend(fontsize=10, ncol=2, loc="upper center")
    fig.savefig(f"CHARA/Figures/Visual_Orbit_{star_name}_1.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    pas = pd.Series(pas)
    pa_errs = pd.Series(pa_errs)
    separations = pd.Series(separations)
    separation_errs = pd.Series(separation_errs)
    obs_times = pd.Series(obs_times)

    df = pd.concat([obs_times, pas, pa_errs, separations, separation_errs], axis="columns")
    # df.columns = ["Wavelength", "Flux"]
    df.to_csv(f"CHARA/{star_name}.txt", index=False, sep="\t", header=False, float_format="%.6f")

    return


if __name__ == '__main__':
    pass
    # logging.basicConfig(
    #     filename='BinaryFit.log',
    #     encoding='utf-8',
    #     format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
    #     datefmt='%d/%m/%Y %I:%M:%S %p',
    #     level=logging.INFO,
    #     force=True  # IMPORTANT: Overwrites previous configs, needed in subprocesses
    # )
    # t1 = time.perf_counter()
    # # files = glob.glob("CHARA/Prepped/HD191610/*28_Cyg*.oifits")
    # # files.append('CHARA/Prepped/HD191610/2025Jul05_HD191610_NFiles01_H_test1split5m_prepped.oifits')
    # # files = ["CHARA/Prepped/HD200310/2025Jul05_HD200310_NFiles01_K_test1split5m_prepped.oifits"]
    # # # print(files)
    # # for file in files:
    # #     print(file)
    # #     binary_fit(f'{file}',
    # #                'HD200310')
    # files = glob.glob("CHARA/Prepped/HD191610/*28_Cyg*.oifits")
    # files.append('CHARA/Prepped/HD191610/2025Jul05_HD191610_NFiles01_H_test1split5m_prepped.oifits')
    # star_names = ['HD191610'] * len(files)
    # bands = ['H'] * len(files)
    #
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(chara_main, files, star_names, bands)
    #
    # t2 = time.perf_counter()
    # print(f'Finished in \033[94m{round(t2 - t1, 2)}\033[0m second(s)')
    # print("\a")

    # On galileo, Finished in 558.12 second(s) w/p multiproc