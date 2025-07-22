from charaHelperFunctions import confidence_ellipse, pa, comp_sep
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from cmcrameri import cm
import pandas as pd

def companion_position(file_name):
    with open(f"{file_name}") as f:
        a = json.load(f)
        cov_mat = np.array([[a["cov_mat"]["x"]["x"], a["cov_mat"]["x"]["y"]],
                            [a["cov_mat"]["y"]["x"], a["cov_mat"]["y"]["y"]]], dtype=float)
        ra, dec = (float(a["companion_params"]["x"]), float(a["companion_params"]["y"]))
        err_ra, err_dec = np.sqrt(np.diag(cov_mat))

        obs_date = file_name.split("/")[-1].split("_")[0]
        obs_time = a["obs_time"]

    return ra, err_ra, dec, err_dec, cov_mat, obs_date, obs_time

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

        print(" "*25 + f"{dat[6]:.4f}\t" + f"\tPA = {pos_angles[0]:.3f} ± {pos_angles[1]:.3e}" +
              f"\tSep = {seps[0]:.3f} ± {seps[1]:.3e}")
        confidence_ellipse(np.array([dat[0]]), np.array([dat[2]]), ax, n_std=3, cov=dat[4], edgecolor='k', zorder=0)
        ax.scatter(dat[0], dat[2], color=colors[i], marker="x", s=100, label=f"MIRCX {dat[5]}")
        ax.errorbar(dat[0], dat[2], color=colors[i], xerr=dat[1], yerr=dat[3], marker="x")

    ax.set_xlim(-9.8,9.8)
    ax.set_ylim(-9.8,9.8)
    ax.xaxis.set_inverted(True)
    ax.set_title(fr'{star_name} Interferometric Orbit', fontsize=20)
    ax.set_xlabel(r"E $\leftarrow\, \Delta \alpha$ (mas)", fontsize=18)
    ax.set_ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)', fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=8, width=1)
    ax.yaxis.get_offset_text().set_size(20)
    ax.legend(fontsize=10, ncol=len(comp_data) // 2, loc="upper center")
    fig.savefig(f"CHARA/VisualOrbits/Visual_Orbit_{star_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    pas = pd.Series(pas)
    pa_errs = pd.Series(pa_errs)
    separations = pd.Series(separations)
    separation_errs = pd.Series(separation_errs)
    obs_times = pd.Series(obs_times)

    df = pd.concat([obs_times, pas, pa_errs, separations, separation_errs], axis="columns")
    # df.columns = ["Wavelength", "Flux"]
    df.to_csv(f"CHARA/VisualOrbits/{star_name}.txt", index=False, sep="\t", header=False, float_format="%.6f")


if __name__ == '__main__':
    json_files = glob.glob("CHARA/CompanionParams/*HD191610.json")

    companion_data = []
    for file in json_files:
        companion_data.append(companion_position(file))

    orbit_plotter(companion_data, "HD191610")

    # json_files = glob.glob("CHARA/CompanionParams/*HD200310.json")
    #
    # companion_data = []
    # for file in json_files:
    #     companion_data.append(companion_position(file))

    # orbit_plotter(companion_data, "HD200310")
    #
    # hdul = fits.open("CHARA/CANDID-master/candid/demo/MIRCX_L2.2021Jul03.28_Cyg.MIRCX_IDL.1.SPLIT.oifits")
    #
    # # Loop through and find all tables with MJD
    # for hdu in hdul:
    #     extname = hdu.header.get('EXTNAME', '')
    #     if extname in ['OI_T3']:
    #         mjds = np.mean(hdu.data['MJD'])
    #         print(f"MJD values from {extname}:")
    #         print(mjds)
    #
    #     else:
    #         mjds = np.nan




'''
ra1, dec1 = (-1.379, -3.971)
ra1_err = np.array([[0.030], [0.028]])
y3_err = np.array([[0.036], [0.035]])
cov_mat3 = np.array([[0.00023588541841400182, -9.229969565858467e-05],
                     [-9.229969565858467e-05, 0.0002318667990351338]])

ra2, dec2 = (-4.422, -0.686)
ra2_err = np.array([[0.040], [0.057]])
dec2_err = np.array([[0.059], [0.051]])
cov_mat7 = np.array([[0.0001588813250489883, -5.672761774141208e-05],
                     [-5.672761774141208e-05, 0.00016870022188802155]])

ra3, dec3 = (1.737, -6.181)
ra3_err = np.array([[0.023], [0.021]])
dec3_err = np.array([[0.019], [0.017]])
cov_mat6 = np.array([[0.0001588813250489883, -5.672761774141208e-05],
                     [-5.672761774141208e-05, 0.00016870022188802155]])

ra4, dec4 = (4.362, -7.346)
ra4_err = np.array([[0.029], [0.032]])
dec4_err = np.array([[0.017], [0.018]])
cov_mat4 = np.array([[0.0001588813250489883, -5.672761774141208e-05],
                     [-5.672761774141208e-05, 0.00016870022188802155]])
'''