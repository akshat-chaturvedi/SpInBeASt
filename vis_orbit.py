from charaHelperFunctions import confidence_ellipse, pa, comp_sep
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from cmcrameri import cm
import pandas as pd
from star_props import stellar_properties
from astropy.io import ascii
import re

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
    fig.savefig(f"CHARA/Figures/Visual_Orbit_{star_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    pas = pd.Series(pas)
    pa_errs = pd.Series(pa_errs)
    separations = pd.Series(separations)
    separation_errs = pd.Series(separation_errs)
    obs_times = pd.Series(obs_times)

    df = pd.concat([obs_times, pas, pa_errs, separations, separation_errs], axis="columns")
    # df.columns = ["Wavelength", "Flux"]
    df.to_csv(f"CHARA/{star_name}.txt", index=False, sep="\t", header=False, float_format="%.6f")


def vis_orbit_1(comp_data, star_name):
    dist = stellar_properties[f"{star_name}"]["distance"]

    mas_to_au = dist * 1e-3

    def ra_mas_to_au(x): return x * mas_to_au

    def ra_au_to_mas(x): return x / mas_to_au

    def dec_mas_to_au(y): return y * mas_to_au

    def dec_au_to_mas(y): return y / mas_to_au

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
    phys_seps_ra = []
    phys_seps_dec = []
    ra = []
    dec = []
    for i, dat in enumerate(comp_data):
        pos_angles = pa(dat[0], dat[1], dat[2], dat[3])
        seps = comp_sep(dat[0], dat[1], dat[2], dat[3])
        pas.append(pa(dat[0], dat[1], dat[2], dat[3])[0])
        pa_errs.append(pa(dat[0], dat[1], dat[2], dat[3])[1])
        separations.append(seps[0])
        separation_errs.append(seps[1])
        obs_times.append(dat[6])
        ra.append(dat[0])
        dec.append(dat[2])

        print(" " * 18 + f"{dat[5]}\t" + f"{dat[6]:.4f}\t" + f"PA = {pos_angles[0]:.3f} ± {pos_angles[1]:.3e}\t" +
              f"\tSep = {seps[0]:.3f} ± {seps[1]:.3e}")
    # confidence_ellipse(np.array([dat[0]]), np.array([dat[2]]), ax, n_std=3, cov=dat[4], edgecolor='k', zorder=0)
    sc = ax.scatter(ra, dec, c=obs_times, marker="x", s=100, cmap=cm.roma)
        # ax.errorbar(dat[0], dat[2], color=colors[i], xerr=dat[1], yerr=dat[3], marker="x")
    cbar = fig.colorbar(sc, orientation='vertical', pad=0.15)
    cbar.set_label('Observation Epoch [HJD]', fontsize=16)
    cbar.ax.tick_params(labelsize=14, length=8, width=1)
    ax.set_xlim(-max(separations) - 0.2, max(separations) + 0.2)
    ax.set_ylim(-max(separations) - 0.2, max(separations) + 0.2)
    ax.xaxis.set_inverted(True)
    # ax.set_title(fr'{star_name} Interferometric Orbit', fontsize=20)
    ax.text(
        0.05, 0.05,  # (x, y) in axes coordinates (1.0 is right/top)
        f"{re.sub(r"([A-Za-z]+)(\d+)", r"\1 \2", star_name)}",  # Text string
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
    ax.set_xlabel(r"E $\leftarrow\, \Delta \alpha$ [mas]", fontsize=18)
    ax.set_ylabel(r'$\Delta \delta\, \rightarrow$ N [mas]', fontsize=18)

    secax_x = ax.secondary_xaxis("top", functions=(ra_mas_to_au, ra_au_to_mas))
    secax_x.set_xlabel(r"E $\leftarrow\, \Delta \alpha$ [AU]", fontsize=18)
    secax_x.tick_params(labelsize=16, which='major')
    secax_x.tick_params(axis='both', which='both', direction='in', length=8, width=1)

    secax_y = ax.secondary_yaxis("right", functions=(dec_mas_to_au, dec_au_to_mas))
    secax_y.set_ylabel(r'$\Delta \delta\, \rightarrow$ N [AU]', fontsize=18)
    secax_y.tick_params(labelsize=16, which='major')
    secax_y.tick_params(axis='both', which='both', direction='in', length=8, width=1)

    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=8, width=1)
    ax.yaxis.get_offset_text().set_size(20)
    # ax.legend(fontsize=10, ncol=2, loc="upper center")
    fig.savefig(f"CHARA/Figures/Visual_Orbits/Visual_Orbit_{star_name}_1.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def vis_orbit_plotter(P, T0, a, i, Omega, omega, star_name, star_pos):
    # P = 10.0  # period (years)
    # T0 = 0.0  # epoch of periastron
    # a = 1.0  # semi-major axis (arcsec)
    # i = np.radians(60.0)  # inclination
    # Omega = np.radians(30.0)  # longitude of ascending node
    # omega = np.radians(90.0)  # argument of periastron (fixed)

    comp_ra = star_pos['col2']
    comp_dec = star_pos['col3']

    dist = stellar_properties[f"{star_name}"]["distance"]

    mas_to_au = dist * 1e-3

    def ra_mas_to_au(x): return x * mas_to_au

    def ra_au_to_mas(x): return x / mas_to_au

    def dec_mas_to_au(y): return y * mas_to_au

    def dec_au_to_mas(y): return y / mas_to_au

    # Time array
    t = np.linspace(0, P, 1000)
    nu = 2 * np.pi * (t - T0) / P  # true anomaly for circular orbit

    # Position in sky-plane
    X = a * (np.cos(Omega) * np.cos(nu + omega) - np.sin(Omega) * np.sin(nu + omega) * np.cos(i))
    Y = a * (np.sin(Omega) * np.cos(nu + omega) + np.cos(Omega) * np.sin(nu + omega) * np.cos(i))

    Xp = -a * (np.cos(Omega) * np.cos(omega) - np.sin(Omega) * np.sin(omega) * np.cos(i))
    Yp = -a * (np.sin(Omega) * np.cos(omega) + np.cos(Omega) * np.sin(omega) * np.cos(i))

    # Line of nodes (through origin, direction (cosΩ, sinΩ))
    node_length = a
    X_nodes = [-node_length * np.cos(Omega), node_length * np.cos(Omega)]
    Y_nodes = [-node_length * np.sin(Omega), node_length * np.sin(Omega)]

    # Plot
    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(12,10))
    ax.plot(X, Y, 'k--')
    ax.scatter(0, 0, color="red", marker="*", s=300, linewidth=3, zorder=10, label="Primary")
    # ax.scatter(comp_ra, comp_dec, c="k", marker="x", s=250, linewidth=3, zorder=10, label="Companion")
    sc = ax.scatter(comp_ra, comp_dec, c=star_pos['col1'], marker="x", s=250, linewidth=3, zorder=10, cmap=cm.berlin, label="Companion")
    # ax.errorbar(dat[0], dat[2], color=colors[i], xerr=dat[1], yerr=dat[3], marker="x")
    cbar = fig.colorbar(sc, orientation='vertical', pad=0.12)
    cbar.set_label('Observation Epoch [HJD]', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    ax.scatter(X_nodes[0], Y_nodes[0], color='xkcd:goldenrod', marker="+", s=300, label='T0', zorder=10, linewidth=3)
    ax.scatter([Xp], [Yp], color='xkcd:kelly green', marker='+', s=300, label="Periastron", linewidth=3, zorder=10)
    ax.plot(X_nodes, Y_nodes, 'k-')
    ax.xaxis.set_inverted(True)
    ax.set_xlabel("ΔRA (arcsec)", fontsize=22)
    ax.set_ylabel("ΔDec (arcsec)", fontsize=22)
    ax.legend(fontsize=18)
    # ax.set_title("Binary Orbit Projection", fontsize=24)
    ax.tick_params(axis='both', which='major', length=10, width=1, labelsize=18)
    secax_x = ax.secondary_xaxis("top", functions=(ra_mas_to_au, ra_au_to_mas))
    secax_x.set_xlabel(r"E $\leftarrow\, \Delta \alpha$ [AU]", fontsize=18)
    secax_x.tick_params(labelsize=18, which='major')
    secax_x.tick_params(axis='both', which='both', direction='in', length=8, width=1)

    secax_y = ax.secondary_yaxis("right", functions=(dec_mas_to_au, dec_au_to_mas))
    secax_y.set_ylabel(r'$\Delta \delta\, \rightarrow$ N [AU]', fontsize=18)
    secax_y.tick_params(labelsize=18, which='major')
    secax_y.tick_params(axis='both', which='both', direction='in', length=8, width=1)
    ax.yaxis.get_offset_text().set_size(20)
    fig.savefig(f"{star_name}_VisualOrbit.pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # json_files = glob.glob("CHARA/CompanionParams/*HD191610_H.json")
    # json_files.sort()
    # companion_data = []
    # for file in json_files:
    #     companion_data.append(companion_position(file))
    #
    # orbit_plotter(companion_data, "HD191610")
    #
    # json_files = glob.glob("CHARA/CompanionParams/*HD200310_H.json")
    # json_files.sort()
    # companion_data = []
    # for file in json_files:
    #     companion_data.append(companion_position(file))
    #
    # orbit_plotter(companion_data, "HD200310")
    # json_files = glob.glob("CHARA/CompanionParams/*HD191610*.json")
    # json_files.sort()
    # companion_data = []
    # for file in json_files:
    #     companion_data.append(companion_position(file))
    # #
    # vis_orbit_1(companion_data, "HD191610")

    dat = ascii.read("HD041335_CHARA.txt")
    print(dat)

    P = 80.8733  # period (years)
    T0 = 59541.3  # epoch of periastron
    a = 1.914  # semi-major axis (arcsec)
    i = np.radians(77.7)  # inclination
    Omega = np.radians(60)  # longitude of ascending node
    omega = np.radians(90.0)  # argument of periastron (fixed)
    vis_orbit_plotter(P, T0, a, i, Omega, omega, 'HD041335', dat)
    # vis_orbit_1('CHARA/HD191610.txt', 'HD191610')
