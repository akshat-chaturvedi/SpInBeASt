import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import quantity_support
from astropy.table import Table
from astropy.io import ascii
from cmcrameri import cm
quantity_support()

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

    data = Table()
    data['Name'] = np.array(star_names)
    data["RA"] = np.array(ra)
    data["Dec"] = np.array(dec)
    data["Number of Observations"] = np.array(overall_obs_count)
    ascii.write(data, 'Be_sdO_Target_Inventory.txt', format='fixed_width', overwrite=True,
                formats={'RA': '%.5f', 'Dec': '%.5f'}, bookend=False, delimiter=None)

    eq = SkyCoord(ra, dec, unit=u.deg)

    eq_ra, eq_dec = -eq.ra.wrap_at('180d').radian, eq.dec.radian

    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "aitoff"})
    ax.set_xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]),
               labels=['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°'])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    scatter = ax.scatter(eq_ra, eq_dec, s=50, zorder=2, c=overall_obs_count, cmap=cm.managua, edgecolors="k")
    ax.tick_params(axis='both', which='major', labelsize=18)
    cbar = fig.colorbar(scatter, orientation='vertical', ticks=np.arange(max(overall_obs_count)+1), pad=0.01)
    cbar.set_label('Number of Observations', fontsize=18)
    cbar.ax.tick_params(labelsize=18, length=8, width=1)

    # Add interactivity with mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(star_names[sel.index]))
    ax.set_title("Be+sdO Targets", fontsize=20)

    if interactive:
        plt.show()

    else:
        fig.savefig("skyplot.pdf", bbox_inches="tight", dpi=300)
        plt.close()


def rv_plotter(filename):
    dat = pd.read_csv(filename, header=None)
    star_name = filename.split("/")[3].split("_")[0]
    jd = dat[0]
    rv = dat[1]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(jd-2460000, rv, s=80, color="r")
    ax.set_xlabel("JD$-2460000$ [days]", fontsize=22)
    ax.set_ylabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    ax.set_title(f"{star_name} Radial Velocity", fontsize=24)
    fig.savefig(f"CHIRON_Spectra/StarSpectra/RV_Measurements/RV_{star_name}.pdf", bbox_inches="tight",
                dpi=300)
    plt.close()