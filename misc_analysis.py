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
import json
quantity_support()

def add_constellations(ax, geojson_data, label_fontsize=8):
    for feature in geojson_data['features']:
        name = feature['id']  # constellation name
        multi_lines = feature['geometry']['coordinates']

        all_ra = []
        all_dec = []

        for line in multi_lines:
            if len(line) < 2:
                continue
            ra = np.array([p[0] for p in line])
            dec = np.array([p[1] for p in line])

            # wrap RA to [-180, 180]
            ra_wrapped = ((ra + 180) % 360) - 180

            all_ra.extend(ra_wrapped)
            all_dec.extend(dec)

            # split lines across RA wrap as before
            split_indices = np.where(np.abs(np.diff(ra_wrapped)) > 180)[0]
            start = 0
            for idx in split_indices:
                coords = SkyCoord(ra_wrapped[start:idx+1]*u.deg, dec[start:idx+1]*u.deg, frame='icrs')
                ra_r = -coords.ra.wrap_at('180d').radian
                dec_r = coords.dec.radian
                ax.plot(ra_r, dec_r, color='gray', linewidth=0.6, alpha=0.5, zorder=1)
                start = idx+1

            # final segment
            coords = SkyCoord(ra_wrapped[start:]*u.deg, dec[start:]*u.deg, frame='icrs')
            ra_r = -coords.ra.wrap_at('180d').radian
            dec_r = coords.dec.radian
            ax.plot(ra_r, dec_r, color='gray', linewidth=0.6, alpha=0.5, zorder=1)

        # add label at centroid
        if len(all_ra) > 0:
            centroid_coord = SkyCoord(np.mean(all_ra)*u.deg, np.mean(all_dec)*u.deg, frame='icrs')
            ra_r = -centroid_coord.ra.wrap_at('180d').radian
            dec_r = centroid_coord.dec.radian
            ax.text(ra_r, dec_r, name, fontsize=label_fontsize, color='black', alpha=0.7,
                    ha='center', va='center', zorder=3)

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
        if name in np.array(hst_inventory[1]):
            individual_obs_count += np.array(hst_inventory[2][np.array(hst_inventory[1]) == name])[0]
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
    ax.set_title("Be+sdOB Targets", fontsize=20)
    with open("constellations_lines.json", "r") as f:
        constellation_data = json.load(f)
    add_constellations(ax, constellation_data)

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

# sky_plot(interactive=False)