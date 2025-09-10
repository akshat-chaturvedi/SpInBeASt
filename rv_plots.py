import thejoker as tj
import matplotlib.pyplot as plt
import numpy as np
from thejoker.plot import plot_rv_curves
import pandas as pd
import astropy.units as u
import pickle
import os
from SpectrumAnalyzer import spin_beast

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def rv_plot(star_name, show_raw_plot=False, params=None):
    spec_type = input(f"Are the observations {RED}optical [O]{RESET} or {BLUE}UV [U]{RESET}?: ")
    if spec_type in ["o", "O"]:
        spec_flag = "Optical"
        print(f"->Starting analysis of {RED}optical{RESET} observations of {RED}{star_name}{RESET}")
    elif spec_type in ["u", "U"]:
        spec_flag = "UV"
        print(f"->Starting analysis of {BLUE}UV{RESET} observations of {BLUE}{star_name}{RESET}")
    else:
        exit("ERROR: Please enter valid observation type!")

    # Create path for star if it doesn't already exist
    if os.path.exists(f"RV_Plots/{spec_flag}/{star_name}"):
        pass
    else:
        os.mkdir(f"RV_Plots/{spec_flag}/{star_name}")

        print(f"{GREEN}-->RV_Plots/{spec_flag}/{star_name} directory created, plots will be saved here!{RESET}")

    # Read in data, and plot measured RVs if needed
    plt.rcParams['font.family'] = 'Geneva'
    t = list(dat[0])
    rv = list(dat[1]) * u.km/u.s
    try:
        err = list(dat[2]) * u.km / u.s
    except:
        err = np.ones(len(dat[1])) * 0.5 * u.km/u.s  # If errors aren't provided, assume a given error
    data = tj.RVData(t=t, rv=rv, rv_err=err)
    if show_raw_plot:
        ax = data.plot()
        # ax.set_xlim(-10, 200)
        plt.show()

    pickle_check = False  # Flag to check if {star_name}_true-orbit.pkl exists
    pickle_file_name = f"RV_Plots/{spec_flag}/{star_name}/{star_name}_true-orbit.pkl"
    if os.path.isfile(pickle_file_name):  # Check if {star_name}_true-orbit.pkl exists
        pickle_check = True
        print(f"{GREEN}-->Saved best-fit orbit parameters found!{RESET}")
        with open(pickle_file_name, "rb") as f:
            b = pickle.load(f)
            p_min = b["P"].value - 20
            p_max = b["P"].value + 20
            semi_amplitude = b["K"].value
            systemic_velocity = b["v0"].value

    # If pickle file doesn't exist, either take in values from parameters provided in the function, or type them out
    if not pickle_check:
        print(f"{YELLOW}-->Saved best-fit orbit parameters not found, checking to see if priors are provided in the "
              f"function call{RESET}")
        if params:
            print(f"{GREEN}-->Priors found in function call!{RESET}")
            p_min, p_max, semi_amplitude, systemic_velocity = params

        else:
            print(f"{YELLOW}-->Priors not found in function call, please enter priors bellow{RESET}")
            p_min = float(input("--->Enter minimum period estimate: "))
            p_max = float(input("--->Enter maximum period estimate: "))
            semi_amplitude = float(input("--->Enter semi-amplitude estimate: "))
            systemic_velocity = float(input("--->Enter systemic velocity estimate: "))

    print(f"-->Starting TheJoker sampling with provided priors. 3 plots will be created:\n"
          f"   •{star_name}_posterior.pdf\n"
          f"   •{star_name}_rv_plot.pdf\n"
          f"   •{star_name}_phase_folded.pdf")
    # Create prior for TheJoker with provided parameters
    prior = tj.JokerPrior.default(
        P_min=p_min * u.day, P_max=p_max * u.day,
        sigma_K0=semi_amplitude * u.km / u.s,
        sigma_v=systemic_velocity * u.km / u.s
    )

    joker = tj.TheJoker(prior)
    prior_samples = prior.sample(size=200_000)
    samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)

    # Creating posterior plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")
    ax.scatter(samples['P'].value, samples['K'].to(u.km / u.s).value,
               marker='.', color='k', alpha=0.45)
    ax.set_xlabel("$P$ [day]")
    ax.set_ylabel("$K$ [km/s]")
    fig.savefig(f"RV_Plots/{spec_flag}/{star_name}/{star_name}_posterior.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Creating RV with all posteriors plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")
    plot_rv_curves(samples, rv_unit=u.km / u.s, data=data, ax=ax,
                   plot_kwargs=dict(color='#888888'),
                   data_plot_kwargs=dict(color="tab:red"))
    fig.savefig(f"RV_Plots/{spec_flag}/{star_name}/{star_name}_rv_plot.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Create dictionary of median orbital parameters and save it as a pickle file
    my_arr = samples.median_period().pack()
    orbit = samples.median_period()
    orbit_dict = {
        'P': orbit["P"][0],
        'M0': orbit["M0"][0],
        'omega': orbit["omega"][0],
        'e': orbit["e"][0],
        'K': orbit["K"][0],
        'v0': orbit["v0"][0]
    }

    if not pickle_check:
        print(f"{GREEN}-->Saving best-fit parameters!{RESET}")
        with open(f"RV_Plots/{spec_flag}/{star_name}/{star_name}_true-orbit.pkl", "wb") as f:
            pickle.dump(orbit_dict, f)

    # Create TwoBody object with orbital parameters
    orbit = samples.median_period().get_orbit()

    # Create phase-folded RV plot
    P = orbit.P
    t0 = samples.median_period().get_t0()
    unit_phase_grid = np.linspace(0, 1, 1000)
    phase_grid = unit_phase_grid

    dt_jd = (data.t - t0).tcb.jd * u.day  # Get JD time relative to t0
    phase = (dt_jd / P) % 1  # Get phase

    # jds = Time(t, format='mjd', scale='tcb')
    resids = data.rv - orbit.radial_velocity(data.t)  # Calculate residuals

    # Plot phase-folded RV curve
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
    plt.subplots_adjust(hspace=0)
    fig.supxlabel("Phase", fontsize=24)
    ax[0].errorbar(phase, data.rv, yerr=data.rv_err, color="red", fmt="o", markersize=10, capsize=5, mec="k", mew=2,
                   mfc="r", label=f"{star_name}")
    ax[0].plot(phase_grid, orbit.radial_velocity(t0+P*unit_phase_grid), color="k", label=f'P={my_arr[0][0][0]:.2f} d')
    ax[0].set_ylabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    ax[0].legend(fontsize=22)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].set_xlim(0, 1)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                   width=1)
    ax[0].text(
        0.85, 0.7,  # (x, y) in axes coordinates (1.0 is right/top)
        f"P={orbit_dict['P']:.2f}\nM={orbit_dict['M0']:.2f}\nomega={orbit_dict['omega']:.2f}\ne={orbit_dict['e']:.2f}\nK={orbit_dict['K']:.2f}"
        f"\nv0={orbit_dict['v0']:.2f}",  # Text string
        ha='left', va='bottom',  # Horizontal and vertical alignment
        transform=ax[0].transAxes,  # Use axes coordinates
        fontsize=16,
        fontweight='bold',
        bbox=dict(
            facecolor='white',  # Box background color
            edgecolor='black',  # Box border color
            boxstyle='square,pad=0.3',  # Rounded box with padding
            alpha=0.9  # Slight transparency
        )
    )
    ax[1].errorbar(phase, resids, yerr=data.rv_err, color="red", fmt="o", ms=8, capsize=5, mec="k", mew=2, mfc="r")
    ax[1].set_ylabel("O-C", fontsize=24)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
                      width=1)
    ax[1].yaxis.get_offset_text().set_size(22)
    ax[1].hlines(0, 0, 1, color="k", linestyle="--", zorder=0)
    fig.savefig(f"RV_Plots/{spec_flag}/{star_name}/{star_name}_phase_folded.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    final_dat = pd.concat([pd.Series(phase), pd.Series(data.rv)], axis='columns')
    final_dat.columns = ['Phase', 'RV_obs']
    final_dat.to_csv(f"RV_Plots/{spec_flag}/{star_name}/{star_name}_phase_folded.txt", index=False)

    final_dat = pd.concat([pd.Series(phase_grid), pd.Series(orbit.radial_velocity(t0+P*unit_phase_grid))], axis='columns')
    final_dat.columns = ['Phase', 'RV_calc']
    final_dat.to_csv(f"RV_Plots/{spec_flag}/{star_name}/{star_name}_phase_folded_model.txt", index=False)

    print(f"{GREEN}RV orbit-fitting analysis finished!{RESET}")
    return


if __name__ == '__main__':
    print(spin_beast)
    dat = pd.read_csv("RV_Data/LSMus.txt", header=None)
    rv_plot("HD113120", show_raw_plot=True)

    # opt_dat = pd.read_csv("RV_Plots/Optical/HD113120/HD113120_phase_folded.txt")
    # opt_model = pd.read_csv("RV_Plots/Optical/HD113120/HD113120_phase_folded_model.txt")
    #
    # uv_dat = pd.read_csv("RV_Plots/UV/HD113120/HD113120_phase_folded.txt")
    # uv_model = pd.read_csv("RV_Plots/UV/HD113120/HD113120_phase_folded_model.txt")
    #
    # fig, ax = plt.subplots(figsize=(20, 10))
    # plt.subplots_adjust(hspace=0)
    # fig.supxlabel("Phase", fontsize=24)
    # ax.scatter(opt_dat['Phase'], opt_dat['RV_obs']-np.mean(opt_dat['RV_obs']), color="red", s=100, label="CHIRON", zorder=15)
    # ax.plot(opt_model['Phase'], opt_model['RV_calc']-np.mean(opt_model['RV_calc']), color="k")
    # ax.scatter(uv_dat['Phase'], uv_dat['RV_obs']-np.mean(uv_dat['RV_obs']), color="dodgerblue", s=100, label="HST", zorder=15)
    # ax.plot(uv_model['Phase'], uv_model['RV_calc']-np.mean(uv_model['RV_calc']), color="k")
    # ax.set_ylabel("Radial Velocity [km s$^{-1}$]", fontsize=22)
    # ax.legend(fontsize=22)
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='y', labelsize=20)
    # ax.set_xlim(0, 1)
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=22, top=True, right=True, length=10,
    #                   width=1)
    # fig.savefig(f"RV_Plots/HD113120_phase_folded_both.pdf", bbox_inches="tight", dpi=300)
    # plt.close()
