import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d

def model_func(x, a, b, c, d, e, f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def list_fits_files(directory):
    """
    Recursively finds all files with a .fits extension in a given directory and its subdirectories.

    Parameters:
        directory (str): The path to the directory to search for .fits files.

    Returns:
        list: A list of file paths to all .fits files found in the directory and its subdirectories.
     """
    fits_files = []

    # Loop through all subdirectories
    for subdir, _, files in os.walk(directory):
        # Loop through files in each subdirectory
        for file in files:
            # Check if the file has a .fits extension
            if file.endswith(".fits"):
                # Append full file path to the list
                fits_files.append(os.path.join(subdir, file))

    return fits_files


# Define the recursive sigma clipping function
def recursive_sigma_clipping(wavelengths, fluxes, star_name, order, degree=5, sigma_threshold=3, max_iterations=10,
                             blaze_plots=False):
    """
    Perform a recursive sigma clipping algorithm to fit the continuum of spectral data using scipy curve_fit.
    This pipeline follows the steps laid out by §3.3 of Paredes, L. A., et al. 2021, AJ, 162, 176 and assumes slicer mode

    Parameters:
        wavelengths (array-like): Array of wavelengths.
        fluxes (array-like): Array of flux values corresponding to the wavelengths.
        star_name (string): Name of star
        order (int or str): Echelle order number for blaze plotting
        degree (int): Degree of the polynomial used for fitting the continuum.
        sigma_threshold (float): Number of standard deviations to use as the clipping threshold.
        max_iterations (int): Maximum number of iterations to perform.
        blaze_plots (bool): Default=False, plots the blaze function fits for this spectrum

    Returns:
        tuple: (continuum_fit, mask), where:
            - continuum_fit (array-like): The fitted continuum values.
            - mask (array-like): Boolean array indicating which points are retained.
    """
    # Ensure input data is in numpy array format
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)

    # Normalize wavelengths to improve numerical stability
    wavelengths_norm = (wavelengths - np.mean(wavelengths)) / np.std(wavelengths)

    # Define the polynomial function for fitting
    def polynomial(x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))

    # Initialize the mask to include all points initially
    mask = np.ones_like(fluxes, dtype=bool)

    for iteration in range(max_iterations):
        # Fit the data using scipy curve_fit for the retained points
        popt, _ = curve_fit(polynomial, wavelengths_norm[mask], fluxes[mask], p0=np.zeros(degree + 1))
        continuum = polynomial(wavelengths_norm, *popt)

        # Compute residuals and standard deviation of residuals
        residuals = fluxes - continuum
        std_dev = np.std(residuals[mask])

        # Update the mask by excluding points outside the sigma threshold
        new_mask = np.abs(residuals) <= sigma_threshold * std_dev

        # If the mask doesn't change, we have converged
        if np.array_equal(mask, new_mask):
            break

        mask = new_mask

    # Compute the final continuum fit
    continuum_fit = polynomial(wavelengths_norm, *popt)

    if blaze_plots:
        if os.path.exists(f"CHIRON_Spectra/StarSpectra/Plots/Blaze_Function/{star_name}"):
            pass
        else:
            os.mkdir(f"CHIRON_Spectra/StarSpectra/Plots/Blaze_Function/{star_name}")
            print(f"-->CHIRON_Spectra/StarSpectra/Plots/Blaze_Function/{star_name} directory created, blaze function "
                  f"plots will be saved here!")

        plt.rcParams['font.family'] = 'Geneva'
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(wavelengths, fluxes, c='k')
        ax.plot(wavelengths, continuum_fit, c="red", alpha=0.7, linewidth=5)
        ax.set_title(f'Blaze Function', fontsize=24)
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.set_ylabel("Un-Normalized Flux", fontsize=22)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.tick_params(axis='y', which='major', labelsize=20)
        ax.tick_params(axis='x', which='major', labelsize=20)
        ax.tick_params(axis='both', which='major', length=10, width=1)
        ax.yaxis.get_offset_text().set_size(20)
        fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/Blaze_Function/{star_name}/{star_name}_{order}.pdf",
                    bbox_inches="tight", dpi=300)

    return continuum_fit, mask


def double_gaussian_fit(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return A1*np.exp(-0.5*((x-mu1)**2)/(sigma1**2)) + A2*np.exp(-0.5*((x-mu2)**2)/(sigma2**2))

def gaussian_pair(v, sep, sigma):
    """
    Construct a pair of oppositely signed Gaussians at ±sep/2.

    Parameters:
        v: list of x values for Gaussian function
        sep: the separation of the two Gaussians
        sigma: the width of the Gaussians

    Returns:
        Array of corresponding y values.
    """
    g1 = np.exp(-0.5 * ((v - sep/2) / sigma) ** 2)
    g2 = -np.exp(-0.5 * ((v + sep / 2) / sigma) ** 2)
    return g1 + g2

def velocity_grid(wavelengths, line_center):
    """
    Converts input wavelengths to velocity shifts (in units of km/s) relative to line center.

    Parameters:
        wavelengths: array of input wavelengths
        line_center: rest wavelength of line (e.g. 6562.8 Å for H Alpha)

    Returns:
        Array of velocity shifts (in units of km/s) relative to line center.
    """
    c = 3e5
    return c * (wavelengths - line_center) / line_center

def find_zero_crossing_nearest_zero(x, y):
    """
    Finds the crossing point of a function where the sign changes from negative to positive, or vice versa, closest to
    x=0.

    Parameters:
        x: x values of function
        y: y values of function

    Returns:
        x coordinate of crossing point nearest to x=0. Returns None if no zero-crossings found.
    """
    zero_crossings = []
    sign_changes = np.where(np.diff(np.sign(y)))[0]

    for i in sign_changes:
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
        zero_crossings.append(x_zero)

    if not zero_crossings:
        return None

    return min(zero_crossings, key=lambda v: abs(v))

def shafter_bisector_velocity(wavelengths, fluxes, line_center=6562.8, sep=10, sigma=5, v_window=5000, v_step=2.6):
    """
    Finds the bisector velocity of the wings of the H Alpha line in a 1D spectrum using the oppositely signed double
    Gaussian cross-correlation method of Shafter, A. W., Szkody, P., & Thorstensen, J. R. 1986, ApJ, 308, 765.

    Parameters:
        wavelengths: array of input wavelengths
        fluxes: array of input fluxes
        line_center: rest wavelength of H Alpha (6562.8 Å)
        sep: separation of the Gaussian pair
        sigma: Gaussian width
        v_window: ± velocity range to explore (in units of km/s)
        v_step: velocity step size (in units of km/s)

    Returns:
        bisector_velocity: velocity at zero crossing closest to 0 (in km/s)
        v_grid: velocity shift grid
        ccf: cross-correlation function values

    """
    v_obs = velocity_grid(wavelengths, line_center)
    interp_flux = interp1d(v_obs, fluxes, kind="linear", bounds_error=False, fill_value=0)

    v_grid = np.arange(-v_window, v_window+v_step, v_step)
    flux_grid = interp_flux(v_grid)

    kernel = gaussian_pair(v_grid, sep=sep, sigma=sigma)

    ccf = np.correlate(flux_grid, kernel, mode="same")

    bisector_velocity = find_zero_crossing_nearest_zero(v_grid, ccf)

    return bisector_velocity, v_grid, ccf


if __name__ == '__main__':
    print("""
        ========================== This is ACHILLES ============================
          [A]kshat's [CHI]RON [L]ogistics [L]ayout for [E]chelle [S]pectroscopy
                    https://github.com/akshat-chaturvedi/BeStars
        ========================================================================
        """)
    print(' version:', __version__)
    # infile = "CHIRON_Spectra/241120_planid_1034/achi241120.1157.fits"
    #
    # with fits.open(infile) as hdul:
    #     dat = hdul[0].data
    #     # print(dir(hdul[0].data))
    #
    #
    # totalWavelengths = []
    # blazeFlux = []
    #
    # for i in tqdm(range(59)):
    #     wavs = []
    #     fluxes = []
    #     for j in range(3200):
    #         wavs.append(dat[i][j][0])
    #         fluxes.append(dat[i][j][1])
    #
    #     continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, degree=5, sigma_threshold=3)
    #     totalWavelengths.append(wavs)
    #     wavelengths = np.array(wavs)
    #     fluxes = np.array(fluxes)
    #
    #     blazeFlux.append(fluxes/continuum_fit)
    #
    # totalWavelengths = np.array(totalWavelengths).flatten()
    # blazeFlux = np.array(blazeFlux).flatten()
    #
    # plt.rcParams['font.family'] = 'Geneva'
    # fig, ax = plt.subplots(figsize=(20,10))
    # # ax.plot(wavelengths, fluxes)
    # ax.plot(totalWavelengths, blazeFlux, 'k', label='Original Data')
    # # ax.plot(wavelengths[mask], fluxes[mask], 'b.', label='Retained Data', alpha=0.8)
    # # ax.plot(wavelengths, continuum_fit, 'r-', label='Fitted Continuum', linewidth=2)
    # fig.savefig("trial_1.pdf", bbox_inches="tight", dpi=300)
    #
    # # fig, ax = plt.subplots(figsize=(20,10))
    # # # ax.plot(wavelengths, fluxes)
    # # ax.plot(wavelengths, fluxes/continuum_fit, 'k', label='Original Data')
    # # fig.savefig("trial.pdf", bbox_inches="tight", dpi=300)


