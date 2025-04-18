import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import os

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
def recursive_sigma_clipping(wavelengths, fluxes, degree=5, sigma_threshold=3, max_iterations=10):
    """
    Perform a recursive sigma clipping algorithm to fit the continuum of spectral data using scipy curve_fit.
    This pipeline follows the steps laid out by §3.3 of Paredes, L. A., et al. 2021, AJ, 162, 176 and assumes slicer mode

    Parameters:
        wavelengths (array-like): Array of wavelengths.
        fluxes (array-like): Array of flux values corresponding to the wavelengths.
        degree (int): Degree of the polynomial used for fitting the continuum.
        sigma_threshold (float): Number of standard deviations to use as the clipping threshold.
        max_iterations (int): Maximum number of iterations to perform.

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
            # print("\033[92mSigma clipping algorithm converged!\033[0m")
            break

        mask = new_mask

    # Compute the final continuum fit
    continuum_fit = polynomial(wavelengths_norm, *popt)

    return continuum_fit, mask


def double_gaussian_fit(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return A1*np.exp(-0.5*((x-mu1)**2)/(sigma1**2)) + A2*np.exp(-0.5*((x-mu2)**2)/(sigma2**2))


if __name__ == '__main__':

    infile = "CHIRON_Spectra/241120_planid_1034/achi241120.1157.fits"

    with fits.open(infile) as hdul:
        dat = hdul[0].data
        # print(dir(hdul[0].data))


    totalWavelengths = []
    blazeFlux = []

    for i in tqdm(range(59)):
        wavs = []
        fluxes = []
        for j in range(3200):
            wavs.append(dat[i][j][0])
            fluxes.append(dat[i][j][1])

        continuum_fit, mask = recursive_sigma_clipping(wavs, fluxes, degree=5, sigma_threshold=3)
        totalWavelengths.append(wavs)
        wavelengths = np.array(wavs)
        fluxes = np.array(fluxes)

        blazeFlux.append(fluxes/continuum_fit)

    totalWavelengths = np.array(totalWavelengths).flatten()
    blazeFlux = np.array(blazeFlux).flatten()

    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(20,10))
    # ax.plot(wavelengths, fluxes)
    ax.plot(totalWavelengths, blazeFlux, 'k', label='Original Data')
    # ax.plot(wavelengths[mask], fluxes[mask], 'b.', label='Retained Data', alpha=0.8)
    # ax.plot(wavelengths, continuum_fit, 'r-', label='Fitted Continuum', linewidth=2)
    fig.savefig("trial_1.pdf", bbox_inches="tight", dpi=300)

    # fig, ax = plt.subplots(figsize=(20,10))
    # # ax.plot(wavelengths, fluxes)
    # ax.plot(wavelengths, fluxes/continuum_fit, 'k', label='Original Data')
    # fig.savefig("trial.pdf", bbox_inches="tight", dpi=300)


