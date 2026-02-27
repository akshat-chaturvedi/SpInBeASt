import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d
import re
from astropy.stats import sigma_clipped_stats
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

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
def recursive_sigma_clipping(wavelengths, fluxes, star_name, obs_date, order, degree=5, sigma_threshold=3, max_iterations=10,
                             blaze_plots=False):
    """
    Perform a recursive sigma clipping algorithm to fit the continuum of spectral data using scipy curve_fit.
    This pipeline follows the steps laid out by §3.3 of Paredes, L. A., et al. 2021, AJ, 162, 176 and assumes slicer mode

    Parameters:
        wavelengths (array-like): Array of wavelengths.
        fluxes (array-like): Array of flux values corresponding to the wavelengths.
        star_name (string): Name of star
        obs_date (string): Observation date
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
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(wavelengths, fluxes, c='k', label="Spectrum")
        ax.scatter(wavelengths[mask], fluxes[mask], c='xkcd:goldenrod', label="Points for Continuum Fit")
        ax.plot(wavelengths, continuum_fit, c="red", alpha=0.7, linewidth=5, label="Blaze Function Fit")
        ax.set_title(f'Blaze Function for {star_name} Order {order}', fontsize=24)
        ax.text(0.15, 0.8, fr"{clean_star_name3(star_name)} H$\alpha$",
                   color="k", fontsize=18, transform=ax.transAxes,
                   bbox=dict(
                       facecolor='white',  # Box background color
                       edgecolor='black',  # Box border color
                       boxstyle='square,pad=0.3',  # Rounded box with padding
                       alpha=0.9  # Slight transparency
                   )
                   )
        ax.set_xlabel("Wavelength [Å]", fontsize=22)
        ax.set_ylabel("Un-Normalized Flux", fontsize=22)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.tick_params(axis='y', which='major', labelsize=20)
        ax.tick_params(axis='x', which='major', labelsize=20)
        ax.tick_params(axis='both', which='major', length=10, width=1)
        ax.yaxis.get_offset_text().set_size(20)
        ax.legend(loc="upper right", fontsize=18)
        fig.savefig(f"CHIRON_Spectra/StarSpectra/Plots/Blaze_Function/{star_name}/{star_name}_{obs_date}_Order{order}.pdf",
                    bbox_inches="tight", dpi=300)

    return continuum_fit, mask


def gaussian_fit(x, mu, sigma, a):
    return a*np.exp(-0.5*((x-mu)**2)/(sigma**2))

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
    g1 = np.exp(-0.5 * ((v + sep/2) / sigma) ** 2)
    g2 = -np.exp(-0.5 * ((v - sep / 2) / sigma) ** 2)
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

def find_zero_crossing_nearest_zero(x, y, print_flag=False):
    """
    Finds the crossing point of a function where the sign changes from negative to positive, or vice versa, closest to
    x=0.

    Parameters:
        x: x values of function
        y: y values of function
        print_flag: Flag to check if the function should print out the zero crossings

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
    if print_flag:
        print([round(float(zero_crossings[i]), 3)
               for i in range(len(zero_crossings))
               if -100 < zero_crossings[i] < 100])
    return min(zero_crossings, key=lambda v: abs(v))

def find_all_crossings(x, y, level):
    """
    Return all x-locations where y crosses 'level'.
    Linear interpolation is used between points.
    """
    crossings = []
    s = y - level
    idx = np.where(np.diff(np.sign(s)) != 0)[0]

    for i in idx:
        x0, x1 = x[i], x[i+1]
        y0, y1 = s[i], s[i+1]
        xc = x0 - y0*(x1 - x0)/(y1 - y0)
        crossings.append(xc)

    return np.array(crossings)

def estimate_sep_from_25pct(wavelengths, fluxes, line_center):
    """
    Returns Gaussian separation SEP based on the two *outermost* crossings of the 25% flux level.
    """
    # define 25% level between min and max
    fmin, fmax = np.min(fluxes), np.max(fluxes)
    f25 = fmin + 0.25*(fmax - fmin)

    # find all crossings
    crossings = find_all_crossings(wavelengths, fluxes, f25)
    if len(crossings) < 2:
        return None  # cannot determine sep

    left, right = crossings[0], crossings[-1]

    # convert to velocities
    v = velocity_grid(crossings, line_center)
    sep = abs(v[-1] - v[0])

    return sep

def shafter_bisector_velocity(wavelengths, fluxes, line_center=6562.8, sep=500, v_window=5000, v_step=2.6, print_flag=False):
    """
    Finds the bisector velocity of the wings of the H Alpha line in a 1D spectrum using the oppositely signed double
    Gaussian cross-correlation method of Shafter, A. W., Szkody, P., & Thorstensen, J. R. 1986, ApJ, 308, 765.

    Parameters:
        wavelengths: array of input wavelengths
        fluxes: array of input fluxes
        line_center: rest wavelength of H Alpha (6562.8 Å)
        sep: separation of the Gaussian pair
        v_window: ± velocity range to explore (in units of km/s)
        v_step: velocity step size (in units of km/s)
        print_flag: Flag to check if the function should print out the zero crossings
    Returns:
        bisector_velocity: velocity at zero crossing closest to 0 (in km/s)
        v_grid: velocity shift grid
        ccf: cross-correlation function values

    """
    if sep is None:
        sep = estimate_sep_from_25pct(wavelengths, fluxes, line_center)
        if sep is None:
            raise ValueError("Could not determine Gaussian separation from 25% crossings.")

    sigma = sep/(7*np.sqrt(2*np.log(2)))

    v_obs = velocity_grid(wavelengths, line_center)
    interp_flux = interp1d(v_obs, fluxes, kind="linear", bounds_error=False, fill_value=1)

    v_grid = np.arange(-v_window, v_window+v_step, v_step)
    flux_grid = interp_flux(v_grid)
    flux_grid -= np.median(flux_grid)

    kernel = gaussian_pair(v_grid, sep=sep, sigma=sigma)
    ccf = np.correlate(flux_grid, kernel, mode="same")

    ccf = (ccf-np.median(ccf)) / np.std(ccf)
    bisector_velocity = find_zero_crossing_nearest_zero(v_grid, ccf, print_flag)

    return bisector_velocity, v_grid, ccf, sigma


def monte_carlo_bisector_error(
    wavelengths, fluxes,
    shafter_func,       # your shafter_bisector_velocity function
    n_trials=500,
    sigma_flux=None,    # scalar or array; if None, estimate via estimate_continuum_sigma
    rng=None,
    **shafter_kwargs
):
    """
    Monte Carlo error on the bisector velocity.

    Parameters
    ----------
    wavelengths, fluxes : arrays
    shafter_func : callable
        Function returning (v_bis, v_grid, ccf)
    n_trials : int
    sigma_flux : float or array or None
        Per-pixel 1σ noise. If None, estimated via estimate_continuum_sigma (scalar).
    rng : np.random.Generator or None
    **shafter_kwargs : passed to shafter_func

    Returns
    -------
    v_mean, v_std, v_all : float, float, array
    """

    if rng is None:
        rng = np.random.default_rng()

    if np.isscalar(sigma_flux):
        sigma_arr = np.full_like(fluxes, float(sigma_flux))
    else:
        sigma_arr = np.asarray(sigma_flux, dtype=float)

    v_list = []
    for _ in range(n_trials):
        noisy = fluxes + rng.normal(0.0, sigma_arr)
        v_bis, _, _ = shafter_func(wavelengths, noisy, **shafter_kwargs)
        if v_bis is not None and np.isfinite(v_bis):
            v_list.append(v_bis)

    v_all = np.array(v_list)
    if len(v_all) == 0:
        return np.nan, np.nan, v_all

    return float(np.mean(v_all)), float(np.std(v_all, ddof=1)), v_all


def clean_star_name(raw_name: str) -> str:
    """
    Cleans up the star name from a FITS file header to make it SIMBAD-friendly:
      - Removes trailing suffixes like "_2", "_A", etc.
      - Normalizes spacing and removes dashes
      - Keeps only the HD/HR/HIP number (discarding extra designations)
      - Ensures the catalog prefix is uppercase
    """
    name = raw_name.strip()

    # 1. Remove trailing suffixes like "_2", "_A", etc.
    name = re.sub(r'_[A-Za-z0-9]+$', '', name)

    # 2. Capture only the first occurrence of HD/HR/HIP + number
    match = re.search(r'\b(HD|HR|HIP)[-\s]?(\d+)', name, re.IGNORECASE)
    if match:
        # Normalize prefix to uppercase and ensure space between prefix and number
        prefix = match.group(1).upper()
        number = match.group(2)
        name = f"{prefix} {number}"
    else:
        # If no HD/HR/HIP prefix found, just clean minor formatting
        name = re.sub(r'[-_]+', ' ', name)

    return name

def clean_star_name_2(raw_name: str) -> str:
    """
    Cleans up the star name from a FITS file header to make it SIMBAD-friendly:
      - Remove trailing suffixes like "_2", "_A", etc.
      - Normalize spacing
      - Remove dashes between catalog prefixes (HD/HR/HIP) and numbers
      - Keep only the HD/HR/HIP number, discarding extra designations
    """
    name = raw_name.strip()

    # 1. Remove trailing suffixes like "_2", "_A", etc.
    name = re.sub(r'_[A-Za-z0-9]+$', '', name)

    # 2. Capture only the first occurrence of HD/HR/HIP + number
    match = re.search(r'\b(HD|HR|HIP)[-\s]?(\d+)', name, re.IGNORECASE)
    if match:
        # Keep just "HD 12345" (or HR/HIP)
        name = f"{match.group(1).upper()} {match.group(2)}"
    else:
        # If no match, leave it as-is (after suffix cleanup)
        return name

    return name

# ============================================================
# Greek mapping: (regex pattern) → Greek letter
# ============================================================
GREEK_MAP = {
    r"(alpha|alp|alf)": "α",
    r"(beta|bet)": "β",
    r"(gamma|gam)": "γ",
    r"(delta|del)": "δ",
    r"(epsilon|eps)": "ε",
    r"(zeta|zet)": "ζ",
    r"(eta)": "η",
    r"(theta|the|tet)": "θ",
    r"(iota|iot)": "ι",
    r"(kappa|kap)": "κ",
    r"(lambda|lam)": "λ",
    r"(mu)": "μ",
    r"(nu)": "ν",
    r"(xi)": "ξ",
    r"(omicron|omi)": "ο",
    r"(pi)": "π",
    r"(rho)": "ρ",
    r"(sigma|sig)": "σ",
    r"(tau|ta)": "τ",
    r"(upsilon|ups)": "υ",
    r"(phi|phe)": "φ",
    r"(chi|che)": "χ",
    r"(psi|ps)": "ψ",
    r"(omega|ome)": "ω",
}

# Superscript map for numbered Bayer designations
SUP = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}
def to_superscript(num: str) -> str:
    """Convert digits to superscript Unicode."""
    return "".join(SUP.get(ch, ch) for ch in num)


# ============================================================
# Greek handling
# ============================================================
def convert_greek_with_index(name: str) -> str:
    """
    Convert forms like 'kappa01 Aps' → 'κ¹ Aps'
    Must run BEFORE plain Greek conversion.
    """
    for pat, greek in GREEK_MAP.items():
        # Look for Greek token + digits at start, e.g., kappa01
        m = re.match(rf"^{pat}(\d{{1,2}})\b", name, flags=re.IGNORECASE)
        if m:
            idx = m.group(2) if m.lastindex >= 2 else None
            if idx:
                sup = to_superscript(idx.lstrip("0"))
                return re.sub(
                    rf"^{pat}(\d{{1,2}})",
                    greek + sup,
                    name,
                    flags=re.IGNORECASE,
                )
    return name


def convert_greek_plain(name: str) -> str:
    """
    Convert simple spellings like 'alpha Per' → 'α Per'
    Must run AFTER convert_greek_with_index().
    """
    for pat, greek in GREEK_MAP.items():
        name = re.sub(rf"^{pat}", greek, name, flags=re.IGNORECASE)
    return name


# ============================================================
# Main function
# ============================================================
def clean_star_name3(raw_name: str) -> str:
    """
    Clean star name to SIMBAD-friendly form.

    Rules:
    - Remove trailing suffixes (_A, _2, etc.)
    - If HD/HIP/HR + number is present → return only that (truncate rest)
    - Convert bayer names → Greek letters (e.g. 'alpha Per' → 'α Per')
    - Support Bayer + index (e.g. 'kappa01 Aps' → 'κ¹ Aps')
    - Remove dashes
    - Normalize whitespace
    """
    name = raw_name.strip()

    # 1) Remove trailing suffix: _A, _2, etc.
    name = re.sub(r"_[A-Za-z0-9]+$", "", name)

    # 2) Detect HD/HR/HIP + number; keep only this
    m = re.search(r"\b(HD|HR|HIP)[-\s]?(\d+)", name, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"

    # If no HD/HR/HIP, continue into Greek handling

    # 3) Remove dashes between tokens
    name = name.replace("-", " ")

    # 4) Bayer with enumeration first
    name = convert_greek_with_index(name)

    # 5) Plain Greek token second
    name = convert_greek_plain(name)

    # 6) Normalize whitespace
    name = re.sub(r"\s+", " ", name)

    return name.strip()

def robust_A_topN(fluxes, N=5):
    """
    Return robust amplitude A (median of top N pixels) and the indices used.
    Assumes fluxes are continuum-normalized to 1 (not continuum-subtracted).
    """
    arr = np.array(fluxes) - 1.0  # work in continuum-subtracted units
    # handle case N > number of pixels
    Nuse = min(N, arr.size)
    top_idxs = np.argsort(arr)[-Nuse:]
    A_topN = np.median(arr[top_idxs])
    return A_topN, top_idxs

def estimate_A_err_by_noise_injection(fluxes, sig_cont, N=5, M=500, rng=None):
    """
    Estimate uncertainty of amplitude estimator (median(top N)) by adding white noise M times.
    Returns (A_median, A_err).
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = np.array(fluxes) - 1.0
    A_samps = np.empty(M)
    for i in range(M):
        noisy = arr + rng.normal(0.0, sig_cont, size=arr.size)
        A_samps[i] = np.median(noisy[np.argsort(noisy)[-min(N, arr.size):]])
    return np.median(A_samps), np.std(A_samps, ddof=1)

def analytic_sigma_v_mc_from_nonparam(wavs, fluxes,
                                      gaussian_width_kms,
                                      p=0.25,
                                      Ntop=5,
                                      M_inject=500,
                                      MC_samples=2000,
                                      cont_windows=None,
                                      rng_seed=123):
    """
    Combine robust amplitude estimate + noise injection + analytic MC for sigma_v.
    - wavs: wavelength array (Å)
    - fluxes: flux array (continuum-normalized to 1)
    - gaussian_width_kms: sigma (in km/s) used in analytic formula (your gaussian_width)
    - cont_windows: list of (min,max) tuples to use to measure continuum RMS; if None, defaults used relative to Halpha
    Returns dictionary with MC distribution and summary.
    """
    rng = np.random.default_rng(rng_seed)
    lam0 = 6562.8

    # 1) continuum windows default if not provided
    if cont_windows is None:
        cont_windows = [(lam0 + 50, lam0 + 200), (lam0 - 200, lam0 - 50)]
    cont_mask = np.zeros_like(wavs, dtype=bool)
    for (a, b) in cont_windows:
        cont_mask |= ((wavs > a) & (wavs < b))

    # robust continuum RMS (in continuum-subtracted units)
    cont_vals = (np.array(fluxes)[cont_mask] - 1.0)
    _, _, sig_cont = sigma_clipped_stats(cont_vals, sigma=3.0, maxiters=5)
    # fall back if empty
    if np.isnan(sig_cont) or sig_cont <= 0:
        sig_cont = np.std(cont_vals) if cont_vals.size > 0 else 0.01

    # 2) robust A and estimate A_err via noise injection
    A0, topidxs = robust_A_topN(fluxes, N=Ntop)
    A_med, A_err = estimate_A_err_by_noise_injection(fluxes, sig_cont, N=Ntop, M=M_inject, rng=rng)

    # if A is extremely small or negative, guard
    A0 = max(A0, 1e-8)
    A_med = max(A_med, 1e-8)
    A_err = max(A_err, 1e-8)

    # 3) estimate uncertainty on sigma_g (if you have sep uncertainty, use it; otherwise assume small relative error)
    # simple conservative choice: 5% relative uncertainty on gaussian_width (tweak as needed)
    sigma_g_kms = float(gaussian_width_kms)
    sigma_g_err = 0.05 * sigma_g_kms

    # 4) analytic MC: sample sig_cont, A, sigma_g
    # choose sig_cont uncertainty: you can use sqrt(2/Npix) or a fraction. Use 10% as conservative baseline.
    sig_cont_err = 0.10 * sig_cont

    # draw samples
    sig_cont_samps = rng.normal(sig_cont, max(sig_cont_err, 1e-12), size=MC_samples)
    A_samps = rng.normal(A_med, max(A_err, 1e-12), size=MC_samples)
    A_samps = np.clip(A_samps, 1e-12, None)
    sigma_g_samps = rng.normal(sigma_g_kms, max(sigma_g_err, 1e-12), size=MC_samples)
    sigma_g_samps = np.clip(sigma_g_samps, 1e-12, None)

    sqrtterm = np.sqrt(-2.0 * np.log(p))
    sigma_v_samps = (sig_cont_samps * sigma_g_samps) / (A_samps * p * sqrtterm)

    # summary
    med = np.median(sigma_v_samps)
    lo68, hi68 = np.percentile(sigma_v_samps, [16, 84])
    lo95, hi95 = np.percentile(sigma_v_samps, [2.5, 97.5])

    return {
        "sigma_v_samps": sigma_v_samps,
        "sigma_v_median": med,
        "sigma_v_68": (lo68, hi68),
        "sigma_v_95": (lo95, hi95),
        "A0": A0,
        "A_med": A_med,
        "A_err": A_err,
        "sig_cont": sig_cont,
        "sigma_g_kms": sigma_g_kms
        }

def wav_corr(wav, bar_vel, rv_vel):
    """
    Correct a stellar spectrum to the barycentric velocity of the solar system, adapted from the original by Sebastián
    Carrazco Gaxiola.

    :param wav: The "raw" wavelength grid
    :param bar_vel: The barycentric velocity
    :param rv_vel: The radial velocity of the star
    :return: Corrected wavelength grid, coefficient of correction
    """

    c = 299792.4580  # km/s

    vel = float(bar_vel) - float(rv_vel)
    corr_coef = ((c - vel) / c)

    dlamb = wav / np.array(corr_coef)

    return dlamb, corr_coef


def make_vel_wav_transforms(rest_wavelength):
    """
    Returns (wav_to_vel, vel_to_wav) functions for a given rest wavelength.
    rest_wavelength in Angstroms.
    """

    c = 3e5  # km/s

    def wav_to_vel(w):
        return ( (w - rest_wavelength) / rest_wavelength ) * c

    def vel_to_wav(v):
        return v / c * rest_wavelength + rest_wavelength

    return wav_to_vel, vel_to_wav


def barycentric_correct(wav, bar_vel):
    """
    Shift an entire spectrum from observatory frame to the barycentric frame.

    :param wav: Observed wavelength grid
    :param bar_vel: Barycentric velocity in km/s (positive = observatory moving away from target)

    :return: wav_corr (Barycentric-corrected wavelengths) and coef (Applied scale factor)
    """

    c = 299792.4580  # km/s

    coef = 1.0 - (bar_vel / c)

    wav_corrected = wav / coef

    return wav_corrected

def stellar_rest_frame(barycentric_wav, stellar_rv):
    """
       Shift spectrum from barycentric frame to stellar rest frame.

       :param barycentric_wav: Barycentric-corrected wavelength grid
       :param stellar_rv: Stellar radial velocity in km/s (positive = receding)

       :return: wav_rest
   """
    c = 299792.4580  # km/s

    coef = 1 + (stellar_rv / c)

    wav_rest = barycentric_wav / coef

    return wav_rest


def pixel_velocity_finder(vel: float):
    c = 299792.4580  # km/s
    pixel_size = vel / (c*np.log(10))

    return pixel_size

def log_wavelength_grid(start_wav, end_wav, pixel_velocity):
    """
       Create a linear log-wavelength grid that has a uniform velocity spacing

       :param start_wav: The starting wavelength (in Å)
       :param end_wav: The ending wavelength (in Å)
       :param pixel_velocity: The desired pixel-velocity spacing (in km/s)

       :return: A log-wavelength grid that has a uniform velocity spacing
   """
    x_arr = np.arange(np.log(start_wav), np.log(end_wav), pixel_velocity / 3e5)
    return x_arr

def der_snr(flux):
    """
    This function computes the signal-to-noise ratio (SNR) using the DER_SNR algorithm following the definition set
    forth by the Spectral Container Working Group of ST-ECF, MAST and CADC (https://spektroskopie.vdsastro.de/files/pdfs/snr.pdf).

    :param flux: The flux vector of the spectrum for which the SNR is to be determined
    :return: The SNR
    """
    from numpy import array, where, median, abs

    flux = array(flux)

    # Values that are exactly zero (padded) are skipped
    flux = array(flux[where(flux != 0.0)])
    n = len(flux)

    # For spectra shorter than this, no value can be returned
    if n > 4:
        signal = median(flux)

        noise = 0.6052697 * median(abs(2.0 * flux[2:n - 2] - flux[0:n - 4] - flux[4:n]))

        return float(signal / noise)

    else:

        return 0.0

if __name__ == '__main__':
    print(pixel_velocity_finder(2.6))
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
    # plt.rcParams['font.family'] = 'Trebuchet MS'
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
