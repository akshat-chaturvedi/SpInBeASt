import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tomographyHelperFunctions import auto_rectify, readsbcm
from astropy.io import fits, ascii
import pandas as pd
from tqdm import tqdm
import sys


def tomography(star_name: str, n_iters: int, delta: float | int, obs_spec, rec_spec, pixel_shift, ratios):
    """
    Iterative Least Squares Technique (ILST) tomography algorithm based on Dr. Rafael Wiemker's master's thesis. Works
    in the case of two hot stars observed in the Rayleigh-Jeans part of the spectrum

    :param star_name: The name of the star
    :param n_iters: The number of iterations to run
    :param delta: The convergence acceleration factor (0 to 1)
    :param obs_spec: The observed composite spectra, as a spectral stack
    :param rec_spec: The reconstructed spectra (current estimate)
    :param pixel_shift: The pixel shifts for each spectrum, component
    :param ratios: The flux ratio for each spectrum, component

    :returns: The reconstructed spectra (revised estimate) and the composite spectra for same shifts as the observed
    spectra.
    """
    #Create weighting factors:
    int_shift_1 = np.zeros((num_spectra, num_components), dtype=int)  # Integer shifts
    int_shift_2 = np.zeros((num_spectra, num_components), dtype=int)  # Integer shifts

    frac_shift_1 = np.zeros((num_spectra, num_components), dtype=float)  # Fractional shifts
    frac_shift_2 = np.zeros((num_spectra, num_components), dtype=float)  # Fractional shifts

    weight_1 = np.zeros((num_spectra, num_components), dtype=int)  # Weights
    weight_2 = np.zeros((num_spectra, num_components), dtype=int)  # Weights

    for i in range(num_spectra):  # For each spectrum
        for j in range(num_components):  # For each component
            int_shift_1[i][j] = int(pixel_shift[i][j])
            if pixel_shift[i][j] < 0:
                int_shift_2[i][j] = int_shift_1[i][j]-1
            else:
                int_shift_2[i][j] = int_shift_1[i][j]+1

            frac_shift_2[i][j] = abs(pixel_shift[i][j]-int_shift_1[i][j])
            frac_shift_1 = 1-frac_shift_2[i][j]

        weight_sum = sum(frac_shift_1[i,:]**2 + frac_shift_2[i,:]**2)
        weight_1[i][0] = delta/ratios[i,:] * frac_shift_1[i,:]/weight_sum
        weight_2[i][0] = delta/ratios[i,:] * frac_shift_2[i,:]/weight_sum

        frac_shift_1[i][0] = ratios[i,:]*frac_shift_1[i,:]
        frac_shift_2[i][0] = ratios[i,:]*frac_shift_2[i,:]

    # Main iteration
    npp = num_pixels+100
    rs = np.ones((num_components, npp), dtype=float)
    cs = np.zeros((num_components, npp), dtype=float)
    resid = np.zeros((num_spectra, npp), dtype=float)
    cr = np.zeros((num_spectra, npp), dtype=float)
    top = num_pixels + 49

    for _ in tqdm(range(n_iters), colour='#8e82fe', file=sys.stdout, desc=f"ILST Tomography Algorithm for {star_name}"):
        for k in range(num_components):
            rs[50:50+num_pixels, k] = rec_spec[:, k]
        sim_spec = 0*obs_spec
        for j in range(num_spectra):
            for k in range(num_components):
                cs[k][0] = (frac_shift_1[j][k] * np.roll(resid[k,:], -int_shift_1[k][j]) +
                            frac_shift_2[j][k] * np.roll(resid[k,:], -int_shift_2[k][j]))
            css = cs.sum(axis=1, keepdims=True)
            sim_spec[j][0] = css[50:top]
            # Calculate residuals
            resid[j][50] = obs_spec[j,:] - sim_spec[j,:]

        # Calculate corrections matrix
        for j in range(num_components):  # For each component
            # Get corrections from each spectrum
            for k in range(num_spectra):
                cr[k][0] = (weight_1[k][j]*np.roll(resid[k,:], -int_shift_1[k][j]) +
                            weight_2[k][j]*np.roll(resid[k,:], -int_shift_2[k][j]))

            rec_spec[j][0] = rec_spec[j] + cr[50:top+1,:].mean(axis=1, keepdims=True)

        # Enforce fluxes > 0
        rec_spec = rec_spec[rec_spec > 0]

    # Final matrix of simulated spectra
    sim_spec = 0*obs_spec
    for k in range(num_components):
        rs[50:50+num_pixels, k] = rec_spec[:, k]
    for j in range(num_spectra):  # Compute each spectrum by adding contributions from each component
        for k in range(num_components):
            cs[k][0] = (frac_shift_1[j][k] * np.roll(rs[k,:], int_shift_1[j][k]) +
                        frac_shift_2[j][k] * np.roll(rs[k,:], int_shift_2[j][k]))
        css = cs.sum(axis=1, keepdims=True)
        sim_spec[j][0] = css[50:top+1,:]

    return rec_spec, sim_spec


def tomography_r_lambda(star_name: str, n_iters: int, delta: float | int, obs_spec, rec_spec, pixel_shift, ratios):
    """
    Iterative Least Squares Technique (ILST) tomography algorithm based on Dr. Rafael Wiemker's master's thesis. This
    version uses wavelength variable flux ratios in the case where the components are quite different in temperature, so
    their relative spectral contributions vary significantly across the observed spectrum.

    :param star_name: The name of the star
    :param n_iters: The number of iterations to run
    :param delta: The convergence acceleration factor (0 to 1)
    :param obs_spec: The observed composite spectra, as a spectral stack
    :param rec_spec: The reconstructed spectra (current estimate)
    :param pixel_shift: The pixel shifts for each spectrum, component
    :param ratios: The flux ratio for each spectrum, component

    :returns: The reconstructed spectra (revised estimate) and the composite spectra for same shifts as the observed
    spectra.
    """
    # Create weighting factors:
    int_shift_1 = np.zeros((num_spectra, num_components), dtype=int)  # Integer shifts
    int_shift_2 = np.zeros((num_spectra, num_components), dtype=int)  # Integer shifts

    frac_shift_1 = np.zeros((num_components, num_pixels, num_spectra), dtype=float)  # Fractional shifts
    frac_shift_2 = np.zeros((num_components, num_pixels, num_spectra), dtype=float)  # Fractional shifts

    weight_1 = np.zeros((num_components, num_pixels, num_spectra), dtype=float)  # Weights
    weight_2 = np.zeros((num_components, num_pixels, num_spectra), dtype=float)  # Weights

    for j in range(num_spectra):  # For each spectrum
        for k in range(num_components):  # For each component
            int_shift_1[j][k] = int(pixel_shift[j][k])
            if pixel_shift[j][k] < 0:
                int_shift_2[j][k] = int_shift_1[j][k] - 1
            else:
                int_shift_2[j][k] = int_shift_1[j][k] + 1

            frac_shift_2[k,:,j] = abs(pixel_shift[j][k] - int_shift_1[j][k])
            frac_shift_1[k,:,j] = 1 - frac_shift_2[k,:,j]
        weight_sum = sum(frac_shift_1[0,:,j] ** 2 + frac_shift_2[0,:,j] ** 2)
        for k in range(num_components):
            for i in range(num_pixels):
                weight_1[k,i,j] = delta/ratios[k,i,j] * frac_shift_1[k,i,j] / weight_sum
                weight_2[k,i,j] = delta / ratios[k,i,j] * frac_shift_2[k,i,j] / weight_sum
                frac_shift_1[k,i,j] = ratios[k,i,j] * frac_shift_1[k,i,j]
                frac_shift_2[k,i,j] = ratios[k,i,j] * frac_shift_2[k,i,j]

    # Main iteration
    npp = num_pixels + 100
    rs = np.ones((num_components, npp))  # (nc, npp)
    cs = np.zeros((num_components, npp))  # (nc, npp)
    resid = np.zeros((num_spectra, npp))  # (ns, npp)
    cr = np.zeros((npp, num_spectra))  # (ns, npp)

    top = num_pixels + 49
    sample = np.arange(num_pixels)

    # Get extrapolated versions of frac_shift_1, frac_shift_2, weight_1, and weight_2
    xpgrid = np.arange(num_pixels)+50
    xppgrid = np.arange(npp)
    frac_shift_1_e = np.zeros((num_components, npp, num_spectra), dtype=float)
    frac_shift_2_e = np.zeros((num_components, npp, num_spectra), dtype=float)
    weight_1_e = np.zeros((num_components, npp, num_spectra), dtype=float)
    weight_2_e = np.zeros((num_components, npp, num_spectra), dtype=float)
    for j in range(num_spectra):
        for k in range(num_components):
            f = RegularGridInterpolator(
                (xpgrid,), frac_shift_1[k, :, j], bounds_error=False, fill_value=0
            )
            frac_shift_1_e[k, :, j] = f(xppgrid)

            f = RegularGridInterpolator(
                (xpgrid,), frac_shift_2[k, :, j], bounds_error=False, fill_value=0
            )
            frac_shift_2_e[k, :, j] = f(xppgrid)

            f = RegularGridInterpolator(
                (xpgrid,), weight_1[k, :, j], bounds_error=False, fill_value=0
            )
            weight_1_e[k, :, j] = f(xppgrid)

            f = RegularGridInterpolator(
                (xpgrid,), weight_2[k, :, j], bounds_error=False, fill_value=0
            )
            weight_2_e[k, :, j] = f(xppgrid)

    for _ in tqdm(range(n_iters), colour='#8e82fe', file=sys.stdout, desc=f"ILST Tomography Algorithm for {star_name}"):
        for k in range(num_components):
            rs[k, 50:50+num_pixels] = rec_spec[k, :]

        # Form matrix of simulated spectra
        sim_spec = np.zeros((num_spectra, npp))
        for j in range(num_spectra):
            for k in range(num_components):
                cs[k, :] = (
                        np.roll(frac_shift_1_e[k, :, j] * rs[k, :], int_shift_1[j][k])
                        + np.roll(frac_shift_2_e[k, :, j] * rs[k, :], int_shift_2[j][k])
                )
            css = cs.sum(axis=0)
            sim_spec[j, 50:50 + num_pixels] = css[50:50 + num_pixels]
            resid[j, 50:50 + num_pixels] = obs_spec[j, :] - sim_spec[j, 50:50 + num_pixels]


        # Calculate corrections matrix
        for k in range(num_components):
            cr_total = np.zeros(num_pixels)
            for j in range(num_spectra):
                rolled_1 = np.roll(weight_1_e[k, 50:50 + num_pixels, j] * resid[j, 50:50 + num_pixels],
                                   -int_shift_1[j, k])
                rolled_2 = np.roll(weight_2_e[k, 50:50 + num_pixels, j] * resid[j, 50:50 + num_pixels],
                                   -int_shift_2[j, k])
                cr_total += rolled_1 + rolled_2

            # Update rec_spec in-place
            rec_spec[k, :] += cr_total / num_spectra

        # Enforce fluxes > 0
        rec_spec_masked = np.where(rec_spec > 0, rec_spec, 0)
        for k in range(num_components):
            rec_spec_masked[k, :] = auto_rectify(rec_spec_masked[k, :], sample, 4, 10, 1, 2)

    # Final matrix of simulated spectra
    sim_spec = np.zeros((num_spectra, npp))
    for k in range(num_components):
        rs[k, 50:50 + num_pixels] = rec_spec[k, :]

    for j in range(num_spectra):
        for k in range(num_components):
            cs[k, :] = (
                    np.roll(frac_shift_1_e[k, :, j] * rs[k, :], int_shift_1[j, k])
                    + np.roll(frac_shift_2_e[k, :, j] * rs[k, :], int_shift_2[j, k])
            )
        css = cs.sum(axis=0)
        sim_spec[j, 50:top] = css[50:top]

    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(rec_spec[0][50:-50] + 0.3, c="k", label="Primary")
    ax.plot(rec_spec[1][50:-50], c="r", label="Secondary")
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='both', which='major', length=10, width=1)
    ax.set_xlabel(r"Pixel", fontsize=22)
    ax.set_ylabel("Relative Flux", fontsize=22)
    ax.set_title(f"Reconstructed Spectrum for {star}", fontsize=24)
    ax.legend(loc="lower left", fontsize=20)
    fig.savefig(f"{star}_Reconstructed_Spectrum.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    hdu = fits.PrimaryHDU(data=rec_spec)
    hdul = fits.HDUList([hdu])

    hdul.writeto(f"{star}_Reconstructed_Spectrum.fits", overwrite=True)

    return rec_spec, sim_spec


if __name__ == '__main__':
    star = 'RY Per'
    hjd = ascii.read("../Tomography/ry_per_hjd.txt")
    wavs = pd.read_fwf("../Tomography/ry_per.dat", header=None)
    wavs = pd.concat([wavs[0], wavs[1], wavs[2], wavs[3]], ignore_index=True)
    wavs = np.array(sorted(wavs[np.logical_not(np.isnan(wavs))]))

    g = [13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36]

    with fits.open("../Tomography/ry_per.fits") as hdul:
        observed_spec = hdul[0].data

    observed_spec = observed_spec[g]
    hjd=hjd[g]

    num_pixels = len(wavs) # Number of pixels
    num_components = 2 # Number of components
    num_spectra = len(hjd) # Number of spectra

    p_shift = np.zeros((num_spectra, num_components), dtype=float)
    ratio = np.ones((num_components, num_pixels, num_spectra), dtype=float)

    # Don't actually need this readsbcm, only thing we need is the calculated velocities, can be done with pandas/ascii
    d2, *_ = readsbcm('../Tomography/sbcm.out.sec')
    d2 = d2[g,:]
    d_vel = 21.120012  # Pixel step in km/s for these spectra

    # Set p_shift from calculated orbits. ILST needs the orbital velocities from the outset. Here we read in the
    # calculated radial velocities for the secondary in each spectrum
    for i in range(num_spectra):
        p_shift[i][1] = d2[i][2]/d_vel

    # For RY Per, the secondary's radial velocity curve is better established than the primary's one, so the
    # corresponding velocities of the primary are calculated from the orbital elements

    # Get these values from orbital solutions
    v0 = -5.9696  # Systemic velocity [km/s]
    k1 = 47.3509  # Primary semi-amplitude [km/s]
    k2 = 174.4652 # Secondary semi-amplitude [km/s]

    for i in range(num_spectra):
        p_shift[i][0] = (-1.0*k1/k2*(d2[i][2]-v0)+v0)/d_vel

    # Need to establish the flux contributions from the components (normalized so that they sum to unity).  In the case
    # of two hot stars observed in the Rayleigh-Jeans part of the spectrum, one number will suffice for this ratio.
    # However, for RY Per, the components are quite different in temperature, so their relative spectral contributions
    # vary significantly across the observed spectrum. The version used below, tomography_r_lambda, handles this more
	# complicated case.  Here we estimated the ratio from Kurucz flux models and stellar radii estimates for each
    # wavelength  point in the spectrum (stored in ratio1999m.fits).

    with fits.open("../Tomography/ratio1999m.fits") as hdul:
        r = hdul[0].data

    for i in range(num_pixels):
        ratio[0,i,:] = 1/(1+r[i])
        ratio[1,i,:] = r[i]/(1+r[i])

    # Read in ism spectrum and remove from data. In many cases, the spectra will also contain lines formed in the
    # interstellar medium that do not participate in the orbital motion of the stars.  The best approach is to remove
    # these by division prior to spectral reconstruction. The file ismout.fits is an extracted ISM spectrum.

    with fits.open("../Tomography/ismout.fits") as hdul:
        ism = hdul[0].data

    for i in range(num_spectra):
        observed_spec[i, :] = observed_spec[i, :] / ism

    recon_spec = np.ones((num_components, num_pixels), dtype=float)

    # Advice:
	# The algorithm is ready to run.  The first and second parameters are the number of iterations and gain for each
    # iteration, respectively. Details are given in Bagnuolo et al. 1994, ApJ, 423, 446. Here we use tomography_r_lambda
    # that accounts for a wavelength dependency in the flux ratio. Note that the algorithm may introduce low frequency
    # oscillations in the results that may need to be divided out later (depending upon the application).

    a, b = tomography_r_lambda(star, 20_000, 0.8, observed_spec, recon_spec, p_shift, ratio)
