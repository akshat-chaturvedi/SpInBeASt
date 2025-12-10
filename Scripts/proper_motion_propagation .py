import numpy as np
from coordTransfer import coord_transfer

def epoch_convert(ra, dec, epoch_input, epoch_output, pmra, pmdec):
    """
    Calculates the new coordinates at a desired epoch
    given the proper motion and its initial coordinate in a given epoch

    :param ra: (ndarray) Right Ascension (in degrees) of external objects

    :param dec: (ndarray) Declination (in degrees) of external objects

    :param epoch_input: (float) Epoch of the input coordinates in years. Example: 2000

    :param epoch_output: (float) Epoch of the output coordinates in years. Example: 2015.5

    :param pmra: (float) Proper Motion in Right Ascension (in mas/yr)

    :param pmdec: (float) Proper Motion in Declination of Gaia objects (in mas/yr)

    """

    ra_x = ra + 2.7778e-7 * (epoch_output - epoch_input) * pmra / np.cos(dec / 57.296)
    dec_x = dec + 2.7778e-7 * (epoch_output - epoch_input) * pmdec

    return ra_x, dec_x


star_ra_j2000, star_dec_j2000 = coord_transfer("05 21 16.8603281544", "-34 20 42.173559132")
star_ra_j2016, star_dec_j2016 = epoch_convert(star_ra_j2000, star_dec_j2000, 2000, 2016,
                                              -0.103, 7.591)
print(f"J2000 RA={star_ra_j2000:.5f}, Dec={star_dec_j2000:.5f}")
print(f"J2016 RA={star_ra_j2016:.5f}, Dec={star_dec_j2016:.5f}")