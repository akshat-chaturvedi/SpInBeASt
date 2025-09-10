import numpy as np

def auto_sample_points(s, n_bins: int = 50, percentile: float | int = 90):
    """
    Automatically pick candidate continuum points.

    :param s: The spectrum flux values
    :param n_bins: Number of bins to divide the spectrum into
    :param percentile: The percentile of each bin to select as candidate continuum points

    :returns: Pixel indices of the candidate continuum points
    """
    n = len(s)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    sample = []

    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if end > start:
            chunk = s[start:end]
            # pick the index of the value closest to the percentile
            pval = np.percentile(chunk, percentile)
            idx = np.argmin(np.abs(chunk - pval))
            sample.append(start + idx)

    return np.array(sample)


def auto_rectify(s, sample, n_order=3, n_iter=5, s_low=3.0, s_high=3.0):
    """
    Automatic rectification by iterative rejection of non-continuum features

    :param s: The spectrum flux values
    :param sample: Indices of spectrum points to use as initial “continuum candidates”
    :param n_order: The order of polynomial to fit
    :param n_iter: The number of rejection iterations
    :param s_low: Low sigma-clipping threshold
    :param s_high: High sigma-clipping threshold

    :returns: Rectified spectrum
    """
    n = len(s)
    x = np.arange(n)
    t = s[sample]
    x_fit = x[sample]

    for _ in range(n_iter):
        c = np.polyfit(x_fit, t, n_order)
        fit = np.polyval(c, x_fit)
        res = t - fit
        sig = res.std()
        mask = (res < s_high*sig) & (res > -s_low*sig)
        if mask.sum() > n_order:
            x_fit = x_fit[mask]
            t = t[mask]

    fit_full = np.polyval(c, x)
    return s / fit_full


def readsbcm(file):
    """
    Read a BCM-style file and replicate IDL readsbcm behavior.

    :returns:
        data : np.ndarray of shape (7, num)
        phw, vow, goodw : wrapped phase, velocity, and weight arrays
        pcw, vcw : wrapped calibration arrays
    """
    # Read the main data table, skipping the header
    # Assuming the file has a fixed header until the first data line
    # We'll detect the line starting with a number
    with open(file, 'r') as f:
        lines = f.readlines()

    # Skip non-data lines
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == '':
            continue
        # detect lines that look like data (start with a number)
        if stripped[0].isdigit() or stripped[0] == '.':
            data_lines.append(stripped)

    # Now parse data_lines into a float array
    num = len(data_lines)
    data = np.zeros((7, num), dtype=float)

    for i, line in enumerate(data_lines):
        parts = line.split()
        # Some files might have more or fewer columns
        if len(parts) < 7:
            raise ValueError(f"Line {i} has fewer than 7 columns")
        data[:, i] = [float(p) for p in parts[:7]]

    # Extract phase, velocity, and "goodness"
    ph = data[4, :].copy()  # PHASE
    vo = data[1, :].copy()  # V(OBS)
    good = data[6, :].copy()  # WEIGHT

    # Sort by phase
    order = np.argsort(ph)
    ph = ph[order]
    vo = vo[order]
    good = good[order]

    # Phase wrap-around
    lo = np.where(ph < 0.2)[0]
    hi = np.where(ph > 0.8)[0]

    phw = ph.copy()
    vow = vo.copy()
    goodw = good.copy()

    if len(lo) > 0:
        phw = np.concatenate([phw, ph[lo] + 1.0])
        vow = np.concatenate([vow, vo[lo]])
        goodw = np.concatenate([goodw, good[lo]])

    if len(hi) > 0:
        phw = np.concatenate([ph[hi] - 1.0, phw])
        vow = np.concatenate([vo[hi], vow])
        goodw = np.concatenate([good[hi], goodw])

    # Optional calibration arrays
    # Here we mimic the IDL code: c(0,*) = 0.01, 0.02,... 1.0
    pc = np.linspace(0.01, 1.0, 100)
    vc = np.zeros(100)  # placeholder, could be from file

    lo_pc = np.where(pc < 0.2)[0]
    hi_pc = np.where(pc > 0.8)[0]

    pcw = np.concatenate([pc[hi_pc] - 1.0, pc, pc[lo_pc] + 1.0])
    vcw = np.concatenate([vc[hi_pc], vc, vc[lo_pc]])

    return data.T, phw, vow, goodw, pcw, vcw


def shift_array(arr, n):
    if n > 0:
        return np.concatenate([np.zeros(n), arr[:-n]])
    elif n < 0:
        n = abs(n)
        return np.concatenate([arr[n:], np.zeros(n)])
    else:
        return arr

