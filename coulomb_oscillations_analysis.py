import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from matplotlib import rc


def hann_window(n):
    # Returns a Hann window for a signal with n samples
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def sliding_dft(zs, window_length, shift, xi, xf, fmin=0, fmax=np.inf):
    """
    Performs a sliding-window discrete Fourier transform of a signal.

    For each window, the data is multiplied by a Hann function in order to make the
    selected signal periodic and limit aliasing effects.

    Parameters:
    -----------
    zs: NumPy array
        Input data. Allowed shapes are (L, ) for a single trace with L datapoints
        or (N, L) for a collection of N traces with L datapoints each. The latter case
        can be used for instance to analyze scans taken at different magnetic fields
        at once. In the first case, the input array is first converted to a 2D array
        with shape (1, L).
    window_length: int
        Size of the sliding window in # of datapoints.
    shift: int
        Shift of the sliding window in # of datapoints.
    xi, xf: float
        Horizontal range of the data. Used to determine Fourier frequencies in
        the correct physical units.
    fmin, fmax: float
        Low and high frequency cutoffs for the dft data.

    Returns:
    --------
    freqs: NumPy array
        Array of frequencies at which the discrete Fourier transform is sampled,
        i.e. the output of np.fft.rfftfreq(). Range is [fmin, fmax]
    stft: NumPy array
        DFT output for each window. The output shape is (N, N_bins, F) where F
        is the number of frequencie sampled in the range [fmix, fmax]

    """
    zs = zs.reshape(1, zs.shape[0]) if len(zs.shape) == 1 else zs
    dx = abs(xf - xi) / zs.shape[1]

    # determine number of intervals
    nbins = 1 + int((zs.shape[1] - window_length) / shift)

    # Frequencies sampled
    freqs = np.fft.rfftfreq(window_length, dx)
    freq_mask = np.logical_and(freqs < fmax, freqs > fmin)
    clipped_freqs = freqs[freq_mask]

    # Returns the whole DFT for each bin, if requested.
    sdft = np.zeros((zs.shape[0], nbins, len(clipped_freqs)))

    # Applies Hann window to mitigate effect of shortening the signal
    hann = hann_window(window_length)

    for (i, z) in enumerate(zs):
        bins = np.array([z[n * shift:n * shift + window_length] for n in range(nbins)])
        for (j, b) in enumerate(bins):
            # b -= np.average(b)
            Fb = np.abs(np.fft.rfft(hann * b, norm='ortho'))
            Fb[freqs < fmin] = np.nan
            Fb[freqs > fmax] = np.nan
            # dfreqs[i, j] = freqs[np.nanargmax(Fb)]
            # if full_output:
            sdft[i, j] = Fb[freq_mask]
    return clipped_freqs, sdft


def dominant_frequencies(freqs, sdft):
    dfreqs = np.zeros(sdft.shape[:2])
    for i in range(dfreqs.shape[0]):
        for j in range(dfreqs.shape[1]):
            idx_2e = np.nanargmax(sdft[i, j])
            dfreqs[i, j] = freqs[idx_2e]
    return dfreqs


def dft_spectrum_2e_weight(freqs, sdft, f0):
    idx = (np.abs(freqs - f0)).argmin()
    weights = np.zeros(sdft.shape[:2])
    for i in range(sdft.shape[0]):
        for j in range(sdft.shape[1]):
            weights[i, j] = np.sum(sdft[i, j, :idx]) - np.sum(sdft[i, j, idx:])
            weights[i, j] /= np.sum(sdft[i, j])
    # weights[weights > 0] /= np.max(weights[weights > 0])
    # weights[weights < 0] /= np.max(np.abs(weights[weights < 0]))
    return weights


# Function written by Marcos Duarte
# http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def peak_data(zs, xi, xf, mph, md):
    """
    Finds peak positions and spacings in a signal.

    Parameters:
    -----------
    zs: NumPy array
        Input data. Allowed shapes are (L, ) for a single trace with L datapoints
        or (N, L) for a collection of N traces with L datapoints each. The latter case
        can be used for instance to analyze scans taken at different magnetic fields
        at once. In the first case, the input array is first converted to a 2D array
        with shape (1, L).
    xi, xf: float
        Horizontal range of the data. Used to determine Fourier frequencies in
        the correct physical units.
    mph, md: float
        Thresholds for peak height and distance from neighboring peaks.

    Returns:
    --------
    peak_indices:
        list of peak indices for each of the N data traces.
    peak_positions:
        list of peak position along the segment (xi, xf) for each
        of the N data traces
    peak_spacings:
        list of peak spacing for each of the N data traces.
    """
    if len(zs.shape) == 1:
        dx = abs(xf - xi) / len(zs)
        peak_indices = detect_peaks(zs, mph, md)
        peak_positions = xi + dx * peak_indices
        peak_spacings = np.diff(peak_positions)
    else:
        # zs = zs.reshape(1, zs.shape[0]) if len(zs.shape) == 1 else zs
        dx = abs(xf - xi) / zs.shape[1]
        peak_indices = [detect_peaks(z, mph, md) for z in zs]
        peak_positions = [xi + dx * idxs for idxs in peak_indices]
        peak_spacings = [np.diff(pos) for pos in peak_positions]

    return peak_indices, peak_positions, peak_spacings


def peak_coordinates(peak_positions, Bs):
    """
    Given a list of peaks corresponding to different magnetic fields,
    returns coordinates (B, x) of the peak.
    """
    coord_list = []
    for (n, B) in enumerate(Bs):
        for pos in peak_positions[n]:
            coord_list.append((pos, B))
    return coord_list


def spacings_histograms(peak_spacings, bins, rng):
    histograms = []
    for ps in peak_spacings:
        hist, bin_edges = np.histogram(ps, bins, rng)
        histograms.append(hist)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_centers, np.vstack(histograms)


def average_spacings(positions, xi, xf, nbins, wl, max_spacing, staggered=False):
    """
    Returns average spacing between peaks in a window of a certain length.
    """
    xcenters = np.linspace(xi + wl, xf - wl, nbins)
    average_spacings = np.zeros((len(positions), nbins))
    for (i, pks) in enumerate(positions):
        for (j, xc) in enumerate(xcenters):
            local_peaks = pks[np.where(np.logical_and(pks >= xc - wl, pks <= xc + wl))[0]]
            local_spacings = np.diff(local_peaks)
            if staggered:
                avg_spacing = (max_spacing if local_peaks.size < 2
                               else np.average(local_spacings[::2]) - np.average(local_spacings[1::2]))
                average_spacings[i, j] = avg_spacing if avg_spacing < max_spacing else np.nan
            else:
                avg_spacing = max_spacing if local_peaks.size < 2 else np.average(np.diff(local_peaks))
                average_spacings[i, j] = avg_spacing if avg_spacing < max_spacing else np.nan
    return average_spacings


def even_odd_spacings(zs, ys, xi, xf, mph, md, conditions, remove_ys=True, num_traces=4, plot=True):
    idxs, positions, spacings = peak_data(zs, xi, xf, mph, md)
    coords = peak_coordinates(positions, ys)
    for cond in conditions:
        coords = [c for c in coords if cond(c)]

    # Asks that the number of peaks is equal to that at the highest value of fields.
    # This makes sure the number of peaks at each value of field is constant.
    yvals, ycounts = np.unique(list(zip(*coords))[1], return_counts=True)
    good_ys = [y for (n, y) in enumerate(yvals) if ycounts[n] == num_traces] if remove_ys else yvals
    coords = [c for c in coords if c[1] in good_ys]

    # reorders peak positions by field.
    positions = [[c[0] for c in coords if c[1] == y] for y in good_ys]

    # returns even and odd spacings separately
    even_spacings = []
    odd_spacings = []
    for p in positions:
        spacings = np.diff(p)
        odd_spacings.append(spacings[::2])  # it's assumed first spacing corresponds to odd valley
        even_spacings.append(spacings[1::2])

    if plot:
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        im = ax.matshow(zs, aspect='auto', extent=[xi, xf, np.min(ys), np.max(ys)])
        xs, fs = list(zip(*coords))
        ax.scatter(xs, fs, c='b', marker='.')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        plt.colorbar(im, cax=cax)

    if remove_ys:
        odd_spacings = np.vstack(odd_spacings)
        even_spacings = np.vstack(even_spacings)


    return coords, good_ys, odd_spacings, even_spacings