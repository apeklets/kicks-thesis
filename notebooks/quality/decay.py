import matplotlib.pyplot as plt
import numpy as np
from librosa.effects import trim
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks

from filters import butter_lp, moving_avg


def sigmoid(x, L, x0, k):
    """Sigmoid curve definition"""
    y = L / (1 + np.exp(-k * (x - x0)))
    return y


def ls_func(popt, x, y):
    """Function for least squares to minimize"""
    l = sigmoid(x, *popt)
    return l - y


def env_fit(sample, lp=True, fev=10000, save_fig=None):
    """Find converging least squares fit, or return 0."""
    if lp:
        mv = butter_lp(sample, 1000)
    else:
        mv = moving_avg(np.abs(sample), 8)
    # find peaks
    xpeaks, _ = find_peaks(mv, width=64, height=1e-3)
    ypeaks = np.abs(mv[xpeaks])
    # fit curve
    p0 = [max(ypeaks), np.median(xpeaks), -0.001]
    lbounds = [0, -np.inf, -5]
    ubounds = [1e05, len(sample), 0]
    if lp:
        fit_start = 0
    else:
        fit_start = np.argmax(ypeaks[0:6])
    try:
        popt, pcov = curve_fit(
            sigmoid,
            xpeaks[fit_start:],
            ypeaks[fit_start:],
            p0,
            method="dogbox",
            maxfev=fev,
            bounds=(lbounds, ubounds),
        )
    except:
        return 0
    # save a figure of the fit
    if not save_fig is None:
        x = range(0, len(sample))
        plt.plot(mv)
        plt.plot(xpeaks, mv[xpeaks], "x")
        plt.plot(x, sigmoid(x, *popt))
        plt.savefig(save_fig)
    # compute mean squared error
    l = sigmoid(xpeaks[fit_start:], *popt)
    return np.mean(np.square(l - ypeaks[fit_start:]))


def env_qual(sample, fev=1000):
    """Perform least squares for at most 10000 iterations, and return the best result found."""
    # lowpass at 1000 Hz
    mv = butter_lp(sample, 1000)
    # find peaks
    xpeaks, _ = find_peaks(np.abs(mv), width=64, height=1e-2)
    if len(xpeaks) == 0:
        # we are dealing with a noisy signal, so there are no peaks
        # just use evenly spaced samples to evaluate the envelope
        # expect a high MSE from this
        xpeaks = np.arange(0, len(mv), 1000)
    ypeaks = np.abs(mv[xpeaks])
    # fit curve
    p0 = [max(ypeaks), np.median(xpeaks), -0.001]
    lbounds = [0, -np.inf, -5]
    ubounds = [1e05, len(sample), 0]
    result = least_squares(
        ls_func,
        p0,
        bounds=(lbounds, ubounds),
        method="dogbox",
        kwargs={"x": xpeaks, "y": ypeaks},
        max_nfev=fev,
    )
    return 2 * result.cost / len(xpeaks)


def rms_qual(sample):
    """Quality based on the total RMS value of the sample. """
    sample = trim(sample, top_db=36)[0]  # trim a little higher
    rms = np.sqrt(np.mean(sample ** 2))
    return 1 - rms  # rms = 1 is the "optimal" rms (which would be a square wave)
