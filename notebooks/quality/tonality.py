from fractions import Fraction
from math import floor

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, hilbert, welch
from scipy.signal.windows import gaussian

from filters import butter_lp, moving_avg


#
# "Fundamental Tone" quality measure
#
def last_peak(sample):
    pks, _ = find_peaks(sample, width=64, height=5e-2)
    if len(pks) < 2:
        return 0
    else:
        return pks[len(pks) - 1]


def gauss_filt(signal, N=601, alpha=2):
    win = gaussian(N, alpha)
    win = win / np.sum(win)
    return np.convolve(signal, win)


def hilbert_analysis(signal, save_fig=None):
    """
    Attempts to find the fundamental frequency of a kick drum signal.
    If found, returns the frequency in Hz.
    If not found, return 0 Hz.
    It is expected to return 0 if given signal is not that of a kick drum.
    """
    signal = butter_lp(signal, 500)
    last_pk = last_peak(signal)
    if last_pk == 0:
        return 0
    try:
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        if not save_fig is None:
            plt.plot(signal)
            plt.plot(amplitude_envelope)
            plt.title("Amplitude envelope A(n)")
            plt.savefig(save_fig + "A")
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (
            gauss_filt(np.diff(instantaneous_phase)) / (2.0 * np.pi) * 44100
        )
        if not save_fig is None:
            plt.plot(instantaneous_frequency)
            plt.title("Instantaneous Frequency (IF)")
            plt.savefig(save_fig + "IF")
        pitch_env = moving_avg(instantaneous_frequency[0:last_pk], 256)
        pk_pitch = np.argmax(pitch_env)
        pitch_env = pitch_env[pk_pitch:last_pk]
        argmin = np.argmin(pitch_env)
        if argmin >= 500:
            slc = pitch_env[argmin - 500 : argmin + 500]
        else:
            slc = pitch_env[25:1025]
        note_hz = np.mean(slc)
        if not save_fig is None:
            plt.plot(pitch_env)
            plt.title("Approximated Pitch envelope")
            plt.savefig(save_fig + "P")
        return note_hz
    except:
        return 0


def frq_qual(sample):
    hz = hilbert_analysis(sample)
    if hz <= 20:
        return 1
    elif hz >= 100:
        return hz / 100
    else:
        interp = interp1d([20, 25, 65, 100], [1, 0, 0, 1], kind="linear")
        return max(0, float(interp(hz)))


#
# "Inharmonic Distortion" quality measure.
#
# def w(ratio):
#     r = ratio - floor(ratio)
#     f = Fraction(r).limit_denominator(100)
#     return f.denominator


def ihd_qual(sample, nfft=8192):
    end = last_peak(sample)
    start = end // 5
    sample = sample[start:end]
    freqs, psd = welch(
        sample, nfft=nfft, fs=44100, detrend="linear", nperseg=min(nfft, len(sample))
    )
    peaks, _ = find_peaks(psd, height=1e-9, prominence=2e-9, width=1.2)
    if len(peaks) == 0:
        return 0  # if there are no peaks, all frequencies are masked, and so there is only noise and no tonal content.
    f1 = np.argmax(psd[peaks])
    fund = freqs[peaks[f1]]
    ratios = freqs[peaks[f1 + 1 :]] / fund
    return np.sqrt(np.sum(psd[peaks[f1 + 1 :]] ** 2)) / psd[peaks[f1]]
