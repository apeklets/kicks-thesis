import librosa
import numpy as np
from scipy.signal import lfilter

from filters import K_weighting


def spectral_slope(sample):
    kfilt_b, kfilt_a = K_weighting()
    # sample = librosa.effects.trim(sample, top_db=40)[0]
    sample = lfilter(kfilt_b, kfilt_a, sample[0:2205])  # only grab first 50ms
    cq = np.abs(librosa.cqt(sample, sr=44100, fmin=30.5, n_bins=114))
    spectrum = 20 * np.log10(np.mean(cq, 1))
    x = np.arange(0, 114)
    linear_model, mse, _, _, _ = np.polyfit(x, spectrum, 1, full=True)
    slope = linear_model[0]
    mse = mse[0] / len(spectrum)
    return slope, mse


def res_qual(sample):
    (slope, mse) = spectral_slope(sample)
    slope += 0.4  # offset by 0.4, a slope of -0.4 gets the best quality
    if slope < 0:
        slope = -1 * slope
    else:
        slope = 2 * slope  # punish upward slopes, could smooth this towards 0
    return slope + mse / 250
