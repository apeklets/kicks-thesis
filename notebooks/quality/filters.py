"""
Various signal processing and statistical utility functions used in multiple parts of the quality measure.
"""
from math import pi, tan

import numpy as np
from scipy.signal import butter, lfilter, sosfilt


def rms(sample):
    return np.sqrt(np.mean(sample ** 2))


def moving_avg(sample, n):
    """Simple convolution based moving average"""
    ir = np.ones(n) / n
    return np.convolve(sample, ir)


def butter_lp(data, cutoff, fs=44100, order=4):
    """Simple Butterworth lowpass filter"""
    ncut = 2 * cutoff / fs
    b, a = butter(order, ncut, btype="low", analog=False)
    y = lfilter(b, a, data)
    return y


def butter_hp(data, cutoff, fs=44100, order=4):
    """Simple Butterworth lowpass filter"""
    ncut = 2 * cutoff / fs
    sos = butter(order, ncut, btype="hp", analog=False, output="sos")
    y = sosfilt(sos, data)
    return y


# Adapted from the C++ code in https://github.com/jiixyj/libebur128/blob/v1.0.2/ebur128/ebur128.c#L82
def K_weighting(sampleRate=44100):
    """Calculate K-weighting filter coefficients for given samplerate"""
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196

    K = tan(pi * f0 / sampleRate)
    Vh = pow(10.0, G / 20.0)
    Vb = pow(Vh, 0.4996667741545416)
    a0 = 1.0 + K / Q + K * K

    filterB1 = np.empty(3)
    filterA1 = np.empty(3)
    filterB2 = np.empty(3)
    filterA2 = np.empty(3)

    filterB1[0] = (Vh + Vb * K / Q + K * K) / a0
    filterB1[1] = 2.0 * (K * K - Vh) / a0
    filterB1[2] = (Vh - Vb * K / Q + K * K) / a0

    filterA1[0] = 1.0
    filterA1[1] = 2.0 * (K * K - 1.0) / a0
    filterA1[2] = (1.0 - K / Q + K * K) / a0

    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = tan(pi * f0 / sampleRate)

    filterB2[0] = 1.0
    filterB2[1] = -2.0
    filterB2[2] = 1.0

    filterA2[0] = 1.0
    filterA2[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    filterA2[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)

    # combine two filters into one
    filterB = np.empty(5)
    filterA = np.empty(5)

    filterB[0] = filterB1[0] * filterB2[0]
    filterB[1] = filterB1[0] * filterB2[1] + filterB1[1] * filterB2[0]
    filterB[2] = (
        filterB1[0] * filterB2[2]
        + filterB1[1] * filterB2[1]
        + filterB1[2] * filterB2[0]
    )
    filterB[3] = filterB1[1] * filterB2[2] + filterB1[2] * filterB2[1]
    filterB[4] = filterB1[2] * filterB2[2]

    filterA[0] = filterA1[0] * filterA2[0]
    filterA[1] = filterA1[0] * filterA2[1] + filterA1[1] * filterA2[0]
    filterA[2] = (
        filterA1[0] * filterA2[2]
        + filterA1[1] * filterA2[1]
        + filterA1[2] * filterA2[0]
    )
    filterA[3] = filterA1[1] * filterA2[2] + filterA1[2] * filterA2[1]
    filterA[4] = filterA1[2] * filterA2[2]

    return filterB, filterA
