"""Reference signal analysis for the simulator.

This is the NumPy side of the contract that `engine/src/harmonic_features.cpp`
must reproduce in fixed point: the same definitions, computed the obvious way,
so the C implementation has something to be checked against.
"""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def crest_factor(x: np.ndarray) -> float:
    r = rms(x)
    return float(peak(x) / r) if r > 0 else 0.0


def kurtosis(x: np.ndarray) -> float:
    """Non-excess kurtosis: 3.0 for a Gaussian signal.

    Impulsive faults — a bearing spall, cavitation implosions — raise this
    well before they move the RMS, which is why it earns a feature slot.
    """
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma == 0.0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4))


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum with a Hann window.

    Amplitudes are corrected for the window's coherent gain, so a pure tone
    of amplitude A reads back as A.
    """
    n = len(x)
    # Periodic Hann (denominator n), matching harmonic_features.cpp. NumPy's
    # np.hanning is the symmetric form (denominator n-1); the difference is
    # small but it is a difference, and this file is the reference the C
    # extractor is held to.
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    coherent_gain = float(np.mean(win))
    spec = np.fft.rfft(x * win)
    amp = np.abs(spec) * 2.0 / (n * coherent_gain)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amp


def amplitude_at(freqs: np.ndarray, amp: np.ndarray, target_hz: float,
                 tolerance_hz: float = 2.0) -> float:
    """Peak amplitude within +-tolerance of a target frequency.

    A tolerance band rather than the single nearest bin: shaft speed drifts,
    and a defect frequency is only ever approximately where the geometry says.
    """
    lo, hi = target_hz - tolerance_hz, target_hz + tolerance_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.max(amp[mask]))


def envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope via the analytic signal.

    Bearing defects are impulses that ring the structure at a resonance far
    above the defect rate. The defect rate shows up in how those impulses are
    *modulated*, not in the raw spectrum — so demodulate first, then transform.
    """
    n = len(x)
    spec = np.fft.fft(x - np.mean(x))
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spec * h)
    return np.abs(analytic)


def envelope_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Spectrum of the amplitude envelope, with the DC component removed."""
    env = envelope(x)
    return spectrum(env - np.mean(env), fs)
